# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.40", "httpx>=0.27", "pypdfium2>=4.30", "pillow>=11"]
# ///
"""HTTP-tunneled vLLM OCR client.

Architecture (per worker):
  1. SSH connection #1 (long-lived): runs
     `srun ... bash hpc/slurm/vllm_serve_apptainer.slurm`.
     vLLM listens on a random high port on the GPU compute node and prints
     "VLLM_LISTEN host=cNNN port=NNNNN api_key=HHHHH" to stderr early.
  2. We parse that marker from the SSH stderr stream.
  3. SSH connection #2 (long-lived, -N): tunnels localhost:LP → compute:NNNNN
     via the cluster login node. Auto-picks a free local port.
  4. We send PDF pages as base64-encoded image_url content blocks to
     localhost:LP/v1/chat/completions with the `openai` async client. Many
     pages can be in flight; vLLM internally batches.

Disk:
  - On cluster: nothing. vLLM holds requests in memory only. Slurm log on
    the trusted local client (we drive via ssh+srun, not sbatch).
  - On the trusted local client: PDFs read from disk, markdown written via
    atomic temp+rename.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import concurrent.futures
import io
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from PIL import Image
from openai import AsyncOpenAI

DEFAULT_HOST = os.environ.get("HPC_HOST", "hpc")
DEFAULT_USER = os.environ.get("HPC_USER") or os.environ.get("USER", "")
DEFAULT_KEY = os.path.expanduser(os.environ.get("HPC_KEY", ""))
DEFAULT_REMOTE_DIR = os.environ.get("HPC_REMOTE_DIR", "ocr-examples")

# Defaults preserve historical olmOCR-2 behavior. Pass --model and
# --prompt-file (or --prompt) to switch the served model and the per-page
# instruction text. The slurm-side env var VLLM_MODEL is set from --model
# in launch_vllm_serve.
DEFAULT_MODEL_ID = "allenai/olmOCR-2-7B-1025"
DEFAULT_PROMPT = (
    "Below is one page of a document. Reproduce its content as plain "
    "text + markdown, preserving every detail faithfully:\n"
    "- Tables with all rows and columns\n"
    "- Labels, headings, stamps, handwriting, marks, and symbols where visible\n"
    "- Math/equations as LaTeX where appropriate\n"
    "- Legends, scale definitions, narrative remarks, and footnotes — full text\n"
    "Do not summarize, paraphrase, or omit information. Skip pure page-number/header/footer chrome."
)

LISTEN_RE = re.compile(r"VLLM_LISTEN\s+host=(\S+)\s+port=(\d+)\s+api_key=(\S+)")
SLURM_JOB_RE = re.compile(r'"event":"job_start".*"job":"([^"}]+)"')


def ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}" if user else host


def ssh_base_args(ssh_key: str) -> list[str]:
    args: list[str] = []
    if ssh_key:
        args.extend(["-i", ssh_key, "-o", "IdentitiesOnly=yes"])
    return args


# ---- rendering ----

def render_pages(pdf_bytes: bytes, scale: float) -> list[Image.Image]:
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    pages: list[Image.Image] = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            pages.append(page.render(scale=scale).to_pil())
            page.close()
    finally:
        doc.close()
    return pages


def image_to_data_url(img: Image.Image, jpeg_quality: int = 85) -> str:
    """Encode a rendered page as a base64 data URL.

    JPEG is ~5x faster to encode than PNG and ~3x smaller, with no observable
    OCR-quality loss at q=85 for typical document scans. PIL needs RGB (not
    RGBA) for JPEG, so we explicitly convert.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---- SSH plumbing ----

@dataclass
class ServerEndpoint:
    compute_host: str
    compute_port: int
    api_key: str
    local_port: int = 0
    slurm_job_id: str | None = None


async def launch_vllm_serve(
    ssh_host: str, ssh_user: str, ssh_key: str, remote_dir: str,
    partition: str, gres: str, cpus_per_task: int, mem: str, time_str: str,
    job_name: str,
    model_id: str, max_model_len: int,
    slurm_script: str, image: str | None,
    pip_deps: str | None = None,
    patch_mrope: bool = False,
    extra_env: dict[str, str] | None = None,
    exclude: str | None = None,
) -> tuple[asyncio.subprocess.Process, ServerEndpoint]:
    """Open SSH #1, launch the chosen slurm_script, return process + parsed endpoint.

    Blocks until VLLM_LISTEN marker is parsed (i.e. vLLM has bound its port and
    started loading; model weights may still be loading at this point).

    model_id, max_model_len, and image are forwarded to the slurm script via
    env vars (VLLM_MODEL, VLLM_MAX_MODEL_LEN, VLLM_IMAGE). GPU serving is
    containerized; image is required.
    """
    # env vars must travel through srun's environment — they reach the slurm
    # script which reads them. We do NOT log model_id here as it may be
    # sensitive in some setups; the slurm script echoes its own model_select
    # event back through stderr.
    env_parts = [
        f"VLLM_MODEL={shlex.quote(model_id)}",
        f"VLLM_MAX_MODEL_LEN={int(max_model_len)}",
    ]
    if not image:
        raise ValueError("--image is required for the containerized vLLM service")
    env_parts.append(f"VLLM_IMAGE={shlex.quote(image)}")
    if pip_deps:
        env_parts.append(f"VLLM_PIP_DEPS={shlex.quote(pip_deps)}")
    if patch_mrope:
        env_parts.append("VLLM_PATCH_MROPE=1")
    if extra_env:
        for name, value in sorted(extra_env.items()):
            if not re.match(r"^[A-Z_][A-Z0-9_]*$", name):
                raise ValueError(f"bad environment variable name: {name!r}")
            env_parts.append(f"{name}={shlex.quote(str(value))}")
    # Forward VLLM_TP / VLLM_ENFORCE_EAGER from our env if set. SSH doesn't
    # forward arbitrary env vars by default, so we have to name them here.
    for var in ("VLLM_TP", "VLLM_ENFORCE_EAGER"):
        val = os.environ.get(var)
        if val is not None:
            env_parts.append(f"{var}={shlex.quote(val)}")
    env_prefix = " ".join(env_parts) + " "
    # --export=ALL,FOO,BAR makes ALL inherited vars + the explicitly named
    # vars reach the job (including ones we just set, which are not yet in
    # the inherited environment).
    export_vars = ["VLLM_MODEL", "VLLM_MAX_MODEL_LEN"]
    export_vars.append("VLLM_IMAGE")
    if pip_deps:
        export_vars.append("VLLM_PIP_DEPS")
    if patch_mrope:
        export_vars.append("VLLM_PATCH_MROPE")
    if extra_env:
        export_vars.extend(sorted(extra_env))
    for var in ("VLLM_TP", "VLLM_ENFORCE_EAGER"):
        if os.environ.get(var) is not None:
            export_vars.append(var)
    export_arg = "ALL," + ",".join(export_vars)
    exclude_arg = f"--exclude={shlex.quote(exclude)} " if exclude else ""
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"{env_prefix}"
        f"srun "
        f"--partition={partition} "
        f"{exclude_arg}"
        f"--gres={gres} "
        f"--cpus-per-task={cpus_per_task} "
        f"--mem={mem} "
        f"--time={time_str} "
        f"--job-name={shlex.quote(job_name)} "
        f"--export={shlex.quote(export_arg)} "
        f"bash {shlex.quote(slurm_script)}"
    )
    cmd = [
        "ssh", *ssh_base_args(ssh_key), "-T",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "BatchMode=yes",
        ssh_target(ssh_user, ssh_host),
        remote_cmd,
    ]
    print(f"[serve] launching: ssh ... '{remote_cmd[:120]}…'", file=sys.stderr, flush=True)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,  # vLLM doesn't write to stdout (only stderr)
        stderr=asyncio.subprocess.PIPE,
    )

    endpoint: Optional[ServerEndpoint] = None
    slurm_job_id: str | None = None
    # generous: includes srun queue wait + model load. On busy days slurm
    # may take 15-25 min to allocate a GPU.
    timeout_s = int(os.environ.get("HPC_LAUNCH_TIMEOUT_S", "1800"))
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            line_bytes = await asyncio.wait_for(proc.stderr.readline(), timeout=10)
        except asyncio.TimeoutError:
            if proc.returncode is not None:
                raise RuntimeError(f"ssh exited with rc={proc.returncode} before LISTEN")
            continue
        if not line_bytes:
            raise RuntimeError("ssh stderr closed before LISTEN marker")
        line = line_bytes.decode("utf-8", "replace").rstrip("\n")
        print(f"[serve] {line}", file=sys.stderr, flush=True)
        job_match = SLURM_JOB_RE.search(line)
        if job_match:
            slurm_job_id = job_match.group(1)
        m = LISTEN_RE.search(line)
        if m:
            endpoint = ServerEndpoint(
                compute_host=m.group(1),
                compute_port=int(m.group(2)),
                api_key=m.group(3),
                slurm_job_id=slurm_job_id,
            )
            break

    if endpoint is None:
        proc.terminate()
        raise RuntimeError("did not see VLLM_LISTEN marker within timeout")

    # Keep pumping stderr in the background so the SSH process doesn't block
    # on its stderr buffer (and so we see vLLM logs as they happen).
    async def pump_remaining_stderr():
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                # Sanitize: drop anything that might contain the api_key from
                # later log lines. We already captured it; suppress further
                # occurrences just in case.
                txt = line.decode("utf-8", "replace").rstrip("\n")
                if endpoint.api_key in txt:
                    txt = txt.replace(endpoint.api_key, "<api-key>")
                print(f"[serve] {txt}", file=sys.stderr, flush=True)
        except Exception:
            pass

    asyncio.create_task(pump_remaining_stderr())
    return proc, endpoint


async def open_tunnel(
    ssh_host: str, ssh_user: str, ssh_key: str, endpoint: ServerEndpoint,
) -> asyncio.subprocess.Process:
    """Open SSH #2: ssh -N -L localhost:0:compute:port login.node.

    Returns the SSH process. endpoint.local_port is filled in with the auto-
    picked local port (we ask the kernel by binding to 0 ourselves first,
    then immediately freeing it — there's a TOCTOU window but in practice
    it works fine for tooling).
    """
    # Pick a free local port by binding 0 then releasing. Small race but the
    # alternative (ssh -L 0:...) requires parsing ssh's "Allocated port" which
    # depends on -v output.
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    local_port = s.getsockname()[1]
    s.close()
    endpoint.local_port = local_port

    cmd = [
        "ssh", *ssh_base_args(ssh_key), "-N",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-L", f"localhost:{local_port}:{endpoint.compute_host}:{endpoint.compute_port}",
        ssh_target(ssh_user, ssh_host),
    ]
    print(
        f"[tunnel] localhost:{local_port} -> {endpoint.compute_host}:{endpoint.compute_port}",
        file=sys.stderr, flush=True,
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    # We deliberately swallow the noisy "channel N: open failed: connect failed:
    # Connection refused" lines that occur while vLLM is still loading. They're
    # expected (the openai client polls /v1/models) and they otherwise flood
    # the output. Other tunnel messages still pass through.
    NOISE_PATTERNS = (
        b"open failed: connect failed: Connection refused",
        b"** WARNING: connection is not using a post-quantum",
        b"** This session may be vulnerable to",
        b"** The server may need to be upgraded",
    )

    async def pump():
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                if any(p in line for p in NOISE_PATTERNS):
                    continue
                print(f"[tunnel] {line.decode('utf-8', 'replace').rstrip()}",
                      file=sys.stderr, flush=True)
        except Exception:
            pass

    asyncio.create_task(pump())
    return proc


# ---- HTTP request ----

async def wait_for_ready(client: AsyncOpenAI, model_id: str, deadline_s: float) -> None:
    """Poll /v1/models until vLLM is serving (i.e. model weights loaded)."""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            r = await client.models.list()
            for m in r.data:
                if m.id == model_id:
                    elapsed = time.time() - t0
                    print(f"[ready] vLLM serving {model_id} after {elapsed:.1f}s",
                          file=sys.stderr, flush=True)
                    return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise RuntimeError(f"vLLM didn't become ready within {deadline_s}s")


async def ocr_page(client: AsyncOpenAI, data_url: str, model_id: str, prompt: str, max_tokens: int) -> str:
    resp = await client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


@dataclass
class PdfJob:
    in_pdf: Path
    out_md: Path
    n_pages: int = 0
    n_chars: int = 0
    ok: bool = False
    elapsed_s: float = 0.0
    error: Optional[str] = None


def render_and_encode(pdf_bytes: bytes, scale: float, jpeg_quality: int) -> list[str]:
    """Render every page and encode as JPEG data URLs, in one executor call.

    Keeps the render+encode work off the asyncio loop so the loop can keep
    pumping network I/O while pages get prepared.
    """
    pages = render_pages(pdf_bytes, scale)
    return [image_to_data_url(p, jpeg_quality) for p in pages]


async def ocr_pdf(
    client: AsyncOpenAI, job: PdfJob, scale: float, max_tokens: int,
    pdf_sem: asyncio.Semaphore,
    render_executor: concurrent.futures.Executor,
    jpeg_quality: int,
    model_id: str, prompt: str,
) -> PdfJob:
    """Render → submit all pages concurrently → join markdown → write atomically.

    Rendering+encoding goes through a single-threaded executor because
    pypdfium2's underlying pdfium library is not thread-safe — concurrent
    renders SIGABRT in native code.
    """
    async with pdf_sem:
        t0 = time.time()
        try:
            pdf_bytes = job.in_pdf.read_bytes()
            loop = asyncio.get_running_loop()
            data_urls = await loop.run_in_executor(
                render_executor, render_and_encode, pdf_bytes, scale, jpeg_quality,
            )
            if not data_urls:
                job.error = "no pages"
                job.elapsed_s = time.time() - t0
                return job
            page_texts = await asyncio.gather(
                *[ocr_page(client, du, model_id, prompt, max_tokens) for du in data_urls]
            )
            parts = [f"<!-- page {i+1} -->\n{t}" for i, t in enumerate(page_texts)]
            markdown = "\n\n".join(parts) + "\n"
            payload = markdown.encode("utf-8")

            job.out_md.parent.mkdir(parents=True, exist_ok=True)
            tmp = job.out_md.with_suffix(job.out_md.suffix + ".tmp")
            tmp.write_bytes(payload)
            tmp.replace(job.out_md)

            job.n_pages = len(data_urls)
            job.n_chars = len(markdown)
            job.ok = True
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
        job.elapsed_s = time.time() - t0
        return job


# ---- multi-worker orchestration ----

@dataclass
class Worker:
    idx: int
    serve_proc: asyncio.subprocess.Process
    tunnel_proc: asyncio.subprocess.Process
    client: AsyncOpenAI
    endpoint: ServerEndpoint
    ssh_host: str
    ssh_user: str
    ssh_key: str


async def launch_one_worker(args: argparse.Namespace, idx: int) -> Worker:
    """Launch one Slurm job + tunnel + openai client for it.

    Returns after the worker's vLLM is fully ready to serve requests.
    """
    print(f"[orch] launching worker w{idx}…", file=sys.stderr, flush=True)
    serve_proc, endpoint = await launch_vllm_serve(
        ssh_host=args.host, ssh_user=args.user, ssh_key=args.key,
        remote_dir=args.remote_dir,
        partition=args.partition, gres=args.gres,
        cpus_per_task=args.cpus_per_task, mem=args.mem, time_str=args.time,
        job_name=f"vllm-serve-w{idx}-{os.getpid()}",
        model_id=args.model, max_model_len=args.max_model_len,
        slurm_script=args.slurm_script, image=args.image,
        pip_deps=args.pip_deps, patch_mrope=args.patch_mrope,
        exclude=args.exclude,
    )
    tunnel_proc = await open_tunnel(args.host, args.user, args.key, endpoint)
    await asyncio.sleep(1.0)
    client = AsyncOpenAI(
        base_url=f"http://localhost:{endpoint.local_port}/v1",
        api_key=endpoint.api_key,
        max_retries=4,
        timeout=600,
    )
    # 1800s covers first-time Apptainer SIF unpack (~10 min for multi-GB
    # vllm-openai images) + HF model weight download + vLLM cold start.
    # Cached runs are sub-2-minute; the headroom is mostly for new images
    # or models that haven't been pulled to /gpfs/scratch60 yet. Override
    # via HPC_LAUNCH_TIMEOUT_S env var if you want it tighter.
    ready_deadline = int(os.environ.get("HPC_LAUNCH_TIMEOUT_S", "1800"))
    await wait_for_ready(client, model_id=args.model, deadline_s=ready_deadline)
    print(f"[orch] w{idx} ready (port {endpoint.local_port} → {endpoint.compute_host}:{endpoint.compute_port})",
          file=sys.stderr, flush=True)
    return Worker(
        idx=idx, serve_proc=serve_proc, tunnel_proc=tunnel_proc,
        client=client, endpoint=endpoint,
        ssh_host=args.host, ssh_user=args.user, ssh_key=args.key,
    )


async def cancel_slurm_job(ssh_host: str, ssh_user: str, ssh_key: str, job_id: str | None) -> None:
    """Best-effort cleanup for ssh-driven srun jobs.

    Killing the local ssh process does not always tear down the remote Slurm
    allocation immediately. scancel by the job id emitted by the Slurm script
    prevents orphan GPU jobs from waiting out their --time limit.
    """
    if not job_id or not re.match(r"^[0-9]+$", job_id):
        return
    cmd = [
        "ssh", *ssh_base_args(ssh_key), "-T",
        "-o", "BatchMode=yes",
        ssh_target(ssh_user, ssh_host),
        f"scancel {shlex.quote(job_id)}",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode not in (0, None):
            msg = (stderr or b"").decode("utf-8", "replace").strip()
            print(f"[orch] scancel {job_id} rc={proc.returncode}: {msg}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[orch] scancel {job_id} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


async def teardown_worker(w: Worker) -> None:
    await cancel_slurm_job(w.ssh_host, w.ssh_user, w.ssh_key, w.endpoint.slurm_job_id)
    for p in (w.tunnel_proc, w.serve_proc):
        try:
            p.terminate()
        except Exception:
            pass
    await asyncio.gather(
        *(asyncio.wait_for(p.wait(), timeout=20) for p in (w.tunnel_proc, w.serve_proc)),
        return_exceptions=True,
    )


def parse_pdf_list(list_path: Path, default_out_dir: Path) -> list[tuple[Path, Path]]:
    """Parse --pdf-list. Each non-empty line is either:
      - "<pdf_path>"                   → output goes to default_out_dir/<stem>.md
      - "<pdf_path>\t<out_md_path>"    → output goes to the explicit path

    The TSV form lets callers (e.g. scripts/olmocr2_extract.py --use-hpc)
    place outputs into per-stream dirs without symlinks or post-move.
    """
    pairs: list[tuple[Path, Path]] = []
    for line in list_path.read_text().splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if "\t" in line:
            in_str, out_str = line.split("\t", 1)
            pairs.append((Path(in_str.strip()), Path(out_str.strip())))
        else:
            in_p = Path(line.strip())
            pairs.append((in_p, default_out_dir / (in_p.stem + ".md")))
    return pairs


async def main_async_keep_alive(args: argparse.Namespace) -> int:
    """Launch one worker + tunnel, print READY marker to stdout, block on
    stdin EOF. For consumers that want to drive the served model with their
    own client rather than the built-in PDF queue."""
    print(f"[orch] launching keep-alive tunnel for model {args.model}",
          file=sys.stderr, flush=True)
    worker = await launch_one_worker(args, 0)

    # Emit the discovery line on stdout so the parent process can parse it.
    # api_key intentionally on the line — caller already trusts our stdout.
    base_url = f"http://localhost:{worker.endpoint.local_port}/v1"
    print(f"READY base_url={base_url} api_key={worker.endpoint.api_key} model={args.model}",
          flush=True)
    print(f"[orch] tunnel ready; waiting for stdin EOF to teardown",
          file=sys.stderr, flush=True)

    # Block until stdin closes (parent process exits or explicitly closes
    # our stdin). asyncio.StreamReader on sys.stdin is the cleanest portable
    # approach.
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    try:
        # Reads block until EOF or new data. We don't care what comes in;
        # we only care when the stream closes.
        while True:
            data = await reader.read(4096)
            if not data:
                break
    except Exception:
        pass

    print("[orch] stdin closed; tearing down worker", file=sys.stderr, flush=True)
    await teardown_worker(worker)
    return 0


async def main_async(args: argparse.Namespace) -> int:
    if args.keep_alive:
        return await main_async_keep_alive(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_pdf_list(args.pdf_list, args.out_dir)
    if not pairs:
        sys.exit("--pdf-list contained no paths")
    missing = [in_p for in_p, _ in pairs if not in_p.exists()]
    if missing:
        sys.exit(f"missing PDFs: {[str(p) for p in missing[:5]]}…")

    jobs = [
        PdfJob(in_pdf=in_p, out_md=out_md)
        for in_p, out_md in pairs
        if args.force or not out_md.exists()
    ]
    print(
        f"[orch] {len(jobs)} PDFs to process; "
        f"workers={args.workers}, in-flight={args.in_flight} per worker "
        f"(total concurrent: {args.workers * args.in_flight})",
        file=sys.stderr, flush=True,
    )
    if not jobs:
        return 0

    # Launch N workers in parallel. Each blocks until its vLLM is ready.
    # A worker may legitimately fail/time-out if Slurm can't allocate its GPU
    # within HPC_LAUNCH_TIMEOUT_S. As long as ≥1 worker succeeds, proceed —
    # better to OCR slowly than to abandon a several-hour batch.
    workers = await asyncio.gather(
        *(launch_one_worker(args, i) for i in range(args.workers)),
        return_exceptions=True,
    )
    bad = [w for w in workers if isinstance(w, BaseException)]
    live = [w for w in workers if isinstance(w, Worker)]
    if bad:
        for b in bad:
            print(f"[orch] worker launch failure: {b!r}", file=sys.stderr, flush=True)
    if not live:
        print("[orch] NO workers came up; aborting", file=sys.stderr, flush=True)
        return 2
    if bad:
        print(
            f"[orch] proceeding with {len(live)}/{args.workers} worker(s)",
            file=sys.stderr, flush=True,
        )
    workers = live

    # Shared work queue. Producers (none, we pre-fill) + N×in-flight consumers.
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)

    # pdfium isn't thread-safe; single-threaded render executor shared by ALL
    # workers (in one Python process).
    render_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="pdfium",
    )

    t_start = time.time()
    completed: list[PdfJob] = []
    # No-op semaphore from ocr_pdf's perspective: per-consumer concurrency is
    # bounded by the consumer count, not by the semaphore. Pass a permissive
    # one and let consumers limit naturally.
    dummy_sem = asyncio.Semaphore(args.workers * args.in_flight)

    async def consumer(w: Worker) -> None:
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            j = await ocr_pdf(w.client, job, args.scale, args.max_tokens,
                              dummy_sem, render_executor, args.jpeg_quality,
                              args.model, args.prompt_text)
            completed.append(j)
            status = "ok" if j.ok else f"ERR {j.error}"
            print(
                f"[done] w{w.idx} {j.in_pdf.name} → {j.out_md.name}  "
                f"({j.n_pages}pp, {j.n_chars}B, {j.elapsed_s:.1f}s, {status}; "
                f"{len(completed)}/{len(jobs)})",
                file=sys.stderr, flush=True,
            )
            queue.task_done()

    try:
        # N × in-flight consumer coroutines pulling from the shared queue.
        consumers = [
            asyncio.create_task(consumer(w))
            for w in workers
            for _ in range(args.in_flight)
        ]
        await asyncio.gather(*consumers)
        total = time.time() - t_start
        n_ok = sum(1 for j in completed if j.ok)
        n_err = sum(1 for j in completed if not j.ok)
        print(
            f"\n=== SUMMARY === {n_ok} ok / {n_err} err in {total:.1f}s "
            f"({n_ok/total*60:.1f} PDFs/min, "
            f"{args.workers} worker(s) × {args.in_flight} in-flight)",
            file=sys.stderr, flush=True,
        )

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                "name\tpages\tchars\telapsed_s\tok\terror\n" +
                "\n".join(
                    f"{j.in_pdf.name}\t{j.n_pages}\t{j.n_chars}\t"
                    f"{j.elapsed_s:.2f}\t{int(j.ok)}\t{j.error or ''}"
                    for j in completed
                ) + "\n"
            )
            print(f"[orch] timing report → {args.report}", file=sys.stderr)
    finally:
        render_executor.shutdown(wait=False)
        await asyncio.gather(
            *(teardown_worker(w) for w in workers),
            return_exceptions=True,
        )

    return 0 if n_err == 0 else 1


async def main_async_safe(args: argparse.Namespace) -> int:
    """Wrapper that catches and logs any top-level exception so it ends up in
    the output rather than silently disappearing via SIGABRT."""
    try:
        return await main_async(args)
    except Exception as exc:
        import traceback
        print(
            f"\n[FATAL] {type(exc).__name__}: {exc}\n"
            + traceback.format_exc(),
            file=sys.stderr, flush=True,
        )
        return 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-list", type=Path, default=None,
                   help="text file of <input_pdf> or <input_pdf>\\t<output_md> per line. "
                        "Required unless --keep-alive.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output dir (used when --pdf-list lines omit explicit dst). "
                        "Required unless --keep-alive.")
    p.add_argument("--keep-alive", action="store_true",
                   help="launch one worker + tunnel, print 'READY base_url=... api_key=... "
                        "model=...' to stdout, then block on stdin EOF. Used by consumers "
                        "that drive the served model "
                        "with their own OpenAI client instead of the built-in PDF queue.")
    p.add_argument("--workers", type=int, default=1,
                   help="number of parallel Slurm jobs (one vLLM server per GPU)")
    p.add_argument("--in-flight", type=int, default=32,
                   help="max concurrent PDFs in flight PER WORKER")

    # SSH / Slurm
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    p.add_argument("--partition", default="gpunormal")
    p.add_argument("--exclude", default="",
                   help="Slurm node exclude list, e.g. c001 or c001,c002")
    p.add_argument(
        "--gres", default="gpu:1",
        help="Slurm GRES. 'gpu:1' = any GPU (good default). 'gpu:a100:1' to "
             "demand A100, 'gpu:rtx8000:1' for RTX 8000, etc.",
    )
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="02:00:00",
                   help="Slurm time limit. Full 2169-corpus runs in ~20 min on "
                        "4 A100s; 2h is generous headroom. Shorter limits queue "
                        "faster on a busy cluster.")

    # Model + prompt — both are model-specific. Defaults preserve olmOCR-2.
    p.add_argument("--model", default=DEFAULT_MODEL_ID,
                   help="HF model id to serve. Passed to the Slurm service as "
                        "VLLM_MODEL. Default preserves olmOCR-2 back-compat.")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM --max-model-len. Forwarded as VLLM_MAX_MODEL_LEN.")

    # Per-model dependency isolation. Default uses the Apptainer slurm script
    # so colleagues do not need a prebuilt repo-local vLLM virtualenv.
    p.add_argument("--slurm-script", default="hpc/slurm/vllm_serve_apptainer.slurm",
                   help="path (relative to remote_dir) of the slurm script to "
                        "execute.")
    p.add_argument("--image", default=None,
                   help="apptainer image URI (e.g. docker://vllm/vllm-openai:v0.6.3). "
                        "Required; forwarded as VLLM_IMAGE env var.")
    p.add_argument("--pip-deps", default=None,
                   help="space-separated extra Python deps the model's trust-remote-code "
                        "modeling imports (e.g. 'addict matplotlib' for DeepSeek-OCR-2). "
                        "Installed with uv inside the apptainer container at startup. "
                        "Forwarded as VLLM_PIP_DEPS env var.")
    p.add_argument("--patch-mrope", action="store_true",
                   help="Bind-mount the patched mrope.py from vLLM PR #42765 into the "
                        "container. Required for GLM-OCR — the unpatched Triton MRoPE "
                        "kernel hardcodes NeoX-style rotation and produces degenerate "
                        "output for GLM models which use GPT-J-style rotation. The "
                        "patched file must exist at hpc/patches/vllm-pr42765/mrope.py "
                        "on the cluster side.")
    pg = p.add_mutually_exclusive_group()
    pg.add_argument("--prompt", default=None,
                    help="per-page instruction text. Overrides the default. "
                         "Mutually exclusive with --prompt-file.")
    pg.add_argument("--prompt-file", type=Path, default=None,
                    help="path to a UTF-8 text file containing the per-page "
                         "instruction. Mutually exclusive with --prompt.")

    # Rendering / inference
    p.add_argument("--scale", type=float, default=1.5,
                   help="pdfium render scale; 1.5 ≈ 220 DPI, plenty for OCR")
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="JPEG quality 1-95 (85 is a good throughput/quality knee)")
    p.add_argument("--max-tokens", type=int, default=4096)

    p.add_argument("--force", action="store_true",
                   help="re-OCR PDFs whose markdown already exists")
    p.add_argument("--report", type=Path, default=None,
                   help="optional tab-separated per-PDF report")
    args = p.parse_args()

    # Validate. --keep-alive mode doesn't need pdf-list/out-dir/prompt.
    if not args.keep_alive:
        if not args.pdf_list:
            sys.exit("--pdf-list is required (use --keep-alive to skip)")
        if not args.out_dir:
            sys.exit("--out-dir is required (use --keep-alive to skip)")

    # Resolve prompt source. Default to the historical olmOCR-2 prompt.
    # Unused in --keep-alive but kept for client compatibility.
    if args.prompt_file:
        args.prompt_text = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt:
        args.prompt_text = args.prompt
    else:
        args.prompt_text = DEFAULT_PROMPT

    rc = asyncio.run(main_async_safe(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
