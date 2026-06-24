# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27"]
# ///
"""HTTP-tunneled docling-serve client.

Architecture (per worker):
  1. SSH connection #1 (long-lived): runs
     `srun ... bash hpc/slurm/docling_serve.slurm`.
     docling-serve listens on a random high port on the GPU compute node and
     prints "DOCLING_LISTEN host=cNNN port=NNNNN api_key=HHHHH" to stderr.
  2. We parse that marker from the SSH stderr stream.
  3. SSH connection #2 (long-lived, -N): tunnels localhost:LP → compute:NNNNN
     via the cluster login node. Auto-picks a free local port.
  4. We POST each PDF as multipart/form-data to
     localhost:LP/v1/convert/file with the X-Api-Key header. docling-serve
     processes layout+OCR server-side and returns markdown in JSON.

PDF bytes never persist on HPC disk — docling-serve reads the upload into
memory, processes it, returns the result. Same privacy invariant as the
olmOCR-2 vLLM client.

The structure deliberately mirrors hpc/client/vllm_http_client.py (SSH plumbing,
discovery marker parsing, multi-worker orchestration, TSV pdf-list); the only
real difference is the request shape (multipart PDF vs base64 page images) and
the response parsing (JSON document.md_content vs OpenAI chat completion).
"""
from __future__ import annotations
import argparse
import asyncio
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

DEFAULT_HOST = os.environ.get("HPC_HOST", "hpc")
DEFAULT_USER = os.environ.get("HPC_USER") or os.environ.get("USER", "")
DEFAULT_KEY = os.path.expanduser(os.environ.get("HPC_KEY", ""))
DEFAULT_REMOTE_DIR = os.environ.get("HPC_REMOTE_DIR", "ocr-examples")

LISTEN_RE = re.compile(r"DOCLING_LISTEN\s+host=(\S+)\s+port=(\d+)\s+api_key=(\S+)")
SLURM_JOB_RE = re.compile(r'"event":"job_start".*"job":"([^"}]+)"')


def ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}" if user else host


def ssh_base_args(ssh_key: str) -> list[str]:
    args: list[str] = []
    if ssh_key:
        args.extend(["-i", ssh_key, "-o", "IdentitiesOnly=yes"])
    return args


# ---- SSH plumbing ----

@dataclass
class ServerEndpoint:
    compute_host: str
    compute_port: int
    api_key: str
    slurm_job_id: str | None = None
    local_port: int = 0


async def launch_docling_serve(
    ssh_host: str, ssh_user: str, ssh_key: str, remote_dir: str,
    partition: str, gres: str, cpus_per_task: int, mem: str, time_str: str,
    job_name: str,
) -> tuple[asyncio.subprocess.Process, ServerEndpoint]:
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"srun "
        f"--partition={partition} "
        f"--gres={gres} "
        f"--cpus-per-task={cpus_per_task} "
        f"--mem={mem} "
        f"--time={time_str} "
        f"--job-name={shlex.quote(job_name)} "
        f"bash hpc/slurm/docling_serve.slurm"
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
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    endpoint: Optional[ServerEndpoint] = None
    slurm_job_id: str | None = None
    # docling-serve start = apptainer pull (first run, several GB) + model
    # weight download (first run, ~1GB) + uvicorn boot. Give it room.
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
        raise RuntimeError("did not see DOCLING_LISTEN marker within timeout")

    async def pump_remaining_stderr():
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
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

async def wait_for_ready(client: httpx.AsyncClient, deadline_s: float) -> None:
    """Poll /health until docling-serve responds 2xx. The model load happens
    lazily on first request, so /health passing doesn't mean the first
    /convert will be instant — but it does mean uvicorn is up and willing
    to queue."""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            r = await client.get("/health")
            if r.status_code < 500:
                elapsed = time.time() - t0
                print(f"[ready] docling-serve responding after {elapsed:.1f}s "
                      f"(status={r.status_code})", file=sys.stderr, flush=True)
                return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise RuntimeError(f"docling-serve didn't become ready within {deadline_s}s")


@dataclass
class PdfJob:
    in_pdf: Path
    out_md: Path
    n_chars: int = 0
    ok: bool = False
    elapsed_s: float = 0.0
    error: Optional[str] = None
    server_processing_s: Optional[float] = None


async def convert_pdf(client: httpx.AsyncClient, job: PdfJob) -> PdfJob:
    """POST one PDF to /v1/convert/file, write markdown atomically."""
    t0 = time.time()
    try:
        pdf_bytes = job.in_pdf.read_bytes()
        files = {"files": (job.in_pdf.name, pdf_bytes, "application/pdf")}
        # to_formats=md → only md_content populated (saves transfer).
        # do_table_structure stays default-on (documents ARE tables).
        data = {"to_formats": "md"}
        r = await client.post("/v1/convert/file", files=files, data=data)
        r.raise_for_status()
        payload = r.json()
        md = (payload.get("document") or {}).get("md_content", "") or ""
        if not md:
            status = payload.get("status")
            errors = payload.get("errors") or []
            job.error = f"empty md_content; status={status} errors={errors[:1]}"
            job.elapsed_s = time.time() - t0
            return job

        job.out_md.parent.mkdir(parents=True, exist_ok=True)
        tmp = job.out_md.with_suffix(job.out_md.suffix + ".tmp")
        tmp.write_bytes(md.encode("utf-8"))
        tmp.replace(job.out_md)

        job.n_chars = len(md)
        job.server_processing_s = payload.get("processing_time")
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
    client: httpx.AsyncClient
    endpoint: ServerEndpoint
    ssh_host: str
    ssh_user: str
    ssh_key: str


async def launch_one_worker(args: argparse.Namespace, idx: int) -> Worker:
    print(f"[orch] launching worker w{idx}…", file=sys.stderr, flush=True)
    serve_proc, endpoint = await launch_docling_serve(
        ssh_host=args.host, ssh_user=args.user, ssh_key=args.key,
        remote_dir=args.remote_dir,
        partition=args.partition, gres=args.gres,
        cpus_per_task=args.cpus_per_task, mem=args.mem, time_str=args.time,
        job_name=f"docling-serve-w{idx}-{os.getpid()}",
    )
    tunnel_proc = await open_tunnel(args.host, args.user, args.key, endpoint)
    await asyncio.sleep(1.0)
    client = httpx.AsyncClient(
        base_url=f"http://localhost:{endpoint.local_port}",
        headers={"X-Api-Key": endpoint.api_key, "accept": "application/json"},
        # connect: cheap; read: docling can take 30-60s on big scans; pool: small.
        timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=10.0),
        # one HTTP connection per in-flight request, capped at args.in_flight.
        limits=httpx.Limits(max_keepalive_connections=args.in_flight,
                            max_connections=args.in_flight),
    )
    await wait_for_ready(client, deadline_s=600)
    print(f"[orch] w{idx} ready (port {endpoint.local_port} → {endpoint.compute_host}:{endpoint.compute_port})",
          file=sys.stderr, flush=True)
    return Worker(
        idx=idx, serve_proc=serve_proc, tunnel_proc=tunnel_proc,
        client=client, endpoint=endpoint,
        ssh_host=args.host, ssh_user=args.user, ssh_key=args.key,
    )


async def cancel_slurm_job(ssh_host: str, ssh_user: str, ssh_key: str, job_id: str | None) -> None:
    """Best-effort cleanup for ssh-driven srun jobs."""
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
    try:
        await w.client.aclose()
    except Exception:
        pass
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
      - "<pdf_path>\\t<out_md_path>"    → output goes to the explicit path

    The TSV form lets callers place outputs into per-stream dirs without
    symlinks or post-move.
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


async def main_async(args: argparse.Namespace) -> int:
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

    workers_or_exc = await asyncio.gather(
        *(launch_one_worker(args, i) for i in range(args.workers)),
        return_exceptions=True,
    )
    bad = [w for w in workers_or_exc if isinstance(w, BaseException)]
    live = [w for w in workers_or_exc if isinstance(w, Worker)]
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

    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)

    t_start = time.time()
    completed: list[PdfJob] = []

    async def consumer(w: Worker) -> None:
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            j = await convert_pdf(w.client, job)
            completed.append(j)
            status = "ok" if j.ok else f"ERR {j.error}"
            print(
                f"[done] w{w.idx} {j.in_pdf.name} → {j.out_md.name}  "
                f"({j.n_chars}B, {j.elapsed_s:.1f}s, srv={j.server_processing_s}, {status}; "
                f"{len(completed)}/{len(jobs)})",
                file=sys.stderr, flush=True,
            )
            queue.task_done()

    try:
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
                "name\tchars\telapsed_s\tserver_s\tok\terror\n" +
                "\n".join(
                    f"{j.in_pdf.name}\t{j.n_chars}\t"
                    f"{j.elapsed_s:.2f}\t{j.server_processing_s or ''}\t"
                    f"{int(j.ok)}\t{j.error or ''}"
                    for j in completed
                ) + "\n"
            )
            print(f"[orch] timing report → {args.report}", file=sys.stderr)
    finally:
        await asyncio.gather(
            *(teardown_worker(w) for w in workers),
            return_exceptions=True,
        )

    return 0 if n_err == 0 else 1


async def main_async_safe(args: argparse.Namespace) -> int:
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
    p.add_argument("--pdf-list", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1,
                   help="number of parallel Slurm jobs (one docling-serve per GPU)")
    p.add_argument("--in-flight", type=int, default=4,
                   help="max concurrent PDFs in flight PER WORKER. docling-serve does "
                        "its own internal queueing (DOCLING_SERVE_ENG_LOC_NUM_WORKERS); "
                        "pushing more in-flight than that just queues at the server.")

    # SSH / Slurm
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    p.add_argument("--partition", default="gpunormal")
    p.add_argument(
        "--gres", default="gpu:1",
        help="Slurm GRES. 'gpu:1' = any GPU. 'gpu:a100:1' to demand A100.",
    )
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--mem", default="32G")
    p.add_argument("--time", default="04:00:00",
                   help="Slurm time limit. Docling per-PDF is variable; reserve "
                        "generously to ride out queue contention.")

    p.add_argument("--force", action="store_true",
                   help="re-process PDFs whose markdown already exists")
    p.add_argument("--report", type=Path, default=None,
                   help="optional tab-separated per-PDF report")
    args = p.parse_args()
    rc = asyncio.run(main_async_safe(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
