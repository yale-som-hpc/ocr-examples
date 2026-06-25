# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.40", "httpx>=0.27", "pypdfium2>=4.30", "pillow>=11", "dill>=0.3.8"]
# ///
"""HTTP-tunneled SGLang client for Baidu Unlimited-OCR.

This follows the same trusted-local-client→SSH→Slurm→HTTP tunnel pattern as
vllm_http_client.py, but sends the SGLang-specific OCR payload required by
baidu/Unlimited-OCR:

  - model name: Unlimited-OCR
  - images_config: {"image_mode": "gundam" | "base"}
  - optional custom no-repeat n-gram logit processor

Input PDF bytes never touch HPC disk. Pages are rendered and JPEG-encoded on
the trusted local client, then sent through the SSH tunnel to the in-memory
SGLang server.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import dill
import httpx
import pypdfium2 as pdfium
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_http_client import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_KEY,
    DEFAULT_REMOTE_DIR,
    DEFAULT_USER,
    ServerEndpoint,
    cancel_slurm_job,
    launch_vllm_serve,
    open_tunnel,
)

MODEL_ID = "baidu/Unlimited-OCR"
SERVED_MODEL_NAME = "Unlimited-OCR"
SGLANG_IMAGE = "docker://lmsysorg/sglang:dev-cu12"
SGLANG_WHEEL_URL = (
    "https://github.com/baidu/Unlimited-OCR/raw/refs/heads/main/"
    "wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl"
)
DEFAULT_UV_DEPS = f"{SGLANG_WHEEL_URL} kernels==0.11.7"
DEFAULT_PROMPT = "document parsing."


class DeepseekOCRNoRepeatNGramLogitProcessor:
    """Server-side SGLang logit processor, serialized with dill.

    This mirrors Baidu's Unlimited-OCR infer.py dependency on
    sglang.srt.sampling.custom_logit_processor.DeepseekOCRNoRepeatNGramLogitProcessor,
    but keeps the local client from needing a full SGLang install.
    """

    def __call__(
        self, logits: Any, custom_param_list: Optional[list[dict[str, Any]]] = None
    ) -> Any:
        if not custom_param_list:
            return logits
        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue
            req = params.get("__req__")
            if req is None:
                continue
            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue
            if ngram_size <= 0 or window_size <= 0:
                continue

            sequence = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue
            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            current_prefix = (
                tuple(sequence[-(ngram_size - 1) :]) if ngram_size > 1 else tuple()
            )
            banned_tokens: set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned_tokens.add(ngram[-1])

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()
            banned_tokens.difference_update(whitelist)
            if banned_tokens:
                logits[batch_idx, list(banned_tokens)] = -float("inf")
        return logits

    @classmethod
    def to_str(cls) -> str:
        return json.dumps({"callable": dill.dumps(cls).hex()})


@dataclass
class Worker:
    idx: int
    serve_proc: asyncio.subprocess.Process
    tunnel_proc: asyncio.subprocess.Process
    endpoint: ServerEndpoint
    client: httpx.AsyncClient
    ssh_host: str
    ssh_user: str
    ssh_key: str


@dataclass
class PdfJob:
    in_pdf: Path
    out_md: Path
    n_pages: int = 0
    n_chars: int = 0
    ok: bool = False
    elapsed_s: float = 0.0
    error: Optional[str] = None


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


def image_to_data_url(img: Image.Image, jpeg_quality: int) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_and_encode(pdf_bytes: bytes, scale: float, jpeg_quality: int) -> list[str]:
    return [
        image_to_data_url(page, jpeg_quality) for page in render_pages(pdf_bytes, scale)
    ]


def parse_pdf_list(list_path: Path, default_out_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for line in list_path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            in_str, out_str = line.split("\t", 1)
            pairs.append((Path(in_str.strip()), Path(out_str.strip())))
        else:
            in_p = Path(line.strip())
            pairs.append((in_p, default_out_dir / (in_p.stem + ".md")))
    return pairs


def request_payload(args: argparse.Namespace, data_url: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.served_model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0,
        "skip_special_tokens": False,
        "stream": True,
        "images_config": {"image_mode": args.image_mode},
    }
    if args.max_tokens > 0:
        payload["max_tokens"] = args.max_tokens
    if args.ngram_size > 0 and args.ngram_window > 0:
        payload["custom_logit_processor"] = (
            DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
        )
        payload["custom_params"] = {
            "ngram_size": args.ngram_size,
            "window_size": args.ngram_window,
        }
    return payload


async def wait_for_ready(
    client: httpx.AsyncClient,
    serve_proc: asyncio.subprocess.Process,
    deadline_s: float,
) -> None:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if serve_proc.returncode is not None:
            raise RuntimeError(
                f"SGLang serve process exited before health check (rc={serve_proc.returncode})"
            )
        try:
            resp = await client.get("/health", timeout=5)
            if resp.status_code == 200:
                print(
                    f"[ready] SGLang health OK after {time.time() - t0:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise RuntimeError(f"SGLang did not become healthy within {deadline_s}s")


async def launch_one_worker(args: argparse.Namespace, idx: int) -> Worker:
    extra_env = {
        "SGLANG_SERVED_MODEL_NAME": args.served_model_name,
        "SGLANG_MODEL_ARG": "--model",
        "SGLANG_ATTENTION_BACKEND": args.attention_backend,
        "SGLANG_PAGE_SIZE": str(args.page_size),
        "SGLANG_REASONING_PARSER": "none",
        "SGLANG_ENABLE_CUSTOM_LOGIT_PROCESSOR": "1" if args.ngram_size > 0 else "0",
        "SGLANG_DISABLE_OVERLAP_SCHEDULE": "1",
        "SGLANG_SKIP_SERVER_WARMUP": "1",
        "SGLANG_DISABLE_CUDA_GRAPH": "1" if args.disable_cuda_graph else "0",
        "SGLANG_MEM_FRACTION_STATIC": str(args.mem_fraction_static),
        "SGLANG_MAX_RUNNING_REQUESTS": str(args.max_running_requests),
    }
    serve_proc, endpoint = await launch_vllm_serve(
        ssh_host=args.host,
        ssh_user=args.user,
        ssh_key=args.key,
        remote_dir=args.remote_dir,
        partition=args.partition,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time_str=args.time,
        job_name=f"unlimited-ocr-w{idx}-{os.getpid()}",
        model_id=args.model,
        max_model_len=args.context_length,
        slurm_script=args.slurm_script,
        image=args.image,
        pip_deps=args.uv_deps,
        extra_env=extra_env,
        exclude=args.exclude,
    )
    tunnel_proc = await open_tunnel(args.host, args.user, args.key, endpoint)
    client = httpx.AsyncClient(
        base_url=f"http://localhost:{endpoint.local_port}",
        headers={"Authorization": f"Bearer {endpoint.api_key}"},
        timeout=httpx.Timeout(args.request_timeout),
        trust_env=False,
    )
    await wait_for_ready(
        client, serve_proc, int(os.environ.get("HPC_LAUNCH_TIMEOUT_S", "1800"))
    )
    print(
        f"[orch] w{idx} ready (localhost:{endpoint.local_port})",
        file=sys.stderr,
        flush=True,
    )
    return Worker(
        idx, serve_proc, tunnel_proc, endpoint, client, args.host, args.user, args.key
    )


async def teardown_worker(worker: Worker) -> None:
    await worker.client.aclose()
    await cancel_slurm_job(
        worker.ssh_host, worker.ssh_user, worker.ssh_key, worker.endpoint.slurm_job_id
    )
    for proc in (worker.tunnel_proc, worker.serve_proc):
        try:
            proc.terminate()
        except Exception:
            pass
    await asyncio.gather(
        *(
            asyncio.wait_for(proc.wait(), timeout=20)
            for proc in (worker.tunnel_proc, worker.serve_proc)
        ),
        return_exceptions=True,
    )


async def ocr_page(
    client: httpx.AsyncClient, args: argparse.Namespace, data_url: str
) -> str:
    resp = await client.post(
        "/v1/chat/completions",
        json=request_payload(args, data_url),
        timeout=args.request_timeout,
    )
    if resp.status_code >= 400:
        body = resp.text.replace("\n", "\\n")
        raise httpx.HTTPStatusError(
            f"{resp.status_code} {resp.reason_phrase}: {body[:1000]}",
            request=resp.request,
            response=resp,
        )
    chunks: list[str] = []
    async for raw_line in resp.aiter_lines():
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
            delta = event["choices"][0].get("delta", {}).get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        if delta:
            chunks.append(delta)
    return "".join(chunks)


async def ocr_pdf(
    worker: Worker,
    job: PdfJob,
    args: argparse.Namespace,
    render_executor: concurrent.futures.Executor,
) -> PdfJob:
    t0 = time.time()
    try:
        pdf_bytes = job.in_pdf.read_bytes()
        loop = asyncio.get_running_loop()
        data_urls = await loop.run_in_executor(
            render_executor,
            render_and_encode,
            pdf_bytes,
            args.scale,
            args.jpeg_quality,
        )
        if not data_urls:
            job.error = "no pages"
            return job
        page_texts = await asyncio.gather(
            *(ocr_page(worker.client, args, data_url) for data_url in data_urls)
        )
        markdown = (
            "\n\n".join(
                f"<!-- page {idx} -->\n{text.strip()}"
                for idx, text in enumerate(page_texts, 1)
            )
            + "\n"
        )
        job.out_md.parent.mkdir(parents=True, exist_ok=True)
        tmp = job.out_md.with_suffix(job.out_md.suffix + ".tmp")
        tmp.write_text(markdown, encoding="utf-8")
        tmp.replace(job.out_md)
        job.n_pages = len(data_urls)
        job.n_chars = len(markdown)
        job.ok = True
    except Exception as exc:
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.elapsed_s = time.time() - t0
    return job


async def main_async(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_pdf_list(args.pdf_list, args.out_dir)
    missing = [in_p for in_p, _ in pairs if not in_p.exists()]
    if missing:
        sys.exit(f"missing PDFs: {[str(p) for p in missing[:5]]}…")

    jobs = [
        PdfJob(in_pdf, out_md)
        for in_pdf, out_md in pairs
        if args.force or not out_md.exists()
    ]
    print(
        f"[orch] {len(jobs)} PDFs to process; workers={args.workers}, "
        f"in-flight={args.in_flight} per worker; image_mode={args.image_mode}",
        file=sys.stderr,
        flush=True,
    )
    if not jobs:
        return 0

    launched = await asyncio.gather(
        *(launch_one_worker(args, i) for i in range(args.workers)),
        return_exceptions=True,
    )
    workers = [w for w in launched if isinstance(w, Worker)]
    for bad in [w for w in launched if isinstance(w, BaseException)]:
        print(f"[orch] worker launch failure: {bad!r}", file=sys.stderr, flush=True)
    if not workers:
        print("[orch] NO workers came up; aborting", file=sys.stderr, flush=True)
        return 2

    queue: asyncio.Queue[PdfJob] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    render_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="pdfium"
    )
    completed: list[PdfJob] = []
    started = time.time()

    async def consumer(worker: Worker) -> None:
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await ocr_pdf(worker, job, args, render_executor)
            completed.append(result)
            status = "ok" if result.ok else f"ERR {result.error}"
            print(
                f"[done] w{worker.idx} {result.in_pdf.name} → {result.out_md.name} "
                f"({result.n_pages}pp, {result.n_chars}B, {result.elapsed_s:.1f}s, {status}; "
                f"{len(completed)}/{len(jobs)})",
                file=sys.stderr,
                flush=True,
            )
            queue.task_done()

    try:
        consumers = [
            asyncio.create_task(consumer(worker))
            for worker in workers
            for _ in range(args.in_flight)
        ]
        await asyncio.gather(*consumers)
    finally:
        render_executor.shutdown(wait=False)
        await asyncio.gather(
            *(teardown_worker(worker) for worker in workers), return_exceptions=True
        )

    elapsed = time.time() - started
    n_ok = sum(1 for job in completed if job.ok)
    n_err = sum(1 for job in completed if not job.ok)
    print(
        f"\n=== SUMMARY === {n_ok} ok / {n_err} err in {elapsed:.1f}s "
        f"({(n_ok / elapsed * 60) if elapsed else 0:.1f} PDFs/min)",
        file=sys.stderr,
        flush=True,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            "name\tpages\tchars\telapsed_s\tok\terror\n"
            + "\n".join(
                f"{job.in_pdf.name}\t{job.n_pages}\t{job.n_chars}\t"
                f"{job.elapsed_s:.2f}\t{int(job.ok)}\t{job.error or ''}"
                for job in completed
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[orch] timing report → {args.report}", file=sys.stderr, flush=True)
    return 0 if n_err == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Baidu Unlimited-OCR through SGLang on HPC"
    )
    p.add_argument(
        "--pdf-list",
        type=Path,
        required=True,
        help="text file of <input_pdf> or <input_pdf>\t<output_md> per line",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--in-flight", type=int, default=8)

    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    p.add_argument("--partition", default="gpunormal")
    p.add_argument(
        "--exclude", default="", help="Slurm node exclude list, e.g. c001 or c001,c002"
    )
    p.add_argument(
        "--gres",
        default="gpu:a100:1",
        help="Slurm GPU request. Unlimited-OCR/SGLang currently needs A100 on SOM HPC; "
        "the current Baidu/SGLang wheel fails on RTX 8000 during the MoE request path.",
    )
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="02:00:00")

    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--served-model-name", default=SERVED_MODEL_NAME)
    p.add_argument("--image", default=SGLANG_IMAGE)
    p.add_argument(
        "--uv-deps",
        default=DEFAULT_UV_DEPS,
        help="deps/wheels installed with uv inside the SGLang container before launch",
    )
    p.add_argument("--slurm-script", default="hpc/slurm/sglang_serve.slurm")
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable SGLang CUDA graph capture; default true for broader GPU compatibility.",
    )
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument("--max-running-requests", type=int, default=16)

    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--image-mode", choices=("gundam", "base"), default="gundam")
    p.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="pdfium render scale; 4.0 is close to the upstream 300 DPI PDF recipe",
    )
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=30000,
        help="per-page max output tokens; must leave room for image tokens; pass 0 to omit",
    )
    p.add_argument(
        "--ngram-size",
        type=int,
        default=0,
        help="enable custom no-repeat n-gram processor when >0. "
        "Default 0 because some SGLang/Python builds crash in this path.",
    )
    p.add_argument("--ngram-window", type=int, default=128)
    p.add_argument("--request-timeout", type=int, default=1200)

    p.add_argument("--force", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args()

    try:
        rc = asyncio.run(main_async(args))
    except Exception as exc:
        import traceback

        print(
            f"\n[FATAL] {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
