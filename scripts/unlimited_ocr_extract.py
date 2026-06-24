# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run Baidu Unlimited-OCR as one of the OCR engines on selected document PDFs.

Unlimited-OCR is served on HPC through SGLang. This wrapper mirrors the
DeepSeek/GLM wrappers: select document PDFs locally, call the HPC client with a
TSV of input/output paths, then synthesize ocr/unlimited_ocr.{txt,json}
sidecars on trusted-local-client.

No local backend is provided because the upstream direct path requires NVIDIA
CUDA. Use --use-hpc.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_provenance import (  # noqa: E402
    ENGINE_PYPDF,
    ENGINE_UNLIMITED_OCR,
    ocr_sidecar_path,
    ocr_text_path,
    utc_now_iso,
    write_ocr_result,
)

HPC_MODEL = "baidu/Unlimited-OCR"
SERVED_MODEL_NAME = "Unlimited-OCR"
ENGINE_VERSION = "2026-06-22"
HPC_IMAGE = "docker://lmsysorg/sglang:dev-cu12"
SGLANG_WHEEL_URL = (
    "https://github.com/baidu/Unlimited-OCR/raw/refs/heads/main/"
    "wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl"
)
DEFAULT_UV_DEPS = f"{SGLANG_WHEEL_URL} kernels==0.11.7"
GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def resolve_documents(args: argparse.Namespace) -> list[str]:
    documents = list(args.document)
    if args.from_file:
        documents += [line.strip() for line in args.from_file.read_text().splitlines() if line.strip()]
    if args.all_local and args.documents_root.exists():
        documents += [
            child.name
            for child in args.documents_root.iterdir()
            if child.is_dir() and GUID_RE.match(child.name) and (child / "document.pdf").exists()
        ]
    return sorted({s.lower() for s in documents if GUID_RE.match(s)})


def gate_text_native(documents: list[str], documents_root: Path, include_text_native: bool) -> list[str]:
    if include_text_native:
        return documents
    filtered: list[str] = []
    skipped = 0
    for document in documents:
        if ocr_text_path(documents_root / document, ENGINE_PYPDF).exists():
            skipped += 1
            continue
        filtered.append(document)
    if skipped:
        print(
            f"skipping {skipped} documents with usable ocr/pypdf.txt "
            "(pass --include-text-native to override)",
            flush=True,
        )
    return filtered


def collect_jobs(args: argparse.Namespace, documents: list[str]) -> list[tuple[str, Path, Path]]:
    jobs: list[tuple[str, Path, Path]] = []
    for document in documents:
        document_dir = args.documents_root / document
        pdf_path = document_dir / "document.pdf"
        txt_path = ocr_text_path(document_dir, ENGINE_UNLIMITED_OCR)
        sidecar_path = ocr_sidecar_path(document_dir, ENGINE_UNLIMITED_OCR)
        if not pdf_path.exists():
            print(f"  ! missing PDF for {document}", flush=True)
            continue
        if txt_path.exists() and sidecar_path.exists() and not args.force:
            continue
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((document, pdf_path.resolve(), txt_path.resolve()))
    return jobs


def run_hpc(args: argparse.Namespace, jobs: list[tuple[str, Path, Path]]) -> int:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as fh:
        tsv_path = Path(fh.name)
        for _, pdf_path, txt_path in jobs:
            fh.write(f"{pdf_path}\t{txt_path}\n")
    scratch_dir = Path(tempfile.mkdtemp(prefix="unlimited_ocr_hpc_"))

    cmd = [
        "uv", "run",
        "--with", "openai>=1.40",
        "--with", "httpx>=0.27",
        "--with", "pypdfium2>=4.30",
        "--with", "pillow>=11",
        "--with", "dill>=0.3.8",
        args.hpc_client,
        "--pdf-list", str(tsv_path),
        "--out-dir", str(scratch_dir),
        "--workers", str(args.workers),
        "--in-flight", str(args.in_flight),
        "--gres", args.hpc_gres,
        "--time", args.hpc_time,
        "--model", args.hpc_model,
        "--served-model-name", SERVED_MODEL_NAME,
        "--image", args.hpc_image,
        "--uv-deps", args.hpc_uv_deps,
        "--context-length", str(args.context_length),
        "--max-tokens", str(args.max_tokens),
        "--image-mode", args.image_mode,
        "--scale", str(args.scale),
        "--jpeg-quality", str(args.jpeg_quality),
        "--ngram-size", str(args.ngram_size),
        "--ngram-window", str(args.ngram_window),
        "--request-timeout", str(args.request_timeout),
    ]
    if args.force:
        cmd.append("--force")

    print(
        f"running Unlimited-OCR on HPC for {len(jobs)} document(s): "
        f"{args.workers} worker(s) × {args.in_flight} in-flight; model={args.hpc_model}",
        flush=True,
    )
    try:
        return subprocess.call(cmd)
    finally:
        tsv_path.unlink(missing_ok=True)
        try:
            scratch_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Run Baidu Unlimited-OCR on document PDFs")
    p.add_argument("--document", action="append", default=[])
    p.add_argument("--from-file", type=Path)
    p.add_argument("--all-local", action="store_true")
    p.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-text-native", action="store_true",
                   help="Also run Unlimited-OCR on documents where pypdf already produced clean text.")

    p.add_argument("--use-hpc", action="store_true", help="required; run through SGLang on HPC")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--in-flight", type=int, default=8)
    p.add_argument("--hpc-client",
                   default=str(Path(__file__).resolve().parent.parent / "hpc/client/unlimited_ocr_client.py"))
    p.add_argument("--hpc-gres", default="gpu:a100:1")
    p.add_argument("--hpc-time", default="02:00:00")
    p.add_argument("--hpc-model", default=HPC_MODEL)
    p.add_argument("--hpc-image", default=HPC_IMAGE)
    p.add_argument("--hpc-uv-deps", default=DEFAULT_UV_DEPS,
                   help="deps/wheels installed with uv inside the SGLang container before launch")
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--max-tokens", type=int, default=30000)
    p.add_argument("--image-mode", choices=("gundam", "base"), default="gundam")
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--ngram-size", type=int, default=35)
    p.add_argument("--ngram-window", type=int, default=128)
    p.add_argument("--request-timeout", type=int, default=1200)
    args = p.parse_args()

    if not args.use_hpc:
        sys.exit("Unlimited-OCR currently requires --use-hpc (CUDA/SGLang backend)")

    documents = resolve_documents(args)
    if not documents:
        sys.exit("no valid document GUIDs supplied")
    documents = gate_text_native(documents, args.documents_root, args.include_text_native)
    if not documents:
        print("no documents left to process after gating; done", flush=True)
        return

    jobs = collect_jobs(args, documents)
    if not jobs:
        print("no documents left to process; done", flush=True)
        return

    run_started = utc_now_iso()
    rc = run_hpc(args, jobs)
    run_finished = utc_now_iso()

    host_label = f"hpc-sglang-{args.hpc_gres.replace(':', '-')}"
    written = 0
    for document, _, txt_path in jobs:
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        write_ocr_result(
            args.documents_root / document,
            ENGINE_UNLIMITED_OCR,
            text,
            provenance={
                "engine_version": ENGINE_VERSION,
                "method": "vlm-vision",
                "host": host_label,
                "started_at": run_started,
                "finished_at": run_finished,
                "params": {
                    "model": args.hpc_model,
                    "served_model_name": SERVED_MODEL_NAME,
                    "image": args.hpc_image,
                    "context_length": args.context_length,
                    "max_tokens": args.max_tokens,
                    "image_mode": args.image_mode,
                    "scale": args.scale,
                    "jpeg_quality": args.jpeg_quality,
                    "ngram_size": args.ngram_size,
                    "ngram_window": args.ngram_window,
                    "workers": args.workers,
                    "in_flight": args.in_flight,
                    "gres": args.hpc_gres,
                    "note": "per-document timing not recorded; bracket spans the full run",
                },
            },
        )
        written += 1
    print(f"\nHPC backend rc={rc}; wrote {written}/{len(jobs)} (.txt + .json)", flush=True)
    sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()
