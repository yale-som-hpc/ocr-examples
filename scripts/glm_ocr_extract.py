# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-vlm>=0.3.11; platform_system == 'Darwin' and platform_machine == 'arm64'",
#     "pypdfium2>=4.30",
# ]
# ///
"""Run GLM-OCR as one of the OCR engines on selected document PDFs.

GLM-OCR (zai-org/GLM-OCR) is a 0.9B-param VLM that currently leads
OmniDocBench v1.5 at a fraction of the size of larger OCR models. MIT
licensed. Strong on tables, formulas, and stamps — good complement to
olmOCR-2 and docling.

Two backends:
  - default: local MLX (mlx-community/GLM-OCR-8bit). Apple Silicon, no GPU.
    ~15-30s/PDF depending on page count.
  - --use-hpc: vLLM HTTP-tunneled via hpc/client/vllm_http_client.py against
    the containerized vLLM Slurm service (passes VLLM_MODEL=zai-org/GLM-OCR).
    Throughput similar to olmOCR-2 on the same A100.

Writes ocr/glm_ocr.{txt,json} per document. The sidecar records engine,
version, host, render scale, max_tokens, and pdf_sha256 — downstream consumers
discover it alongside any other engines under ocr/.

By default skips documents that already have a usable ocr/pypdf.txt (pypdf
already extracted clean text; VLM adds little). Pass --include-text-native
to run on every selected document regardless.

Usage:
  uv run --script scripts/glm_ocr_extract.py --document <guid>
  uv run --script scripts/glm_ocr_extract.py --all-local --use-hpc --workers 4
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_provenance import (  # noqa: E402
    ENGINE_GLM_OCR,
    ENGINE_PYPDF,
    ocr_sidecar_path,
    ocr_text_path,
    utc_now_iso,
    write_ocr_result,
)

# Local MLX model id (Apple Silicon). The mlx-community quants are day-0
# converts of the upstream zai-org/GLM-OCR weights.
MLX_MODEL = "mlx-community/GLM-OCR-8bit"

# vLLM-served model id on HPC. Forwarded to the slurm script as VLLM_MODEL.
HPC_MODEL = "zai-org/GLM-OCR"
ENGINE_VERSION = "v1"  # placeholder; bump when zai-org releases v2

# Apptainer image for the vLLM worker. GLM-OCR's `glm_ocr` architecture is
# only in transformers>=5. The model card's official recipe is
# `vllm/vllm-openai:nightly` with transformers from git. The nightly image
# rolls forward; if a regression surfaces, pin to a dated tag like
# `nightly-aarch64-<commit>` from docker hub.
HPC_IMAGE = "docker://vllm/vllm-openai:nightly"

GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# GLM-OCR uses short keyword prompts per the model card:
#   "Text Recognition:"     — general document text + structure
#   "Table Recognition:"    — tables specifically
#   "Formula Recognition:"  — math/formulas
# Anything else (e.g. the verbose "preserve every detail" prompt that works
# for olmOCR-2) causes the model to degenerate into repetitive HTML-table
# noise. "Text Recognition:" is the right choice for documents — the
# model handles tables/formulas internally without an explicit hint.
PROMPT = "Text Recognition:"


def render_pages(pdf_path: Path, scale: float = 2.0):
    doc = pdfium.PdfDocument(str(pdf_path))
    images = []
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()
        images.append(pil)
        page.close()
    doc.close()
    return images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document", action="append", default=[])
    p.add_argument("--from-file", type=Path)
    p.add_argument("--all-local", action="store_true",
                   help="run on every document dir under --documents-root that has document.pdf")
    p.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="per-page output token cap. GLM-OCR's max_new_tokens recipe is 8192.")
    p.add_argument("--scale", type=float, default=2.0,
                   help="pdfium render scale (local MLX path). HPC path uses 1.5 by default.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-text-native", action="store_true",
                   help="Also run GLM-OCR on documents where pypdf already produced clean text.")

    # HPC backend
    p.add_argument("--use-hpc", action="store_true",
                   help="Outsource OCR to the SOM HPC cluster via vLLM HTTP.")
    p.add_argument("--workers", type=int, default=2,
                   help="(--use-hpc) parallel Slurm jobs / GPUs")
    p.add_argument("--in-flight", type=int, default=16,
                   help="(--use-hpc) concurrent PDFs per worker. GLM-OCR is small (0.9B), "
                        "so KV-cache headroom is high; can usually push higher than olmOCR-2.")
    p.add_argument("--hpc-client",
                   default=str(Path(__file__).resolve().parent.parent / "hpc/client/vllm_http_client.py"),
                   help="(--use-hpc) path to vllm_http_client.py")
    p.add_argument("--hpc-gres", default="gpu:1",
                   help="(--use-hpc) Slurm GRES. Default is any GPU; pass gpu:a100:1 "
                        "if the selected vLLM image fails on older GPU types.")
    p.add_argument("--hpc-exclude", default="",
                   help="(--use-hpc) Slurm node exclude list, e.g. c001")
    p.add_argument("--hpc-mem", default="64G",
                   help="(--use-hpc) Slurm memory request per worker")
    p.add_argument("--hpc-cpus", type=int, default=8,
                   help="(--use-hpc) Slurm CPU request per worker")
    p.add_argument("--hpc-time", default="02:00:00",
                   help="(--use-hpc) Slurm time limit per worker.")
    p.add_argument("--hpc-max-model-len", type=int, default=65536,
                   help="(--use-hpc) vLLM --max-model-len. GLM-OCR's config.json declares "
                        "max_position_embeddings=131072; 64K leaves plenty of headroom for a "
                        "single document page (image tokens + prompt + output) and stays "
                        "well under the model ceiling.")

    args = p.parse_args()

    documents = list(args.document)
    if args.from_file:
        documents += [
            line.strip()
            for line in args.from_file.read_text().splitlines()
            if line.strip()
        ]
    if args.all_local and args.documents_root.exists():
        documents += [
            child.name
            for child in args.documents_root.iterdir()
            if child.is_dir() and GUID_RE.match(child.name) and (child / "document.pdf").exists()
        ]
    documents = sorted({s.lower() for s in documents if GUID_RE.match(s)})
    if not documents:
        sys.exit("no valid document GUIDs supplied")

    if not args.include_text_native:
        filtered = []
        skip_text_native = 0
        for document in documents:
            document_dir = args.documents_root / document
            if ocr_text_path(document_dir, ENGINE_PYPDF).exists():
                skip_text_native += 1
                continue
            filtered.append(document)
        if skip_text_native:
            print(f"skipping {skip_text_native} documents with usable ocr/pypdf.txt (pass --include-text-native to override)", flush=True)
        documents = filtered
        if not documents:
            print("no documents left to process after gating; done", flush=True)
            return

    # ---- HPC backend: TSV pdf-list + shell out to vllm_http_client.py ----
    if args.use_hpc:
        jobs_to_run: list[tuple[str, Path, Path, Path]] = []
        for document in documents:
            document_dir = args.documents_root / document
            pdf_path = document_dir / "document.pdf"
            txt_path = ocr_text_path(document_dir, ENGINE_GLM_OCR)
            sidecar_path = ocr_sidecar_path(document_dir, ENGINE_GLM_OCR)
            if not pdf_path.exists():
                print(f"  ! missing PDF for {document}", flush=True)
                continue
            if txt_path.exists() and sidecar_path.exists() and not args.force:
                continue
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            jobs_to_run.append((document, pdf_path.resolve(), sidecar_path, txt_path.resolve()))
        if not jobs_to_run:
            print("no documents left to process; done", flush=True)
            return

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8",
        ) as fh:
            tsv_path = Path(fh.name)
            for _, in_p, _, out_p in jobs_to_run:
                fh.write(f"{in_p}\t{out_p}\n")
        # Write the prompt to a tmp file so we don't have to worry about
        # shell escaping a multi-line string through argv.
        prompt_path = Path(tempfile.mkstemp(prefix="glm_ocr_prompt_", suffix=".txt")[1])
        prompt_path.write_text(PROMPT, encoding="utf-8")
        scratch_dir = Path(tempfile.mkdtemp(prefix="glm_ocr_hpc_"))

        cmd = [
            "uv", "run",
            "--with", "openai>=1.40",
            "--with", "httpx>=0.27",
            "--with", "pypdfium2>=4.30",
            "--with", "pillow>=11",
            args.hpc_client,
            "--pdf-list", str(tsv_path),
            "--out-dir", str(scratch_dir),
            "--workers", str(args.workers),
            "--in-flight", str(args.in_flight),
            "--gres", args.hpc_gres,
            "--mem", args.hpc_mem,
            "--cpus-per-task", str(args.hpc_cpus),
            "--time", args.hpc_time,
            "--max-tokens", str(args.max_tokens),
            "--model", HPC_MODEL,
            "--max-model-len", str(args.hpc_max_model_len),
            "--prompt-file", str(prompt_path),
            "--slurm-script", "hpc/slurm/vllm_serve_apptainer.slurm",
            "--image", HPC_IMAGE,
            # vLLM's Triton MRoPE kernel hardcoded NeoX-style rotation; GLM
            # models construct rotary with is_neox_style=False (GPT-J-style)
            # and emit garbage without the patch from PR #42765.
            "--patch-mrope",
        ]
        if args.hpc_exclude:
            cmd.extend(["--exclude", args.hpc_exclude])
        if args.force:
            cmd.append("--force")
        print(
            f"running GLM-OCR on HPC for {len(jobs_to_run)} document(s): "
            f"{args.workers} worker(s) × {args.in_flight} in-flight; model={HPC_MODEL}",
            flush=True,
        )
        run_started = utc_now_iso()
        try:
            rc = subprocess.call(cmd)
        finally:
            tsv_path.unlink(missing_ok=True)
            prompt_path.unlink(missing_ok=True)
            try:
                scratch_dir.rmdir()
            except OSError:
                pass
        run_finished = utc_now_iso()

        host_label = f"hpc-vllm-{args.hpc_gres.replace(':', '-')}"
        written = 0
        for document, _, _, txt_p in jobs_to_run:
            if not txt_p.exists():
                continue
            text = txt_p.read_text(encoding="utf-8", errors="replace")
            document_dir = args.documents_root / document
            write_ocr_result(
                document_dir,
                ENGINE_GLM_OCR,
                text,
                provenance={
                    "engine_version": ENGINE_VERSION,
                    "method": "vlm-vision",
                    "host": host_label,
                    "started_at": run_started,
                    "finished_at": run_finished,
                    "params": {
                        "model": HPC_MODEL,
                        "image": HPC_IMAGE,
                        "scale": 1.5,
                        "max_tokens": args.max_tokens,
                        "max_model_len": args.hpc_max_model_len,
                        "workers": args.workers,
                        "in_flight": args.in_flight,
                        "gres": args.hpc_gres,
                        "note": "per-document timing not recorded; bracket spans the full run",
                    },
                },
            )
            written += 1
        print(f"\nHPC backend rc={rc}; wrote {written}/{len(jobs_to_run)} (.txt + .json)",
              flush=True)
        sys.exit(0 if rc == 0 else rc)

    # ---- Local MLX backend ----
    # mlx-vlm support for GLM architecture may not be solid yet. If load()
    # fails, capture the error and ask the user to use --use-hpc.
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"loading {MLX_MODEL}…", flush=True)
    t0 = time.time()
    model, processor = load(MLX_MODEL)
    config = load_config(MLX_MODEL)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    written = skipped = errors = 0
    for document in documents:
        document_dir = args.documents_root / document
        pdf_path = document_dir / "document.pdf"
        txt_path = ocr_text_path(document_dir, ENGINE_GLM_OCR)
        sidecar_path = ocr_sidecar_path(document_dir, ENGINE_GLM_OCR)
        if not pdf_path.exists():
            print(f"  ! missing PDF for {document}", flush=True)
            errors += 1
            continue
        if txt_path.exists() and sidecar_path.exists() and not args.force:
            print(f"  skip {document} (ocr/glm_ocr.* exists)", flush=True)
            skipped += 1
            continue
        t_document = time.time()
        started = utc_now_iso()
        try:
            images = render_pages(pdf_path, scale=args.scale)
        except Exception as exc:
            print(f"  ! render error {document}: {exc}", flush=True)
            errors += 1
            continue
        page_texts = []
        for idx, img in enumerate(images, 1):
            formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
            t_pg = time.time()
            result = generate(
                model,
                processor,
                formatted,
                image=[img],
                max_tokens=args.max_tokens,
                verbose=False,
            )
            text = result.text if hasattr(result, "text") else str(result)
            page_texts.append(f"<!-- page {idx} -->\n{text.strip()}\n")
            print(
                f"    {document} pg{idx}/{len(images)}  "
                f"chars={len(text):,}  {time.time() - t_pg:.1f}s",
                flush=True,
            )
        joined = "\n\n".join(page_texts)
        elapsed = round(time.time() - t_document, 3)
        write_ocr_result(
            document_dir,
            ENGINE_GLM_OCR,
            joined,
            provenance={
                "engine_version": ENGINE_VERSION,
                "method": "vlm-vision",
                "host": "local-mlx",
                "pdf_pages": len(images),
                "started_at": started,
                "finished_at": utc_now_iso(),
                "elapsed_seconds": elapsed,
                "params": {
                    "scale": args.scale,
                    "max_tokens": args.max_tokens,
                    "model": MLX_MODEL,
                },
            },
        )
        print(
            f"  ok {document}  pages={len(images)}  "
            f"chars={len(joined):,}  {elapsed:.1f}s",
            flush=True,
        )
        written += 1

    print(f"\nwritten={written} skipped={skipped} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
