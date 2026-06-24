# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-vlm>=0.3.11",
#     "pypdfium2>=4.30",
# ]
# ///
"""Run DeepSeek-OCR-2 as one of the OCR engines on selected document PDFs.

DeepSeek-OCR-2 (deepseek-ai/DeepSeek-OCR-2) is a ~3B-param MoE VLM
(~570M active) with a token-compression encoder ("DeepEncoder V2") that
trades compute for output density — vendor benchmarks claim 200k+
pages/day on a single A100. MIT licensed. Architecturally distinct from
GLM-OCR and olmOCR-2 → useful disagreement signal in the ensemble.

Two backends:
  - default: local MLX (via mlx-vlm). Apple Silicon. ~20-40s/PDF.
  - --use-hpc: vLLM HTTP-tunneled via hpc/client/vllm_http_client.py
    against the containerized vLLM Slurm service
    (passes VLLM_MODEL=deepseek-ai/DeepSeek-OCR-2).

Writes ocr/deepseek_ocr.{txt,json}. Same gating as glm_ocr_extract.py:
skip documents with usable ocr/pypdf.txt unless --include-text-native.

Usage:
  uv run --script scripts/deepseek_ocr_extract.py --document <guid>
  uv run --script scripts/deepseek_ocr_extract.py --all-local --use-hpc --workers 4
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
    ENGINE_DEEPSEEK_OCR,
    ENGINE_PYPDF,
    ocr_sidecar_path,
    ocr_text_path,
    utc_now_iso,
    write_ocr_result,
)

# Local MLX target. The agent research surfaced a mlx-vlm module
# (deepseekocr_2) but the exact mlx-community quant id may not be
# published yet — keep this configurable via --mlx-model.
MLX_MODEL_DEFAULT = "mlx-community/DeepSeek-OCR-2-bf16"

# vLLM-served model id on HPC.
HPC_MODEL = "deepseek-ai/DeepSeek-OCR-2"
ENGINE_VERSION = "v2"

# Apptainer image. We need a vLLM with native `DeepseekOCR2ForCausalLM`
# support (architecture was added after the model's release in Jan 2026).
# v0.6.x doesn't know about it; v0.23.0 is the most recent versioned tag and
# should. When vLLM has a native impl it bypasses the HF modeling.py's
# LlamaFlashAttention2 import, sidestepping the transformers-version pinch
# that previously forced us to v0.6.x.
HPC_IMAGE = "docker://vllm/vllm-openai:v0.23.0"

GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# DeepSeek-OCR-2's HF model card prescribes a very specific prompt format:
#   <image>\n<|grounding|>Convert the document to markdown.       (full layout)
#   <image>\nFree OCR.                                            (text-only)
# vLLM's chat-completion handler maps the image_url content block to the
# `<image>` token, so we only need to send the instruction text. Anything
# else (the verbose "preserve every detail" prompt we use for olmOCR-2/GLM)
# silently returns empty output — the model is heavily fine-tuned on this
# exact template.
PROMPT = "<|grounding|>Convert the document to markdown."


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
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-text-native", action="store_true",
                   help="Also run DeepSeek-OCR on documents where pypdf already produced clean text.")
    p.add_argument("--mlx-model", default=MLX_MODEL_DEFAULT,
                   help="MLX model id (Apple Silicon). Override if a different quant is preferred.")

    p.add_argument("--use-hpc", action="store_true",
                   help="Outsource OCR to the SOM HPC cluster via vLLM HTTP.")
    p.add_argument("--workers", type=int, default=2,
                   help="(--use-hpc) parallel Slurm jobs / GPUs")
    p.add_argument("--in-flight", type=int, default=16,
                   help="(--use-hpc) concurrent PDFs per worker. DeepSeek-OCR-2's MoE "
                        "design means active params ~570M; headroom is generous on A100.")
    p.add_argument("--hpc-client",
                   default=str(Path(__file__).resolve().parent.parent / "hpc/client/vllm_http_client.py"),
                   help="(--use-hpc) path to vllm_http_client.py")
    p.add_argument("--hpc-gres", default="gpu:a100:1",
                   help="(--use-hpc) Slurm GRES. Default A100 because vLLM v0.23.0 (the "
                        "minimum that knows about DeepseekOCR2 architecture) cleanly exits "
                        "on Turing (RTX 8000, compute 7.x) right after startup — observed "
                        "behavior on the cluster's c004 RTX 8000 node. Ampere or newer "
                        "(compute >=8.0) works reliably.")
    p.add_argument("--hpc-time", default="02:00:00",
                   help="(--use-hpc) Slurm time limit per worker.")
    p.add_argument("--hpc-max-model-len", type=int, default=8192,
                   help="(--use-hpc) vLLM --max-model-len. DeepSeek-OCR-2's config.json "
                        "declares max_position_embeddings=8192 (the token-compression "
                        "encoder produces dense representations rather than long sequences); "
                        "going higher requires VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and risks "
                        "incorrect outputs.")

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

    # ---- HPC backend ----
    if args.use_hpc:
        jobs_to_run: list[tuple[str, Path, Path, Path]] = []
        for document in documents:
            document_dir = args.documents_root / document
            pdf_path = document_dir / "document.pdf"
            txt_path = ocr_text_path(document_dir, ENGINE_DEEPSEEK_OCR)
            sidecar_path = ocr_sidecar_path(document_dir, ENGINE_DEEPSEEK_OCR)
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
        prompt_path = Path(tempfile.mkstemp(prefix="deepseek_ocr_prompt_", suffix=".txt")[1])
        prompt_path.write_text(PROMPT, encoding="utf-8")
        scratch_dir = Path(tempfile.mkdtemp(prefix="deepseek_ocr_hpc_"))

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
            "--time", args.hpc_time,
            "--max-tokens", str(args.max_tokens),
            "--model", HPC_MODEL,
            "--max-model-len", str(args.hpc_max_model_len),
            "--prompt-file", str(prompt_path),
            "--slurm-script", "hpc/slurm/vllm_serve_apptainer.slurm",
            "--image", HPC_IMAGE,
            # DeepSeek-OCR-2's modeling_deepseekv2.py imports `addict` and
            # `matplotlib`. They aren't in the vllm/vllm-openai base image;
            # the slurm script installs them with uv inside the container.
            "--pip-deps", "addict matplotlib",
        ]
        if args.force:
            cmd.append("--force")
        print(
            f"running DeepSeek-OCR-2 on HPC for {len(jobs_to_run)} document(s): "
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
                ENGINE_DEEPSEEK_OCR,
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
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"loading {args.mlx_model}…", flush=True)
    t0 = time.time()
    model, processor = load(args.mlx_model)
    config = load_config(args.mlx_model)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    written = skipped = errors = 0
    for document in documents:
        document_dir = args.documents_root / document
        pdf_path = document_dir / "document.pdf"
        txt_path = ocr_text_path(document_dir, ENGINE_DEEPSEEK_OCR)
        sidecar_path = ocr_sidecar_path(document_dir, ENGINE_DEEPSEEK_OCR)
        if not pdf_path.exists():
            print(f"  ! missing PDF for {document}", flush=True)
            errors += 1
            continue
        if txt_path.exists() and sidecar_path.exists() and not args.force:
            print(f"  skip {document} (ocr/deepseek_ocr.* exists)", flush=True)
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
            ENGINE_DEEPSEEK_OCR,
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
                    "model": args.mlx_model,
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
