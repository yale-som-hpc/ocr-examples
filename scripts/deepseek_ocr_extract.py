# /// script
# requires-python = ">=3.11"
# dependencies = [
# ]
# ///
"""Run DeepSeek-OCR-2 as one of the OCR engines on selected document PDFs.

The --use-hpc backend runs DeepSeek-OCR-2 through a containerized vLLM
Slurm service and SSH tunnel.

Writes ocr/deepseek_ocr.{txt,json}. Same gating as glm_ocr_extract.py:
skip documents with usable ocr/pypdf.txt unless --include-text-native.

Usage:
  uv run --script scripts/deepseek_ocr_extract.py --document <guid>
  uv run --script scripts/deepseek_ocr_extract.py --all-local --use-hpc --workers 4
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
    ENGINE_DEEPSEEK_OCR,
    ENGINE_PYPDF,
    ocr_sidecar_path,
    ocr_text_path,
    utc_now_iso,
    write_ocr_result,
)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document", action="append", default=[])
    p.add_argument("--from-file", type=Path)
    p.add_argument(
        "--all-local",
        action="store_true",
        help="run on every document dir under --documents-root that has document.pdf",
    )
    p.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--include-text-native",
        action="store_true",
        help="Also run DeepSeek-OCR on documents where pypdf already produced clean text.",
    )

    p.add_argument(
        "--use-hpc",
        action="store_true",
        help="Outsource OCR to the SOM HPC cluster via vLLM HTTP.",
    )
    p.add_argument(
        "--workers", type=int, default=1, help="(--use-hpc) parallel Slurm jobs / GPUs"
    )
    p.add_argument(
        "--in-flight",
        type=int,
        default=1,
        help="(--use-hpc) concurrent PDFs per worker",
    )
    p.add_argument(
        "--hpc-client",
        default=str(
            Path(__file__).resolve().parent.parent / "hpc/client/vllm_http_client.py"
        ),
        help="(--use-hpc) path to vllm_http_client.py",
    )
    p.add_argument(
        "--hpc-gres",
        default="gpu:rtx8000:1",
        help="(--use-hpc) Slurm GRES; defaults to one RTX 8000",
    )
    p.add_argument(
        "--hpc-exclude",
        default="c001",
        help="(--use-hpc) Slurm node exclude list, e.g. c001",
    )
    p.add_argument(
        "--hpc-mem", default="64G", help="(--use-hpc) Slurm memory request per worker"
    )
    p.add_argument(
        "--hpc-cpus",
        type=int,
        default=8,
        help="(--use-hpc) Slurm CPU request per worker",
    )
    p.add_argument(
        "--hpc-time",
        default="02:00:00",
        help="(--use-hpc) Slurm time limit per worker.",
    )
    p.add_argument(
        "--hpc-max-model-len",
        type=int,
        default=8192,
        help="(--use-hpc) vLLM --max-model-len. DeepSeek-OCR-2's config.json "
        "declares max_position_embeddings=8192 (the token-compression "
        "encoder produces dense representations rather than long sequences); "
        "going higher requires VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and risks "
        "incorrect outputs.",
    )

    args = p.parse_args()
    if not args.use_hpc:
        sys.exit("this repository runs GPU OCR on HPC; pass --use-hpc")

    documents = list(args.document)
    if args.from_file:
        documents += [
            line.strip()
            for line in args.from_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if args.all_local and args.documents_root.exists():
        documents += [
            child.name
            for child in args.documents_root.iterdir()
            if child.is_dir()
            and GUID_RE.match(child.name)
            and (child / "document.pdf").exists()
        ]
    invalid = [value for value in documents if not GUID_RE.fullmatch(value)]
    if invalid:
        sys.exit(f"invalid document GUID(s): {', '.join(invalid)}")
    documents = sorted({value.lower() for value in documents})
    if not documents:
        sys.exit("no document GUIDs supplied")

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
            print(
                f"skipping {skip_text_native} documents with usable ocr/pypdf.txt (pass --include-text-native to override)",
                flush=True,
            )
        documents = filtered
        if not documents:
            print("no documents left to process after gating; done", flush=True)
            return

    # ---- HPC backend ----
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
        jobs_to_run.append(
            (document, pdf_path.resolve(), sidecar_path, txt_path.resolve())
        )
    if not jobs_to_run:
        print("no documents left to process; done", flush=True)
        return

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tsv",
        delete=False,
        encoding="utf-8",
    ) as fh:
        tsv_path = Path(fh.name)
        for _, in_p, _, out_p in jobs_to_run:
            fh.write(f"{in_p}\t{out_p}\n")
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="deepseek_ocr_prompt_",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as prompt_file:
        prompt_path = Path(prompt_file.name)
        prompt_file.write(PROMPT)
    scratch_dir = Path(tempfile.mkdtemp(prefix="deepseek_ocr_hpc_"))

    cmd = [
        "uv",
        "run",
        "--with",
        "openai>=1.40",
        "--with",
        "httpx>=0.27",
        "--with",
        "pypdfium2>=4.30",
        "--with",
        "pillow>=11",
        args.hpc_client,
        "--pdf-list",
        str(tsv_path),
        "--out-dir",
        str(scratch_dir),
        "--workers",
        str(args.workers),
        "--in-flight",
        str(args.in_flight),
        "--gres",
        args.hpc_gres,
        "--mem",
        args.hpc_mem,
        "--cpus-per-task",
        str(args.hpc_cpus),
        "--time",
        args.hpc_time,
        "--max-tokens",
        str(args.max_tokens),
        "--model",
        HPC_MODEL,
        "--max-model-len",
        str(args.hpc_max_model_len),
        "--prompt-file",
        str(prompt_path),
        "--slurm-script",
        "hpc/slurm/vllm_serve_apptainer.slurm",
        "--image",
        HPC_IMAGE,
        # DeepSeek-OCR-2's modeling_deepseekv2.py imports `addict` and
        # `matplotlib`. They aren't in the vllm/vllm-openai base image;
        # the slurm script installs them with uv inside the container.
        "--pip-deps",
        "addict matplotlib",
    ]
    if args.hpc_exclude:
        cmd.extend(["--exclude", args.hpc_exclude])
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
    print(
        f"\nHPC backend rc={rc}; wrote {written}/{len(jobs_to_run)} (.txt + .json)",
        flush=True,
    )
    sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()
