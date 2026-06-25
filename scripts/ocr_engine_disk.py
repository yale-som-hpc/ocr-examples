#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pypdf>=4",
#     "pillow",
#     "pypdfium2",
#     "mlx-vlm>=0.3.11; platform_system == 'Darwin' and platform_machine == 'arm64'",
# ]
# ///
"""Disk-backed smoke-test runner for the OCR engine set."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from pypdf import PdfReader


PROMPTS = {
    "olmocr2": (
        "Below is one page of a document. Reproduce its content as plain "
        "text + markdown, preserving every detail faithfully."
    ),
    "deepseek_ocr": "<|grounding|>Convert the document to markdown.",
    "glm_ocr": "Text Recognition:",
}

MLX_MODELS = {
    "olmocr2": "mlx-community/olmOCR-2-7B-1025-4bit",
    "deepseek_ocr": "mlx-community/DeepSeek-OCR-2-bf16",
    "glm_ocr": "mlx-community/GLM-OCR-8bit",
}


def out_path(input_pdf: Path, outdir: Path, engine: str, suffix: str = ".txt") -> Path:
    return outdir / f"{input_pdf.stem}.{engine}{suffix}"


def write_sidecar(
    input_pdf: Path, output: Path, engine: str, started: float, params: dict
) -> None:
    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "engine_slug": engine,
                "input": str(input_pdf),
                "output": str(output),
                "elapsed_seconds": round(time.time() - started, 3),
                "output_chars": output.stat().st_size if output.exists() else 0,
                "params": params,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_pypdf(input_pdf: Path, outdir: Path) -> Path:
    started = time.time()
    reader = PdfReader(str(input_pdf))
    chunks = []
    for index, page in enumerate(reader.pages, start=1):
        chunks.append(f"<!-- page {index} -->\n{page.extract_text() or ''}".strip())
    output = out_path(input_pdf, outdir, "pypdf")
    output.write_text("\n\n".join(chunks).strip() + "\n", encoding="utf-8")
    write_sidecar(input_pdf, output, "pypdf", started, {"pages": len(reader.pages)})
    return output


def run_docling(input_pdf: Path, outdir: Path) -> Path:
    started = time.time()
    run_dir = outdir / f"{input_pdf.stem}.docling-output"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--with",
        "docling",
        "--with",
        "onnxruntime",
        "docling",
        "convert",
        str(input_pdf),
        "--to",
        "md",
        "--output",
        str(run_dir),
        "--ocr-engine",
        "rapidocr",
    ]
    log = out_path(input_pdf, outdir, "docling", ".log")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise SystemExit(f"docling failed rc={result.returncode}; see {log}")
    candidates = sorted(run_dir.glob("*.md"))
    if not candidates:
        raise SystemExit(f"docling produced no markdown in {run_dir}")
    output = out_path(input_pdf, outdir, "docling")
    output.write_text(
        candidates[-1].read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
    )
    write_sidecar(
        input_pdf,
        output,
        "docling",
        started,
        {"command": cmd, "raw_output_dir": str(run_dir)},
    )
    return output


def render_pages(input_pdf: Path, scale: float):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(input_pdf))
    try:
        for index in range(len(doc)):
            page = doc[index]
            try:
                yield index + 1, page.render(scale=scale).to_pil()
            finally:
                page.close()
    finally:
        doc.close()


def run_mlx_vlm(
    input_pdf: Path,
    outdir: Path,
    engine: str,
    model_id: str,
    scale: float,
    max_tokens: int,
) -> Path:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(
            f"{engine} disk mode uses mlx-vlm and is only supported on Apple Silicon; "
            "use tunnel mode for HPC GPU OCR."
        )
    started = time.time()
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(model_id)
    config = load_config(model_id)
    page_texts = []
    for page_num, image in render_pages(input_pdf, scale):
        prompt = apply_chat_template(processor, config, PROMPTS[engine], num_images=1)
        result = generate(
            model,
            processor,
            prompt,
            image=[image],
            max_tokens=max_tokens,
            verbose=False,
        )
        text = result.text if hasattr(result, "text") else str(result)
        page_texts.append(f"<!-- page {page_num} -->\n{text.strip()}\n")
    output = out_path(input_pdf, outdir, engine)
    output.write_text("\n\n".join(page_texts), encoding="utf-8")
    write_sidecar(
        input_pdf,
        output,
        engine,
        started,
        {"model": model_id, "scale": scale, "max_tokens": max_tokens},
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument(
        "--engine",
        choices=(
            "pypdf",
            "docling",
            "olmocr2",
            "deepseek_ocr",
            "glm_ocr",
            "unlimited_ocr",
        ),
        required=True,
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/engine-disk"))
    parser.add_argument(
        "--mlx-model", help="Override MLX model id for olmocr2/deepseek_ocr/glm_ocr"
    )
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_pdf = args.input_pdf.expanduser().resolve()
    if input_pdf.suffix.lower() != ".pdf" or not input_pdf.exists():
        raise SystemExit(f"expected an existing PDF: {input_pdf}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.engine == "pypdf":
        output = run_pypdf(input_pdf, args.outdir)
    elif args.engine == "docling":
        output = run_docling(input_pdf, args.outdir)
    elif args.engine in MLX_MODELS:
        output = run_mlx_vlm(
            input_pdf,
            args.outdir,
            args.engine,
            args.mlx_model or MLX_MODELS[args.engine],
            args.scale,
            args.max_tokens,
        )
    else:
        raise SystemExit(
            "unlimited_ocr has no local/disk backend; use the SGLang HPC tunnel path"
        )
    print(output)


if __name__ == "__main__":
    main()
