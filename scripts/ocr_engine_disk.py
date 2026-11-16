#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pypdf>=4",
# ]
# ///
"""Disk-backed smoke-test runner for the OCR engine set."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from pypdf import PdfReader


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument(
        "--engine",
        choices=(
            "pypdf",
            "docling",
        ),
        required=True,
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/engine-disk"))
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
    print(output)


if __name__ == "__main__":
    main()
