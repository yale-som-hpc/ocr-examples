#!/usr/bin/env python3
# /// script
# dependencies = ["Pillow>=10,<13"]
# ///
"""Download full-size examples published by the OCR projects themselves."""

from __future__ import annotations

import argparse
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

CHURRO_DATASET = "stanford-oval/churro-dataset"
CHURRO_REVISION = "cbeaba69179d5de122fbbc9e97b2a7d97942f688"
CHURRO_ROW = 777
CHURRO_EXAMPLE_ID = "icdar2015/072_054_002"
KRAKEN_IMAGE = (
    "https://raw.githubusercontent.com/mittagessen/kraken/"
    "17c8cef65a80ac93848ab91aac798fd3c770edd3/tests/resources/input.jpg"
)
INFINITY_PDF = (
    "https://raw.githubusercontent.com/infly-ai/INF-MLLM/"
    "011866e93026290ba39f05b796c92d58dd0ce40c/"
    "Infinity-Parser2/demo_data/demo.pdf"
)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "yale-som-hpc-ocr-examples"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def save(url: str, path: Path, magic: bytes, force: bool) -> None:
    if path.exists() and not force:
        print(f"exists {path}")
        return
    data = fetch(url)
    if not data.startswith(magic):
        raise SystemExit(f"unexpected file format from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data/samples"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    row_url = (
        "https://datasets-server.huggingface.co/rows?"
        f"dataset={CHURRO_DATASET}&config=default&split=dev"
        f"&offset={CHURRO_ROW}&length=1"
    )
    churro_path = args.outdir / "images/churro-gallery-english.jpg"
    reference_path = args.outdir / "reference/churro-gallery-english.xml"
    if args.force or not churro_path.exists() or not reference_path.exists():
        # The row API serves only the current dataset revision; catch any drift.
        dataset = json.loads(
            fetch(f"https://huggingface.co/api/datasets/{CHURRO_DATASET}")
        )
        if dataset["sha"] != CHURRO_REVISION:
            raise SystemExit("CHURRO dataset revision changed; verify the gallery row")
        row = json.loads(fetch(row_url))["rows"][0]["row"]
        if (
            row["example_id"] != CHURRO_EXAMPLE_ID
            or row["document_type"] != "handwriting"
        ):
            raise SystemExit(
                "CHURRO dataset row changed; choose and verify a new example"
            )
        if (row["image"]["width"], row["image"]["height"]) != (2731, 4096):
            raise SystemExit("CHURRO gallery image dimensions changed")
        save(row["image"]["src"], churro_path, b"\xff\xd8", args.force)
        with Image.open(churro_path) as image:
            if image.size != (2731, 4096):
                raise SystemExit("downloaded CHURRO gallery image dimensions changed")
        reference = row["cleaned_transcription"]
        ET.fromstring(reference)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(reference, encoding="utf-8")
        print(reference_path)
    else:
        print(f"exists {churro_path}")
        print(f"exists {reference_path}")

    gpu_path = args.outdir / "images/churro-gallery-english-rtx8000.jpg"
    if args.force or not gpu_path.exists():
        with Image.open(churro_path) as image:
            image.thumbnail((1400, 2100))
            image.convert("RGB").save(gpu_path, format="JPEG", quality=90)
        print(gpu_path)
    else:
        print(f"exists {gpu_path}")

    save(
        KRAKEN_IMAGE,
        args.outdir / "images/kraken-historical-print.jpg",
        b"\xff\xd8",
        args.force,
    )
    save(
        INFINITY_PDF,
        args.outdir / "pdfs/infinity-financial-table.pdf",
        b"%PDF",
        args.force,
    )


if __name__ == "__main__":
    main()
