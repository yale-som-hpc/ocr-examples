#!/usr/bin/env python3
"""Download public OCR sample PDFs listed in examples/sample-documents.tsv."""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t") if row.get("filename")
        ]
    if not rows:
        print(f"ERROR: no sample rows found in {path}", file=sys.stderr)
        sys.exit(2)
    return rows


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "yale-som-hpc-ocr-examples"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        print(
            f"ERROR: downloaded file does not look like a PDF: {url}", file=sys.stderr
        )
        sys.exit(1)
    path.write_bytes(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("examples/sample-documents.tsv")
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/samples/pdfs"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/samples/manifest.txt")
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_catalog(args.catalog)
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for row in rows:
        path = args.outdir / row["filename"]
        if args.force or not path.exists():
            print(f"downloading {row['filename']}")
            download(row["url"], path)
        else:
            print(f"exists {row['filename']}")
        paths.append(path)

    args.manifest.write_text(
        "\n".join(str(path) for path in paths) + "\n", encoding="utf-8"
    )
    print(args.manifest)


if __name__ == "__main__":
    main()
