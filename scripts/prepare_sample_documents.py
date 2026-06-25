#!/usr/bin/env python3
"""Arrange public sample PDFs in the OCR engine document layout.

The OCR engine scripts expect:

    data/documents/<document_id_guid>/document.pdf

This script copies the public sample PDFs into deterministic fake document ID
directories and writes a document list file for --from-file.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import uuid
from pathlib import Path


NAMESPACE = uuid.UUID("0c3b3ac8-0d39-45c3-a54b-2b9d4de46dc4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("examples/sample-documents.tsv")
    )
    parser.add_argument("--sample-dir", type=Path, default=Path("data/samples/pdfs"))
    parser.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    parser.add_argument(
        "--from-file", type=Path, default=Path("data/samples/documents.txt")
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="optional max number of samples"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing document.pdf copies"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.catalog.open(encoding="utf-8"), delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]
    args.documents_root.mkdir(parents=True, exist_ok=True)
    args.from_file.parent.mkdir(parents=True, exist_ok=True)

    document_ids: list[str] = []
    for row in rows:
        filename = row["filename"]
        src = args.sample_dir / filename
        if not src.exists():
            raise SystemExit(f"missing sample PDF: {src}; run `just samples` first")
        document_id = str(uuid.uuid5(NAMESPACE, filename))
        document_dir = args.documents_root / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        dst = document_dir / "document.pdf"
        if args.force or not dst.exists():
            shutil.copy2(src, dst)
        (document_dir / "sample.json").write_text(
            "{\n"
            f'  "sample_name": "{row["name"]}",\n'
            f'  "filename": "{filename}",\n'
            f'  "source_url": "{row["url"]}"\n'
            "}\n",
            encoding="utf-8",
        )
        document_ids.append(document_id)

    args.from_file.write_text("\n".join(document_ids) + "\n", encoding="utf-8")
    print(args.from_file)


if __name__ == "__main__":
    main()
