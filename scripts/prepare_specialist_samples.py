#!/usr/bin/env python3
"""Render the public Lincoln manuscript PDF as a page image for handwriting OCR."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SAMPLES = {
    "lincoln-owens-1837": ("lincoln-owens-1837.pdf", (100, 75, 1500, 1875)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/samples/pdfs"))
    parser.add_argument("--outdir", type=Path, default=Path("data/samples/images"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for stem, (pdf_name, crop_at_200_dpi) in SAMPLES.items():
        source = args.pdf_dir / pdf_name
        if not source.is_file():
            parser.error(f"missing {source}; run `just specialist-samples` first")
        target = args.outdir / f"{stem}.png"
        if target.exists() and not args.force:
            print(target)
            continue
        command = [
            "pdftoppm",
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            "-r",
            str(args.dpi),
            "-png",
        ]
        x, y, width, height = (
            round(value * args.dpi / 200) for value in crop_at_200_dpi
        )
        command += ["-x", str(x), "-y", str(y), "-W", str(width), "-H", str(height)]
        command += [str(source), str(args.outdir / stem)]
        subprocess.run(command, check=True)
        if not target.is_file():
            raise SystemExit(f"render did not create {target}")
        print(target)


if __name__ == "__main__":
    main()
