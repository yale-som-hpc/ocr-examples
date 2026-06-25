#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Validate OCR output against vendored expected phrase fixtures."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FAILURE_PATTERNS = (
    "traceback",
    "cudaerror",
    "outofmemory",
    "out of memory",
    "no kernel image",
    "error processing",
    "internal server error",
)


@dataclass
class ValidationResult:
    status: str
    reason: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"expected fixture file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON fixture {path}: {exc}") from exc


def sample_filename(args: argparse.Namespace) -> str:
    if args.filename:
        return args.filename
    if args.sample_json:
        data = load_json(args.sample_json)
        filename = data.get("filename")
        if filename:
            return str(filename)
        raise SystemExit(f"sample metadata missing filename: {args.sample_json}")
    raise SystemExit("provide --filename or --sample-json")


def merged_expectation(
    config: dict[str, Any], filename: str, engine: str
) -> dict[str, Any]:
    samples = config.get("samples") or {}
    if filename not in samples:
        raise SystemExit(f"no OCR expectation for sample filename: {filename}")
    merged: dict[str, Any] = dict(config.get("defaults") or {})
    merged.update(samples[filename])
    override = ((samples[filename].get("engine_overrides") or {}).get(engine)) or {}
    merged.update(override)
    return merged


def validate_output(output: Path, expected: dict[str, Any]) -> ValidationResult:
    if expected.get("skip"):
        return ValidationResult("skip", str(expected.get("reason") or "skipped"))
    if not output.exists():
        return ValidationResult("fail", f"output file not found: {output}")

    text = output.read_text(encoding="utf-8", errors="replace")
    text_norm = normalize_text(text)
    if not text_norm:
        return ValidationResult("fail", "normalized OCR output is empty")

    min_chars = int(expected.get("min_chars") or 0)
    if len(text_norm) < min_chars:
        return ValidationResult(
            "fail", f"output too short: {len(text_norm)} normalized chars < {min_chars}"
        )

    for pattern in FAILURE_PATTERNS:
        if pattern in text_norm:
            return ValidationResult(
                "fail", f"output contains failure pattern: {pattern}"
            )

    phrases = [str(phrase) for phrase in expected.get("required_phrases") or []]
    if phrases:
        matched: list[str] = []
        missing: list[str] = []
        for phrase in phrases:
            if normalize_text(phrase) in text_norm:
                matched.append(phrase)
            else:
                missing.append(phrase)
        required_ratio = float(expected.get("required_ratio", 1.0))
        required_count = math.ceil(len(phrases) * required_ratio)
        if len(matched) < required_count:
            return ValidationResult(
                "fail",
                "matched "
                f"{len(matched)}/{len(phrases)} phrases; need {required_count}; "
                f"missing: {missing}",
            )

    regexes = [str(pattern) for pattern in expected.get("required_regexes") or []]
    for pattern in regexes:
        if not re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            return ValidationResult("fail", f"missing regex: {pattern}")

    return ValidationResult("pass", "ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--mode", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--filename")
    parser.add_argument("--sample-json", type=Path)
    parser.add_argument(
        "--expectations", type=Path, default=Path("examples/expected-ocr.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filename = sample_filename(args)
    config = load_json(args.expectations)
    expected = merged_expectation(config, filename, args.engine)
    result = validate_output(args.output, expected)
    label = f"{args.engine}:{args.mode or '?'}:{args.gpu or '-'}:{filename}"
    if result.status == "pass":
        print(f"validation PASS {label} {args.output}", flush=True)
        return 0
    if result.status == "skip":
        print(f"validation SKIP {label}: {result.reason}", flush=True)
        return 0
    print(f"validation FAIL {label}: {result.reason}; output={args.output}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
