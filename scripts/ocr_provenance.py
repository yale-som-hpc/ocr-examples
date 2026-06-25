"""Shared helpers for the per-engine OCR layout under data/documents/<guid>/ocr/.

The shape is intentionally flat: no OCR is privileged. Each engine writes a pair
under `<document>/ocr/`:

    ocr/<engine_slug>.txt    — extracted text (no engine-specific marker bytes)
    ocr/<engine_slug>.json   — provenance sidecar

Adding a new engine = pick a slug, write the two files. Consumers discover
everything present via :func:`discover_ocrs`.

Sidecar schema (v1):

    {
      "schema_version": 1,
      "engine_slug": "olmocr2",
      "engine": "olmOCR-2",
      "engine_version": "7B-1025",     # optional
      "method": "vlm-vision",          # pdf-text-extraction | vlm-vision | other
      "host": "hpc-vllm-a100",         # free-form label
      "pdf_sha256": "<hex>",
      "pdf_pages": 4,                  # optional but recommended
      "output_chars": 12345,
      "started_at": "2026-06-21T01:23:45+00:00",
      "finished_at": "2026-06-21T01:24:13+00:00",
      "elapsed_seconds": 28.1,
      "params": { ... engine-specific ... }
    }

The `engine_slug` is the canonical filename stem; the rest is interpretive.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OCR_SCHEMA_VERSION = 1

# Canonical engine slugs — used as filenames under <document>/ocr/.
ENGINE_PYPDF = "pypdf"
ENGINE_DOCLING = "docling"
ENGINE_OLMOCR2 = "olmocr2"
ENGINE_GLM_OCR = "glm_ocr"
ENGINE_DEEPSEEK_OCR = "deepseek_ocr"
ENGINE_UNLIMITED_OCR = "unlimited_ocr"

# Human-readable engine labels. Looked up when synthesizing a sidecar
# from minimal info (e.g. during migration of legacy files).
ENGINE_LABEL: dict[str, str] = {
    ENGINE_PYPDF: "pypdf",
    ENGINE_DOCLING: "docling",
    ENGINE_OLMOCR2: "olmOCR-2",
    ENGINE_GLM_OCR: "GLM-OCR",
    ENGINE_DEEPSEEK_OCR: "DeepSeek-OCR-2",
    ENGINE_UNLIMITED_OCR: "Unlimited-OCR",
}

# Default method tag per engine. `method` is descriptive (how the engine
# obtains text from the PDF) — `pdf-text-extraction` vs `vlm-vision`.
ENGINE_METHOD: dict[str, str] = {
    ENGINE_PYPDF: "pdf-text-extraction",
    ENGINE_DOCLING: "vlm-vision",
    ENGINE_OLMOCR2: "vlm-vision",
    ENGINE_GLM_OCR: "vlm-vision",
    ENGINE_DEEPSEEK_OCR: "vlm-vision",
    ENGINE_UNLIMITED_OCR: "vlm-vision",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_of_pdf(pdf_path: Path) -> str:
    """SHA-256 hex digest of the source PDF. Used to detect re-OCR-after-replace
    later: a sidecar whose pdf_sha256 no longer matches the current
    document.pdf is stale.
    """
    h = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ocr_dir(document_dir: Path) -> Path:
    return document_dir / "ocr"


def ocr_text_path(document_dir: Path, engine: str) -> Path:
    return ocr_dir(document_dir) / f"{engine}.txt"


def ocr_sidecar_path(document_dir: Path, engine: str) -> Path:
    return ocr_dir(document_dir) / f"{engine}.json"


def write_ocr_result(
    document_dir: Path,
    engine: str,
    text: str,
    *,
    provenance: dict[str, Any],
    pdf_sha256: str | None = None,
) -> None:
    """Atomically write ocr/<engine>.txt + ocr/<engine>.json.

    `provenance` should include at minimum: method, started_at, finished_at,
    and any engine-specific params. `pdf_sha256`, `output_chars`, `engine_slug`,
    `engine`, and `schema_version` are filled in automatically if absent.
    """
    d = ocr_dir(document_dir)
    d.mkdir(parents=True, exist_ok=True)

    if pdf_sha256 is None:
        pdf_path = document_dir / "document.pdf"
        pdf_sha256 = sha256_of_pdf(pdf_path) if pdf_path.exists() else None

    sidecar: dict[str, Any] = {
        "schema_version": OCR_SCHEMA_VERSION,
        "engine_slug": engine,
        "engine": ENGINE_LABEL.get(engine, engine),
        "method": ENGINE_METHOD.get(engine, "unknown"),
        "pdf_sha256": pdf_sha256,
        "output_chars": len(text),
    }
    # provenance overrides defaults (e.g. caller may pass a non-default method).
    sidecar.update(provenance)
    # Force engine_slug + schema_version to the canonical values.
    sidecar["engine_slug"] = engine
    sidecar["schema_version"] = OCR_SCHEMA_VERSION

    txt_path = ocr_text_path(document_dir, engine)
    tmp_txt = txt_path.with_suffix(".txt.tmp")
    tmp_txt.write_text(text, encoding="utf-8")
    tmp_txt.replace(txt_path)

    json_path = ocr_sidecar_path(document_dir, engine)
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    tmp_json.replace(json_path)


def discover_ocrs(document_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    """Return [(engine_slug, txt_path, sidecar), ...] for every OCR present
    under <document>/ocr/. Sorted alphabetically by engine_slug for deterministic
    prompt ordering. Only entries with BOTH .json and .txt are returned —
    a lone .json (sidecar-without-output) is skipped silently.
    """
    d = ocr_dir(document_dir)
    if not d.exists():
        return []
    out: list[tuple[str, Path, dict[str, Any]]] = []
    for json_path in sorted(d.glob("*.json")):
        engine = json_path.stem
        txt_path = d / f"{engine}.txt"
        if not txt_path.exists():
            continue
        try:
            sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sidecar = {}
        out.append((engine, txt_path, sidecar))
    return out


def ocr_label(engine: str, sidecar: dict[str, Any]) -> str:
    """Build a human-readable label for prompt blocks, e.g.
    'OCR pypdf (pypdf, pdf-text-extraction)'. Only relies on sidecar fields
    that are always present (engine name, method)."""
    engine_name = sidecar.get("engine") or ENGINE_LABEL.get(engine, engine)
    method = sidecar.get("method") or ENGINE_METHOD.get(engine, "unknown")
    return f"OCR {engine} ({engine_name}, {method})"
