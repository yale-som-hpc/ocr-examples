#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pypdf>=4.0",
# ]
# ///
"""Extract text from a document PDF using one or more local engines.

Each engine writes a pair under <document>/ocr/:
  ocr/<engine>.txt   — the extracted text
  ocr/<engine>.json  — provenance sidecar (engine, method, pdf_sha256, timing, params)

Engines available here:
  pypdf      — pdf-text-extraction; only writes output when chars/page >=
               --text-threshold (otherwise the PDF is a scan and pypdf yields
               garbage).
  docling    — VLM OCR via the `docling` subprocess. By default runs only as a
               fallback when pypdf is below threshold. Disable with --no-docling.
  paddleocr  — VLM OCR via the in-repo PaddleOCR-VL wrapper. Opt-in via
               --with-paddleocr; runs on every selected document.

No engine is privileged. Downstream consumers can discover all OCR results
under ocr/ and read each with its provenance sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_provenance import (  # noqa: E402
    ENGINE_DOCLING,
    ENGINE_PADDLEOCR,
    ENGINE_PYPDF,
    ocr_sidecar_path,
    ocr_text_path,
    sha256_of_pdf,
    utc_now_iso,
    write_ocr_result,
)

GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
DATA_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")


def clean_markdown(text: str) -> str:
    """Strip inline base64 images that Docling embeds; they bloat the file
    without adding signal for document processing."""
    text = DATA_IMAGE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def probe_text_native(pdf_path: Path) -> tuple[int, int, str]:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    chunks: list[str] = []
    for page in pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    full = "\n\n".join(chunks).strip()
    return len(pages), len(full), full


def run_docling(pdf_path: Path, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "docling.log"
    cmd = ["uv", "run", "--with", "docling", "docling",
           "--to", "md",
           "--output", str(out_dir),
           str(pdf_path)]
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        raise RuntimeError(f"docling failed (returncode={result.returncode}); see {log_path}")
    md_candidates = sorted(out_dir.glob("*.md"))
    if not md_candidates:
        raise RuntimeError(f"docling produced no .md output in {out_dir}")
    md_path = max(md_candidates, key=lambda p: p.stat().st_mtime)
    return md_path, {"elapsed_seconds": elapsed, "log": str(log_path)}


def run_paddleocr(pdf_path: Path, out_dir: Path) -> tuple[str, dict[str, Any]]:
    """Run PaddleOCR-VL through the in-repo wrapper, return concatenated text.

    Reuses scripts/run_paddleocr_document.py with --start-mlx-server. The
    wrapper writes per-page Markdown to:
      data/processed/<hash12-of-pdf-resolved-path>/paddleocr/document/paddleocr-output/input_*.md
    We compute that path the same way and concatenate the per-page outputs.

    The wrapper attempts an OPF cleanup step at the end; that step often fails
    (`opf` not installed) and surfaces a non-zero exit. We treat that as a
    soft error: as long as the per-page Markdown files are present we use
    them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "paddleocr-wrapper.log"
    cmd = [
        "uv", "run", "--script",
        str(Path(__file__).parent / "run_paddleocr_document.py"),
        str(pdf_path),
        "--use-uv-paddleocr",
        "--start-mlx-server",
        "--force",
    ]
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start

    pdf_hash = hashlib.sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:12]
    paddle_dir = Path("data/processed") / pdf_hash / "paddleocr" / "document" / "paddleocr-output"
    page_mds = sorted(paddle_dir.glob("input_*.md"))
    if not page_mds:
        raise RuntimeError(f"paddleocr produced no .md output for {pdf_path} (rc={result.returncode}); see {log_path}")
    combined = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in page_mds)
    return combined, {
        "elapsed_seconds": elapsed,
        "log": str(log_path),
        "page_md_dir": str(paddle_dir),
        "page_count": len(page_mds),
        "wrapper_returncode": result.returncode,  # often nonzero due to OPF; markdown still usable
    }


def extract_one(document_id: str, documents_root: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    document_dir = documents_root / document_id
    pdf_path = document_dir / "document.pdf"
    if not pdf_path.exists():
        return {"document_id": document_id, "skipped": "no document.pdf"}

    # Engines we'll consider for this run. pypdf is always probed (cheap); the
    # rest are conditional. We never re-run an engine if its sidecar+text are
    # already present unless --force is set.
    want_pypdf = True
    want_paddleocr = args.with_paddleocr
    # docling: by default acts as a local fallback (run when pypdf is below
    # threshold). --no-docling skips it entirely. --use-hpc-for-docling defers
    # to a batch HPC pass at end of main() — extract_one only records "needs
    # docling" in that case.
    if args.no_docling:
        docling_mode = "off"
    elif args.use_hpc_for_docling:
        docling_mode = "defer-hpc"
    else:
        docling_mode = "local"

    pypdf_done = ocr_text_path(document_dir, ENGINE_PYPDF).exists() and ocr_sidecar_path(document_dir, ENGINE_PYPDF).exists()
    docling_done = ocr_text_path(document_dir, ENGINE_DOCLING).exists() and ocr_sidecar_path(document_dir, ENGINE_DOCLING).exists()
    paddleocr_done = ocr_text_path(document_dir, ENGINE_PADDLEOCR).exists() and ocr_sidecar_path(document_dir, ENGINE_PADDLEOCR).exists()

    result: dict[str, Any] = {"document_id": document_id, "engines": []}
    pdf_sha = sha256_of_pdf(pdf_path)

    # ---- pypdf ----
    pages = 0
    chars_per_page = 0.0
    pypdf_usable = False
    if want_pypdf and (args.force or not pypdf_done):
        started = utc_now_iso()
        t0 = time.monotonic()
        pages, total_chars, text = probe_text_native(pdf_path)
        chars_per_page = total_chars / max(pages, 1)
        pypdf_usable = chars_per_page >= args.text_threshold
        if pypdf_usable:
            write_ocr_result(
                document_dir,
                ENGINE_PYPDF,
                text,
                provenance={
                    "method": "pdf-text-extraction",
                    "host": "trusted-local-client",
                    "pdf_pages": pages,
                    "started_at": started,
                    "finished_at": utc_now_iso(),
                    "elapsed_seconds": round(time.monotonic() - t0, 3),
                    "params": {
                        "text_threshold": args.text_threshold,
                        "chars_per_page": round(chars_per_page, 1),
                    },
                },
                pdf_sha256=pdf_sha,
            )
            result["engines"].append({"engine": ENGINE_PYPDF, "chars_per_page": round(chars_per_page, 1)})
        else:
            result["pypdf_below_threshold"] = {
                "chars_per_page": round(chars_per_page, 1),
                "text_threshold": args.text_threshold,
                "pdf_pages": pages,
            }
    elif want_pypdf and pypdf_done:
        # Read existing sidecar so docling-fallback decision can reuse the probe.
        try:
            sc = json.loads(ocr_sidecar_path(document_dir, ENGINE_PYPDF).read_text(encoding="utf-8"))
            chars_per_page = (sc.get("params") or {}).get("chars_per_page", 0)
            pages = sc.get("pdf_pages", 0)
            pypdf_usable = True
        except (OSError, json.JSONDecodeError):
            pass

    # ---- docling (fallback when pypdf was unusable) ----
    if docling_mode == "defer-hpc" and not pypdf_usable and (args.force or not docling_done):
        # Defer to the batch HPC pass at end of main(). Don't run docling locally.
        result["needs_hpc_docling"] = True
    if docling_mode == "local" and not pypdf_usable and (args.force or not docling_done):
        docling_run = document_dir / "docling-run"
        if args.force and docling_run.exists():
            shutil.rmtree(docling_run, ignore_errors=True)
        started = utc_now_iso()
        md_path, docling_info = run_docling(pdf_path, docling_run)
        raw_md = md_path.read_text(encoding="utf-8", errors="replace")
        text = clean_markdown(raw_md)
        write_ocr_result(
            document_dir,
            ENGINE_DOCLING,
            text,
            provenance={
                "method": "vlm-vision",
                "host": "trusted-local-client",
                "pdf_pages": pages or None,
                "started_at": started,
                "finished_at": utc_now_iso(),
                "elapsed_seconds": docling_info.get("elapsed_seconds"),
                "params": {
                    "docling_log": docling_info.get("log"),
                    "docling_md_path": str(md_path),
                    "raw_chars": len(raw_md),
                },
            },
            pdf_sha256=pdf_sha,
        )
        result["engines"].append({"engine": ENGINE_DOCLING, "raw_chars": len(raw_md), "elapsed_s": docling_info.get("elapsed_seconds")})

    # ---- paddleocr (opt-in) ----
    if want_paddleocr and (args.force or not paddleocr_done):
        try:
            started = utc_now_iso()
            paddle_text, paddle_info = run_paddleocr(pdf_path, document_dir / "paddleocr-run")
            paddle_clean = clean_markdown(paddle_text)
            unk_count = paddle_clean.count("<unk>")
            text_chars = sum(1 for ch in paddle_clean if ch.isalnum())
            if text_chars < 50 or unk_count * 5 > text_chars:
                result["paddleocr_skipped"] = f"low-signal output (unk={unk_count}, alnum={text_chars})"
            else:
                write_ocr_result(
                    document_dir,
                    ENGINE_PADDLEOCR,
                    paddle_clean,
                    provenance={
                        "method": "vlm-vision",
                        "host": "trusted-local-client",
                        "pdf_pages": pages or None,
                        "started_at": started,
                        "finished_at": utc_now_iso(),
                        "elapsed_seconds": paddle_info.get("elapsed_seconds"),
                        "params": {
                            "paddle_log": paddle_info.get("log"),
                            "page_count": paddle_info.get("page_count"),
                            "wrapper_returncode": paddle_info.get("wrapper_returncode"),
                            "page_md_dir": paddle_info.get("page_md_dir"),
                        },
                    },
                    pdf_sha256=pdf_sha,
                )
                result["engines"].append({"engine": ENGINE_PADDLEOCR, "chars": len(paddle_clean)})
        except Exception as exc:
            result["paddleocr_error"] = f"{type(exc).__name__}: {exc}"

    # If nothing was produced this call (all engines already existed), surface skip.
    if not result["engines"] and "paddleocr_error" not in result and "pypdf_below_threshold" not in result:
        return {"document_id": document_id, "skipped": "all selected engines already present"}

    result["pdf_pages"] = pages
    return result


def discover_local(documents_root: Path) -> list[str]:
    if not documents_root.exists():
        return []
    out: list[str] = []
    for child in sorted(documents_root.iterdir()):
        if child.is_dir() and GUID_RE.match(child.name) and (child / "document.pdf").exists():
            out.append(child.name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from document PDFs")
    parser.add_argument("--document", action="append", default=[], help="document_id GUID (repeatable)")
    parser.add_argument("--all-local", action="store_true", help="every <guid>/document.pdf under documents-root")
    parser.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    parser.add_argument("--text-threshold", type=int, default=200,
                        help="chars/page threshold for pypdf to be considered 'usable'. Below this we "
                             "treat the PDF as a scan and skip writing ocr/pypdf.*")
    parser.add_argument("--no-docling", action="store_true",
                        help="skip the docling fallback even when pypdf is below threshold. Useful when "
                             "another engine (e.g. olmOCR-2 via HPC) has already produced OCR for scans.")
    parser.add_argument("--use-hpc-for-docling", action="store_true",
                        help="Offload docling to the SOM HPC cluster via hpc/client/docling_http_client.py. "
                             "Documents needing docling are collected during the pypdf pass and submitted as "
                             "one batch at the end. Use this for any bulk run — local docling is minutes/PDF.")
    parser.add_argument("--with-paddleocr", action="store_true",
                        help="also run PaddleOCR-VL on selected documents (writes ocr/paddleocr.*).")
    parser.add_argument("--force", action="store_true")

    # HPC tuning (applies when --use-hpc-for-docling).
    parser.add_argument("--hpc-client",
                        default=str(Path(__file__).resolve().parent.parent / "hpc/client/docling_http_client.py"),
                        help="(--use-hpc-for-docling) path to docling_http_client.py")
    parser.add_argument("--hpc-workers", type=int, default=2,
                        help="(--use-hpc-for-docling) parallel Slurm jobs / GPUs")
    parser.add_argument("--hpc-in-flight", type=int, default=4,
                        help="(--use-hpc-for-docling) concurrent PDFs per worker")
    parser.add_argument("--hpc-gres", default="gpu:1",
                        help="(--use-hpc-for-docling) Slurm GRES; 'gpu:a100:1' to demand A100")
    parser.add_argument("--hpc-exclude", default="",
                        help="(--use-hpc-for-docling) Slurm node exclude list, e.g. c001")
    parser.add_argument("--hpc-mem", default="32G",
                        help="(--use-hpc-for-docling) Slurm memory request per worker")
    parser.add_argument("--hpc-cpus", type=int, default=8,
                        help="(--use-hpc-for-docling) Slurm CPU request per worker")
    parser.add_argument("--hpc-time", default="04:00:00",
                        help="(--use-hpc-for-docling) Slurm time limit per worker")
    args = parser.parse_args()

    documents: list[str] = []
    if args.document:
        for raw in args.document:
            if not GUID_RE.match(raw):
                sys.exit(f"--document {raw!r} is not a GUID")
            documents.append(raw.lower())
    if args.all_local:
        documents.extend(discover_local(args.documents_root))
    if not documents:
        sys.exit("no documents selected. use --document / --all-local")

    seen: set[str] = set()
    documents = [s for s in documents if not (s in seen or seen.add(s))]
    print(f"documents: {len(documents)}", flush=True)

    counts: dict[str, int] = {"pypdf": 0, "docling": 0, "paddleocr": 0, "skipped": 0, "error": 0,
                              "pypdf_below_threshold": 0}
    needs_hpc_docling: list[str] = []
    for document in documents:
        try:
            info = extract_one(document, args.documents_root, args)
        except Exception as exc:
            print(f"  ! {document}  error: {exc}", flush=True)
            counts["error"] += 1
            continue
        if not info:
            continue
        if "skipped" in info:
            counts["skipped"] += 1
            continue
        for eng in info.get("engines", []):
            counts[eng["engine"]] = counts.get(eng["engine"], 0) + 1
        if "pypdf_below_threshold" in info:
            counts["pypdf_below_threshold"] += 1
        if info.get("needs_hpc_docling"):
            needs_hpc_docling.append(document)
        engines_run = [e["engine"] for e in info.get("engines", [])]
        pages = info.get("pdf_pages", 0)
        suffix = ""
        if "pypdf_below_threshold" in info:
            ptb = info["pypdf_below_threshold"]
            suffix = f" pypdf<thr(cpp={ptb['chars_per_page']})"
        if info.get("needs_hpc_docling"):
            suffix += " queue=hpc-docling"
        print(f"  ok {document}  engines={','.join(engines_run) or '(none new)'}  pages={pages}{suffix}", flush=True)

    print(f"\nlocal-pass summary: {counts}", flush=True)

    # ---- batched HPC docling pass ----
    if args.use_hpc_for_docling and needs_hpc_docling:
        _run_hpc_docling(needs_hpc_docling, args, counts)
        print(f"final summary: {counts}", flush=True)


def _run_hpc_docling(documents: list[str], args: argparse.Namespace, counts: dict[str, int]) -> None:
    """Build a TSV pdf-list and shell out to docling_http_client.py.
    Mirrors the olmocr2 HPC path: HPC writes the .md, we synthesize sidecars
    from what we know after the subprocess returns.
    """
    import subprocess
    import tempfile

    jobs: list[tuple[str, Path, Path, Path]] = []  # (document, pdf, sidecar, txt)
    for document in documents:
        document_dir = args.documents_root / document
        pdf_path = document_dir / "document.pdf"
        txt_path = ocr_text_path(document_dir, ENGINE_DOCLING)
        sidecar_path = ocr_sidecar_path(document_dir, ENGINE_DOCLING)
        if not pdf_path.exists():
            continue
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((document, pdf_path.resolve(), sidecar_path, txt_path.resolve()))
    if not jobs:
        print("[hpc-docling] no jobs to submit", flush=True)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as fh:
        tsv_path = Path(fh.name)
        for _, in_p, _, out_p in jobs:
            fh.write(f"{in_p}\t{out_p}\n")
    scratch_dir = Path(tempfile.mkdtemp(prefix="docling_hpc_"))

    cmd = [
        "uv", "run",
        "--with", "httpx>=0.27",
        args.hpc_client,
        "--pdf-list", str(tsv_path),
        "--out-dir", str(scratch_dir),
        "--workers", str(args.hpc_workers),
        "--in-flight", str(args.hpc_in_flight),
        "--gres", args.hpc_gres,
        "--mem", args.hpc_mem,
        "--cpus-per-task", str(args.hpc_cpus),
        "--time", args.hpc_time,
    ]
    if args.hpc_exclude:
        cmd.extend(["--exclude", args.hpc_exclude])
    if args.force:
        cmd.append("--force")
    print(
        f"[hpc-docling] running on {len(jobs)} document(s) via "
        f"{args.hpc_workers} worker(s) × {args.hpc_in_flight} in-flight",
        flush=True,
    )
    run_started = utc_now_iso()
    try:
        rc = subprocess.call(cmd)
    finally:
        tsv_path.unlink(missing_ok=True)
        try:
            scratch_dir.rmdir()
        except OSError:
            pass
    run_finished = utc_now_iso()

    host_label = f"hpc-docling-{args.hpc_gres.replace(':', '-')}"
    written = 0
    for document, pdf_p, sidecar_p, txt_p in jobs:
        if not txt_p.exists():
            continue
        text = txt_p.read_text(encoding="utf-8", errors="replace")
        document_dir = args.documents_root / document
        write_ocr_result(
            document_dir,
            ENGINE_DOCLING,
            text,
            provenance={
                "method": "vlm-vision",
                "host": host_label,
                "started_at": run_started,
                "finished_at": run_finished,
                "params": {
                    "workers": args.hpc_workers,
                    "in_flight": args.hpc_in_flight,
                    "gres": args.hpc_gres,
                    "image": "quay.io/docling-project/docling-serve-cu126:latest",
                    "note": "per-document timing not recorded; bracket spans the full run",
                },
            },
        )
        written += 1
        counts["docling"] = counts.get("docling", 0) + 1
    print(f"[hpc-docling] rc={rc}; wrote sidecars for {written}/{len(jobs)}",
          flush=True)


if __name__ == "__main__":
    main()
