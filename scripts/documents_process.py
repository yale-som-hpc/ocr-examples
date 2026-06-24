#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""End-to-end document OCR + processing driver.

Select OCR engines and pipeline stages; dispatch to per-engine wrappers
and per-stage scripts. Idempotent — every underlying script skips documents
whose outputs are already present.

Usage:
  just documents-process --all-local      --engines all --use-hpc
  just documents-process --from-file /tmp/documents.txt --engines all --use-hpc --strict
  just documents-process --all-local --audit                  # no work
  just documents-process --all-local --engines all --dry-run  # what would happen
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
SCRIPT_DIR = Path(__file__).resolve().parent


# ---------- engine + stage registry ----------

# Engine slug → wrapper config. Per-engine HPC defaults (workers/in_flight)
# are sane starting points; user can override via --hpc-workers eng=N.
# pypdf + docling are coupled inside documents_extract.py — both handled
# by a single subprocess call to that script (see run_extract_engines()).
# `ocr_slug` is the filename stem written by the engine wrapper to
# ocr/<ocr_slug>.{txt,json} — must match scripts/ocr_provenance.py
# (deepseek_ocr, glm_ocr) even when the user-facing CLI name is shorter.
#
# `sec_per_pdf` is the observed per-PDF processing time per worker (rough).
# Used to size --hpc-time proportional to N documents (see compute_hpc_time()).
# Slurm jobs that outlive their --time get killed by the scheduler, so
# orphans from a crashed orchestrator can't linger more than the bound.
#
# `gres` is `gpu:a100:1` everywhere because (a) deepseek + glm require A100
# (vLLM ≥0.23 graceful-exits on Turing) and (b) A100 is 2-3x faster than
# RTX 8000 for the other GPU OCR engines too.
ENGINES: dict[str, dict[str, Any]] = {
    "pypdf":    {"script": "documents_extract.py", "has_hpc": False,
                 "ocr_slug": "pypdf"},
    "docling":  {"script": "documents_extract.py", "has_hpc": True,
                 "ocr_slug": "docling",
                 "workers": 4, "in_flight": 4, "sec_per_pdf": 30,
                 "gres": "gpu:a100:1",
                 # documents_extract.py uses --hpc-workers/--hpc-in-flight
                 # prefix (different from olmocr2/deepseek/glm).
                 "workers_flag": "--hpc-workers", "in_flight_flag": "--hpc-in-flight"},
    "olmocr2":  {"script": "olmocr2_extract.py", "has_hpc": True,
                 "ocr_slug": "olmocr2",
                 "workers": 4, "in_flight": 24, "sec_per_pdf": 10,
                 "gres": "gpu:a100:1",
                 "hpc_flag": "--use-hpc",
                 "workers_flag": "--workers", "in_flight_flag": "--in-flight"},
    "deepseek": {"script": "deepseek_ocr_extract.py", "has_hpc": True,
                 "ocr_slug": "deepseek_ocr",
                 "workers": 4, "in_flight": 16, "sec_per_pdf": 20,
                 "gres": "gpu:a100:1",
                 "hpc_flag": "--use-hpc",
                 "workers_flag": "--workers", "in_flight_flag": "--in-flight"},
    "glm":      {"script": "glm_ocr_extract.py", "has_hpc": True,
                 "ocr_slug": "glm_ocr",
                 "workers": 4, "in_flight": 16, "sec_per_pdf": 15,
                 "gres": "gpu:a100:1",
                 "hpc_flag": "--use-hpc",
                 "workers_flag": "--workers", "in_flight_flag": "--in-flight"},
    "unlimited_ocr": {"script": "unlimited_ocr_extract.py", "has_hpc": True,
                 "ocr_slug": "unlimited_ocr",
                 "workers": 2, "in_flight": 8, "sec_per_pdf": 30,
                 "gres": "gpu:a100:1",
                 "hpc_flag": "--use-hpc",
                 "workers_flag": "--workers", "in_flight_flag": "--in-flight"},
}
ENGINE_ORDER = ["pypdf", "docling", "olmocr2", "deepseek", "glm", "unlimited_ocr"]


def compute_hpc_time(eng: str, n_documents: int, workers: int,
                     safety: float = 1.5,
                     floor_min: int = 30, cap_hr: int = 24) -> str:
    """Compute a --hpc-time bound based on the cohort size + worker count.

    Each Slurm worker processes ~n_documents/workers PDFs sequentially at
    `sec_per_pdf` (per-engine empirical rate). Multiply by `safety` to
    survive queue contention, image pulls, and per-PDF variance.

    Floor at 30 min, cap at 24 hr. Returns HH:MM:SS.
    """
    sec_per_pdf = ENGINES[eng].get("sec_per_pdf", 30)
    if workers <= 0:
        workers = 1
    total_min = int(n_documents * sec_per_pdf / workers * safety / 60)
    total_min = max(floor_min, min(cap_hr * 60, total_min))
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}:00"

STAGES = ["ocr"]


def expand_engines(spec: str) -> list[str]:
    if spec == "all":
        return list(ENGINE_ORDER)
    out = [e.strip() for e in spec.split(",") if e.strip()]
    bad = [e for e in out if e not in ENGINES]
    if bad:
        sys.exit(f"unknown engine(s): {bad}. valid: {ENGINE_ORDER + ['all']}")
    return out


def expand_stages(spec: str) -> list[str]:
    if spec == "all":
        return list(STAGES)
    out: list[str] = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        if s == "ocr":
            out.append("ocr")
        elif s in STAGES:
            out.append(s)
        else:
            sys.exit(f"unknown stage: {s}. valid: {STAGES + ['all']}")
    # Dedup, preserve order
    seen: set[str] = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def parse_kv_pairs(s: str) -> dict[str, int]:
    """Parse 'olmocr2=8,deepseek=4' → {'olmocr2': 8, 'deepseek': 4}."""
    if not s:
        return {}
    out: dict[str, int] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, _, v = pair.partition("=")
        if not v:
            sys.exit(f"bad --hpc-workers/--hpc-in-flight pair: {pair!r} (expected eng=N)")
        out[k.strip()] = int(v)
    return out


# ---------- document resolution ----------

def discover_local(documents_root: Path) -> list[str]:
    if not documents_root.exists():
        return []
    return sorted(
        c.name for c in documents_root.iterdir()
        if c.is_dir() and GUID_RE.match(c.name) and (c / "document.pdf").exists()
    )


def filter_complete_only(documents: Iterable[str], documents_root: Path) -> list[str]:
    """Filter documents to those whose .manifest.jsonl record has complete_only=true."""
    manifest = documents_root / ".manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"--complete-only requested but no manifest at {manifest}")
    complete: set[str] = set()
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("complete_only"):
            continue
        for pdf in rec.get("pdfs", []) or []:
            document = pdf.get("document")
            if document:
                complete.add(document.lower())
    return [s for s in documents if s.lower() in complete]


def resolve_documents(args: argparse.Namespace) -> list[str]:
    documents: list[str]
    if args.document:
        documents = []
        for raw in args.document:
            if not GUID_RE.match(raw):
                sys.exit(f"--document {raw!r} is not a GUID")
            documents.append(raw.lower())
    elif args.from_file:
        if not args.from_file.exists():
            sys.exit(f"--from-file {args.from_file} not found")
        documents = []
        for line in args.from_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not GUID_RE.match(line):
                sys.exit(f"--from-file: not a GUID: {line!r}")
            documents.append(line.lower())
    else:  # args.all_local
        documents = discover_local(args.documents_root)

    if args.complete_only:
        documents = filter_complete_only(documents, args.documents_root)

    # Filter to documents with document.pdf on disk.
    on_disk = [s for s in documents if (args.documents_root / s / "document.pdf").exists()]
    n_dropped = len(set(documents)) - len(set(on_disk))
    if n_dropped:
        print(f"note: {n_dropped} document(s) selected but missing document.pdf — skipping", flush=True)
    return sorted(set(on_disk))


# ---------- audit ----------

def audit(args: argparse.Namespace) -> int:
    documents = discover_local(args.documents_root)
    print(f"corpus: {len(documents)} document(s) under {args.documents_root}\n", flush=True)
    print(f"{'engine':<14}{'.txt':>10}{'.json':>10}")
    print("-" * 34)
    for eng in ENGINE_ORDER:
        slug = ENGINES[eng]["ocr_slug"]
        n_txt = sum(1 for s in documents if (args.documents_root / s / "ocr" / f"{slug}.txt").exists())
        n_json = sum(1 for s in documents if (args.documents_root / s / "ocr" / f"{slug}.json").exists())
        print(f"{eng:<14}{n_txt:>10}{n_json:>10}")
    return 0


# ---------- subprocess + manifest ----------

def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def run_subprocess(cmd: list[str], label: str, manifest: list[dict[str, Any]]) -> int:
    print(f"\n=== {label} ===", flush=True)
    head = " ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else "")
    print(f"$ {head}", flush=True)
    manifest.append({"event": "stage_start", "ts": utc_iso(), "stage": label})
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = round(time.time() - t0, 1)
    manifest.append({
        "event": "stage_end", "ts": utc_iso(), "stage": label,
        "exit_code": result.returncode, "elapsed_s": elapsed,
    })
    return result.returncode


# ---------- main orchestration ----------

def main() -> int:
    p = argparse.ArgumentParser(
        description="End-to-end document OCR + processing driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Document selection (one required, except for --audit)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--document", action="append", default=[],
                     help="document_id GUID (repeatable)")
    src.add_argument("--all-local", action="store_true",
                     help="every PDF already under --documents-root")
    src.add_argument("--from-file", type=Path,
                     help="text file with one GUID per line (# comments ok)")
    p.add_argument("--complete-only", action="store_true",
                   help="filter to documents marked complete_only:true in .manifest.jsonl "
                        "(only meaningful for SFTP-tracked documents)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the resolved document list to the first N (deterministic — "
                        "post-sort). Useful for preflight / sanity runs.")

    # Engines + stages
    p.add_argument("--engines", default=None,
                   help=f"REQUIRED (except --audit). Comma-separated subset of "
                        f"{ENGINE_ORDER}, or 'all'.")
    p.add_argument("--stages", default="all",
                   help=f"comma-separated subset of {STAGES}, or 'all'.")

    # HPC routing
    p.add_argument("--use-hpc", action="store_true",
                   help="enable HPC backends for every selected engine that has one. "
                        "pypdf is local-only; docling/olmocr2/deepseek/glm/unlimited_ocr get HPC.")
    p.add_argument("--no-hpc-for", default="",
                   help="comma-separated engines to keep local even when --use-hpc")

    # Modes
    p.add_argument("--strict", action="store_true",
                   help="exit on first stage failure (default: continue, summarize at end)")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve documents + print plan, then exit. No work.")
    p.add_argument("--audit", action="store_true",
                   help="walk corpus, print coverage by engine + stage, exit. No work.")

    # HPC OCR tuning (per-engine overrides; defaults in the ENGINES table)
    p.add_argument("--hpc-workers", default="",
                   help="per-engine workers override, e.g. olmocr2=8,deepseek=4. "
                        f"Defaults: " + ", ".join(
                            f"{e}={ENGINES[e]['workers']}" for e in ENGINE_ORDER
                            if ENGINES[e].get("has_hpc")))
    p.add_argument("--hpc-in-flight", default="",
                   help="per-engine in-flight override, e.g. olmocr2=32,deepseek=24. "
                        f"Defaults: " + ", ".join(
                            f"{e}={ENGINES[e]['in_flight']}" for e in ENGINE_ORDER
                            if ENGINES[e].get("has_hpc")))

    # Filesystem roots
    p.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    p.add_argument("--force", action="store_true",
                   help="re-run each stage even if outputs exist")

    args = p.parse_args()

    # --audit short-circuits everything
    if args.audit:
        return audit(args)

    # Validate source + engines
    if not (args.document or args.all_local or args.from_file):
        sys.exit("must specify a document source: --document / --all-local / "
                 "--from-file (or use --audit)")
    if not args.engines:
        sys.exit("must specify --engines (e.g. pypdf,olmocr2 or all)")

    engines = expand_engines(args.engines)
    stages = expand_stages(args.stages)
    no_hpc = {s.strip() for s in args.no_hpc_for.split(",") if s.strip()}
    workers_override = parse_kv_pairs(args.hpc_workers)
    in_flight_override = parse_kv_pairs(args.hpc_in_flight)

    documents = resolve_documents(args)
    if not documents:
        sys.exit("no documents selected (after on-disk filter)")

    if args.limit is not None and args.limit > 0 and len(documents) > args.limit:
        n_dropped = len(documents) - args.limit
        documents = documents[: args.limit]
        print(f"--limit {args.limit}: keeping first {args.limit}, dropping {n_dropped}", flush=True)

    hpc_engines = [e for e in engines
                   if args.use_hpc and ENGINES[e].get("has_hpc") and e not in no_hpc]

    print(f"documents: {len(documents)}", flush=True)
    print(f"engines: {','.join(engines)}", flush=True)
    print(f"stages:  {','.join(stages)}", flush=True)
    if hpc_engines:
        print(f"HPC:     {','.join(hpc_engines)}", flush=True)
    else:
        print("HPC:     (none — all engines local)", flush=True)

    if args.dry_run:
        print("\n(dry-run; exiting without work)", flush=True)
        return 0

    # Manifest
    Path("logs/runs").mkdir(parents=True, exist_ok=True)
    run_id = utc_iso().replace(":", "-").replace("+00:00", "Z")
    manifest_path = Path(f"logs/runs/{run_id}.jsonl")
    manifest: list[dict[str, Any]] = []

    serializable_args = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }
    manifest.append({
        "event": "run_start", "ts": utc_iso(),
        "n_documents": len(documents),
        "engines": engines, "stages": stages,
        "hpc_engines": hpc_engines,
        "args": serializable_args,
    })

    failures: list[str] = []
    common_document_args: list[str] = []
    for s in documents:
        common_document_args += ["--document", s]
    force_args = ["--force"] if args.force else []
    def fail_or_continue(label: str, rc: int) -> bool:
        """Record failure; return True if we should keep going."""
        if rc == 0:
            return True
        failures.append(label)
        if args.strict:
            return False
        return True

    # ---- OCR ----
    if "ocr" in stages:
        # pypdf + docling are coupled in documents_extract.py.
        if "pypdf" in engines or "docling" in engines:
            extra = ["--documents-root", str(args.documents_root)] + force_args
            if "docling" not in engines:
                extra.append("--no-docling")
            if args.use_hpc and "docling" in engines and "docling" not in no_hpc:
                extra.append("--use-hpc-for-docling")
                e = ENGINES["docling"]
                workers = workers_override.get("docling", e["workers"])
                in_flight = in_flight_override.get("docling", e["in_flight"])
                extra += [
                    e["workers_flag"], str(workers),
                    e["in_flight_flag"], str(in_flight),
                    "--hpc-gres", e["gres"],
                    "--hpc-time", compute_hpc_time("docling", len(documents), workers),
                ]
            cmd = [
                "uv", "run", "--script", str(SCRIPT_DIR / ENGINES["pypdf"]["script"]),
            ] + common_document_args + extra
            label = "ocr:" + ("+".join(e for e in ("pypdf", "docling") if e in engines))
            rc = run_subprocess(cmd, label, manifest)
            if not fail_or_continue(label, rc):
                return _finalize_and_exit(manifest_path, manifest, failures, documents, args, rc)

        # One subprocess per served OCR engine.
        for eng in ("olmocr2", "deepseek", "glm", "unlimited_ocr"):
            if eng not in engines:
                continue
            e = ENGINES[eng]
            extra = ["--documents-root", str(args.documents_root)] + force_args
            if args.use_hpc and eng not in no_hpc:
                extra.append(e["hpc_flag"])
                workers = workers_override.get(eng, e["workers"])
                in_flight = in_flight_override.get(eng, e["in_flight"])
                extra += [
                    e["workers_flag"], str(workers),
                    e["in_flight_flag"], str(in_flight),
                    "--hpc-gres", e["gres"],
                    "--hpc-time", compute_hpc_time(eng, len(documents), workers),
                ]
            cmd = [
                "uv", "run", "--script", str(SCRIPT_DIR / e["script"]),
            ] + common_document_args + extra
            rc = run_subprocess(cmd, f"ocr:{eng}", manifest)
            if not fail_or_continue(f"ocr:{eng}", rc):
                return _finalize_and_exit(manifest_path, manifest, failures, documents, args, rc)

    return _finalize_and_exit(manifest_path, manifest, failures, documents, args,
                              1 if failures else 0)


def _finalize_and_exit(manifest_path: Path, manifest: list[dict[str, Any]],
                       failures: list[str], documents: list[str],
                       args: argparse.Namespace, rc: int) -> int:
    # Compute coverage on the run's documents
    per_engine = {
        eng: sum(1 for s in documents if (args.documents_root / s / "ocr" / f"{ENGINES[eng]['ocr_slug']}.txt").exists())
        for eng in ENGINE_ORDER
    }
    manifest.append({
        "event": "run_end", "ts": utc_iso(),
        "failures": failures,
        "coverage_on_run_documents": {"per_engine": per_engine},
    })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        for line in manifest:
            fh.write(json.dumps(line, default=str) + "\n")

    n = len(documents)
    print(f"\nmanifest: {manifest_path}", flush=True)
    print(f"coverage on this run's {n} documents:", flush=True)
    for eng, k in per_engine.items():
        if k:
            print(f"  ocr:{eng:<10} {k}/{n}", flush=True)
    if failures:
        print(f"\nFAILED stages: {', '.join(failures)}", flush=True)
    else:
        print("\nall stages OK", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
