#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Run OCR engine smoke tests on one public sample document.

Modes:
  disk    reads a PDF from local disk and writes outputs locally
  tunnel  reads a local PDF, starts the HPC service with Slurm, and sends
          request bytes over the SSH tunnel

Unsupported combinations are skipped in the full matrix:
  pypdf tunnel         pypdf is local text extraction, no service backend
  unlimited_ocr disk   Unlimited-OCR is CUDA/SGLang-only in these examples
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ENGINES = ("pypdf", "docling", "olmocr2", "deepseek_ocr", "glm_ocr", "unlimited_ocr")
MODES = ("disk", "tunnel")

DISK_ENGINES = {"pypdf", "docling", "olmocr2", "deepseek_ocr", "glm_ocr"}
TUNNEL_ENGINES = {"docling", "olmocr2", "deepseek_ocr", "glm_ocr", "unlimited_ocr"}
DEFAULT_HPC_GRES = "gpu:1"
UNLIMITED_OCR_DEFAULT_HPC_GRES = "gpu:a100:1"

ENGINE_ALIASES = {
    "deepseek": "deepseek_ocr",
    "glm": "glm_ocr",
    "unlimited": "unlimited_ocr",
}

ACTIVE_PROCS: set[subprocess.Popen] = set()
ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class SmokeCase:
    engine: str
    mode: str


def normalize_engine(engine: str) -> str:
    engine = ENGINE_ALIASES.get(engine, engine)
    if engine != "all" and engine not in ENGINES:
        raise SystemExit(f"unknown engine: {engine}. valid: all,{','.join(ENGINES)}")
    return engine


def normalize_mode(mode: str) -> str:
    if mode != "all" and mode not in MODES:
        raise SystemExit(f"unknown mode: {mode}. valid: all,{','.join(MODES)}")
    return mode


def run(cmd: list[str], *, label: str, timeout: int | None = None) -> int:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(cmd, start_new_session=True)
    with ACTIVE_LOCK:
        ACTIVE_PROCS.add(proc)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        print(
            f"=== {label} TIMEOUT after {elapsed:.1f}s; terminating process group ===",
            flush=True,
        )
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        return 124
    finally:
        with ACTIVE_LOCK:
            ACTIVE_PROCS.discard(proc)
    elapsed = time.monotonic() - started
    print(f"=== {label} exit={rc} elapsed={elapsed:.1f}s ===", flush=True)
    return rc


def terminate_active_processes() -> None:
    with ACTIVE_LOCK:
        procs = list(ACTIVE_PROCS)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def run_case(
    case: SmokeCase, args: argparse.Namespace, ids: list[str]
) -> tuple[str, int]:
    label = f"{case.engine}:{case.mode}"
    if case.mode == "disk":
        cmd = disk_command(case, args, ids[0])
    else:
        cmd = tunnel_command(case, args, ids)
    timeout = args.disk_timeout if case.mode == "disk" else args.tunnel_timeout
    rc = run(cmd, label=f"smoke:{label}", timeout=timeout)
    if rc == 0 and not args.no_validate:
        rc = validate_case(case, args, ids)
    return label, rc


def read_document_ids(path: Path, limit: int) -> list[str]:
    if not path.exists():
        raise SystemExit(f"document list not found: {path}")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
        if limit > 0 and len(ids) >= limit:
            break
    if not ids:
        raise SystemExit(f"no document ids found in {path}")
    return ids


def ensure_sample_documents(args: argparse.Namespace) -> tuple[list[str], Path]:
    """Return smoke document ids and the list file to pass to engine wrappers."""
    if args.from_file:
        ids = read_document_ids(args.from_file, args.sample_count)
        smoke_list = args.smoke_list
        smoke_list.parent.mkdir(parents=True, exist_ok=True)
        smoke_list.write_text("\n".join(ids) + "\n", encoding="utf-8")
        return ids, smoke_list

    rc = run(
        ["uv", "run", "scripts/download_sample_documents.py"],
        label="prepare:download-samples",
    )
    if rc != 0:
        raise SystemExit(rc)

    prepare_cmd = [
        "uv",
        "run",
        "scripts/prepare_sample_documents.py",
        "--documents-root",
        str(args.documents_root),
        "--from-file",
        str(args.smoke_list),
        "--limit",
        str(args.sample_count),
    ]
    if args.force_prepare:
        prepare_cmd.append("--force")
    rc = run(prepare_cmd, label="prepare:document-layout")
    if rc != 0:
        raise SystemExit(rc)

    ids = read_document_ids(args.smoke_list, args.sample_count)
    return ids, args.smoke_list


def selected_cases(args: argparse.Namespace) -> list[SmokeCase]:
    engines = ENGINES if args.engine == "all" else (args.engine,)
    modes = MODES if args.mode == "all" else (args.mode,)
    return [SmokeCase(engine, mode) for mode in modes for engine in engines]


def unsupported_reason(case: SmokeCase) -> str | None:
    if case.mode == "disk" and case.engine not in DISK_ENGINES:
        return "no disk/local backend in this repo"
    if case.mode == "tunnel" and case.engine not in TUNNEL_ENGINES:
        return "no tunneled HPC service backend in this repo"
    return None


def disk_command(
    case: SmokeCase, args: argparse.Namespace, document_id: str
) -> list[str]:
    pdf = args.documents_root / document_id / "document.pdf"
    if not pdf.exists():
        raise SystemExit(f"sample PDF not found: {pdf}")
    outdir = args.out_dir / "disk"
    return [
        "uv",
        "run",
        "scripts/ocr_engine_disk.py",
        str(pdf),
        "--engine",
        case.engine,
        "--outdir",
        str(outdir),
        "--max-tokens",
        str(args.max_tokens),
    ]


def disk_output_path(
    case: SmokeCase, args: argparse.Namespace, document_id: str
) -> Path:
    pdf = args.documents_root / document_id / "document.pdf"
    return args.out_dir / "disk" / f"{pdf.stem}.{case.engine}.txt"


def canonical_output_path(
    case: SmokeCase, args: argparse.Namespace, document_id: str
) -> Path:
    return args.documents_root / document_id / "ocr" / f"{case.engine}.txt"


def validate_one(
    *,
    case: SmokeCase,
    args: argparse.Namespace,
    document_id: str,
    output: Path,
) -> int:
    sample_json = args.documents_root / document_id / "sample.json"
    cmd = [
        "uv",
        "run",
        "scripts/validate_ocr_output.py",
        "--engine",
        case.engine,
        "--mode",
        case.mode,
        "--gpu",
        args.gpu_label,
        "--sample-json",
        str(sample_json),
        "--output",
        str(output),
        "--expectations",
        str(args.expectations),
    ]
    return run(cmd, label=f"validate:{case.engine}:{case.mode}:{document_id}")


def validate_case(case: SmokeCase, args: argparse.Namespace, ids: list[str]) -> int:
    if case.mode == "disk":
        return validate_one(
            case=case,
            args=args,
            document_id=ids[0],
            output=disk_output_path(case, args, ids[0]),
        )

    rc = 0
    for document_id in ids:
        one_rc = validate_one(
            case=case,
            args=args,
            document_id=document_id,
            output=canonical_output_path(case, args, document_id),
        )
        if one_rc != 0:
            rc = one_rc
    return rc


def tunnel_hpc_gres(case: SmokeCase, args: argparse.Namespace) -> str:
    if case.engine == "unlimited_ocr" and args.hpc_gres == DEFAULT_HPC_GRES:
        return UNLIMITED_OCR_DEFAULT_HPC_GRES
    return args.hpc_gres


def tunnel_command(
    case: SmokeCase, args: argparse.Namespace, ids: list[str]
) -> list[str]:
    hpc_gres = tunnel_hpc_gres(case, args)
    common = [
        "--documents-root",
        str(args.documents_root),
        "--from-file",
        str(args.smoke_list),
        "--force",
    ]
    if case.engine == "docling":
        cmd = [
            "uv",
            "run",
            "--script",
            "scripts/documents_extract.py",
            "--documents-root",
            str(args.documents_root),
            "--use-hpc-for-docling",
            "--hpc-workers",
            str(args.workers),
            "--hpc-in-flight",
            str(args.in_flight),
            "--hpc-gres",
            hpc_gres,
            "--hpc-mem",
            args.hpc_mem,
            "--hpc-cpus",
            str(args.hpc_cpus),
            "--hpc-time",
            args.hpc_time,
            "--text-threshold",
            str(args.docling_text_threshold),
            "--force",
        ]
        if args.hpc_exclude:
            cmd.extend(["--hpc-exclude", args.hpc_exclude])
        for document_id in ids:
            cmd += ["--document", document_id]
        return cmd

    script = {
        "olmocr2": "scripts/olmocr2_extract.py",
        "deepseek_ocr": "scripts/deepseek_ocr_extract.py",
        "glm_ocr": "scripts/glm_ocr_extract.py",
        "unlimited_ocr": "scripts/unlimited_ocr_extract.py",
    }[case.engine]
    cmd = [
        "uv",
        "run",
        "--script",
        script,
        *common,
        "--use-hpc",
        "--workers",
        str(args.workers),
        "--in-flight",
        str(args.in_flight),
        "--hpc-gres",
        hpc_gres,
        "--hpc-mem",
        args.hpc_mem,
        "--hpc-cpus",
        str(args.hpc_cpus),
        "--hpc-time",
        args.hpc_time,
        "--include-text-native",
    ]
    if args.hpc_exclude:
        cmd.extend(["--hpc-exclude", args.hpc_exclude])
    return cmd


def cleanup_hpc_jobs() -> int:
    return run(
        ["uv", "run", "scripts/hpc_jobs.py", "cleanup"], label="cleanup:hpc-ocr-jobs"
    )


def hpc_status() -> int:
    return run(
        ["uv", "run", "scripts/hpc_jobs.py", "status", "--ocr-only"],
        label="status:hpc-ocr-jobs",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        default="all",
        help="engine slug or all. Aliases: deepseek, glm, unlimited",
    )
    parser.add_argument("--mode", default="all", choices=("all", "disk", "tunnel"))
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="number of public sample documents to test",
    )
    parser.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    parser.add_argument(
        "--from-file",
        type=Path,
        help="optional existing document-id list; otherwise public samples are prepared",
    )
    parser.add_argument(
        "--smoke-list",
        type=Path,
        default=Path("data/samples/smoke-documents.txt"),
        help="document-id list written/read by smoke tests",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/smoke"))
    parser.add_argument(
        "--expectations", type=Path, default=Path("examples/expected-ocr.json")
    )
    parser.add_argument(
        "--gpu-label",
        default="-",
        help="label used in validation output; matrix runner passes rtx or a100",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip expected-content validation after successful OCR",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="disk-mode MLX max token cap for quick smoke tests",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-flight", type=int, default=1)
    parser.add_argument(
        "--parallel-tunnel",
        type=int,
        default=1,
        help="number of tunnel smoke tests to run at once. Use >1 only when "
        "the GPU partition has room.",
    )
    parser.add_argument(
        "--hpc-gres",
        default=DEFAULT_HPC_GRES,
        help="Slurm GPU request for tunnel smoke tests. Use gpu:rtx8000:1 "
        "for RTX-capable engines. If omitted, Unlimited-OCR smoke tests "
        "request gpu:a100:1 because the current SGLang backend needs it.",
    )
    parser.add_argument(
        "--hpc-exclude",
        default="",
        help="Slurm node exclude list for tunnel smoke tests, e.g. c001",
    )
    parser.add_argument(
        "--hpc-mem",
        default="64G",
        help="Slurm memory request per tunnel worker. Docling has its own lower default "
        "unless this shared smoke-test override is set.",
    )
    parser.add_argument(
        "--hpc-cpus", type=int, default=8, help="Slurm CPU request per tunnel worker"
    )
    parser.add_argument("--hpc-time", default="02:00:00")
    parser.add_argument(
        "--docling-text-threshold",
        type=int,
        default=1_000_000,
        help="force docling smoke paths to run even on text-native PDFs",
    )
    parser.add_argument(
        "--disk-timeout",
        type=int,
        default=600,
        help="per disk smoke timeout in seconds",
    )
    parser.add_argument(
        "--tunnel-timeout",
        type=int,
        default=1800,
        help="per tunnel smoke timeout in seconds",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="overwrite sample document.pdf copies during preparation",
    )
    parser.add_argument(
        "--no-cleanup-on-failure",
        action="store_true",
        help="do not call scripts/hpc_jobs.py cleanup after a failed tunnel smoke",
    )
    args = parser.parse_args()
    args.engine = normalize_engine(args.engine)
    args.mode = normalize_mode(args.mode)
    if args.sample_count < 1:
        raise SystemExit("--sample-count must be >= 1")
    if args.parallel_tunnel < 1:
        raise SystemExit("--parallel-tunnel must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    cases = selected_cases(args)
    exact_single = args.engine != "all" and args.mode != "all"
    if exact_single:
        reason = unsupported_reason(cases[0])
        if reason:
            print(
                f"=== {cases[0].engine}:{cases[0].mode} SKIP: {reason} ===", flush=True
            )
            return 0

    ids, _ = ensure_sample_documents(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    skips: list[str] = []
    disk_cases: list[SmokeCase] = []
    tunnel_cases: list[SmokeCase] = []

    for case in cases:
        label = f"{case.engine}:{case.mode}"
        reason = unsupported_reason(case)
        if reason:
            print(f"\n=== {label} SKIP: {reason} ===", flush=True)
            skips.append(label)
            if exact_single:
                return 0
            continue
        if case.mode == "disk":
            disk_cases.append(case)
        else:
            tunnel_cases.append(case)

    for case in disk_cases:
        label, rc = run_case(case, args, ids)
        if rc != 0:
            failures.append(label)

    if tunnel_cases and args.parallel_tunnel == 1:
        for case in tunnel_cases:
            label, rc = run_case(case, args, ids)
            if rc != 0:
                failures.append(label)
                if not args.no_cleanup_on_failure:
                    cleanup_hpc_jobs()
    elif tunnel_cases:
        print(
            f"\n=== running {len(tunnel_cases)} tunnel smoke tests with "
            f"parallel_tunnel={args.parallel_tunnel} ===",
            flush=True,
        )
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_tunnel)
        try:
            future_to_case = {
                pool.submit(run_case, case, args, ids): case for case in tunnel_cases
            }
            for future in concurrent.futures.as_completed(future_to_case):
                label, rc = future.result()
                if rc != 0:
                    failures.append(label)
        except KeyboardInterrupt:
            terminate_active_processes()
            pool.shutdown(wait=False, cancel_futures=True)
            if not args.no_cleanup_on_failure:
                cleanup_hpc_jobs()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if failures and not args.no_cleanup_on_failure:
            cleanup_hpc_jobs()

    if tunnel_cases:
        hpc_status()

    print("\n=== smoke summary ===", flush=True)
    if skips:
        print("skipped: " + ", ".join(skips), flush=True)
    if failures:
        print("failed:  " + ", ".join(failures), flush=True)
        return 1
    print("passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
