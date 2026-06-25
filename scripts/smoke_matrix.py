#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Run the OCR smoke-test matrix across engines, modes, and GPU classes."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ENGINES = ("pypdf", "docling", "olmocr2", "deepseek_ocr", "glm_ocr", "unlimited_ocr")
MODES = ("disk", "tunnel")
GPU_PROFILES = ("rtx", "a100")

DISK_ENGINES = {"pypdf", "docling", "olmocr2", "deepseek_ocr", "glm_ocr"}
TUNNEL_ENGINES = {"docling", "olmocr2", "deepseek_ocr", "glm_ocr", "unlimited_ocr"}

GPU_GRES = {
    "rtx": "gpu:rtx8000:1",
    "a100": "gpu:a100:1",
}
GPU_EXCLUDE = {
    "rtx": "c001",
    "a100": "",
}


@dataclass(frozen=True)
class MatrixCase:
    engine: str
    mode: str
    gpu: str

    @property
    def label(self) -> str:
        return f"{self.engine}-{self.mode}-{self.gpu}"


@dataclass
class MatrixResult:
    case: MatrixCase
    status: str
    rc: int
    elapsed_s: float
    reason: str
    command: list[str]


ACTIVE_PROCS: set[subprocess.Popen] = set()


def split_csv(value: str, valid: tuple[str, ...], label: str) -> list[str]:
    if value == "all":
        return list(valid)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    bad = [part for part in parts if part not in valid]
    if bad:
        raise SystemExit(
            f"unknown {label}: {','.join(bad)}. valid: all,{','.join(valid)}"
        )
    if not parts:
        raise SystemExit(f"{label} list is empty")
    return parts


def unsupported_reason(case: MatrixCase) -> str | None:
    if case.mode == "disk" and case.engine not in DISK_ENGINES:
        return "no disk/local backend in this repo"
    if case.mode == "tunnel" and case.engine not in TUNNEL_ENGINES:
        return "no tunneled HPC service backend in this repo"
    return None


def gpu_gres(args: argparse.Namespace, gpu: str) -> str:
    if gpu == "rtx":
        return args.rtx_gres
    if gpu == "a100":
        return args.a100_gres
    raise AssertionError(gpu)


def gpu_exclude(args: argparse.Namespace, gpu: str) -> str:
    if gpu == "rtx":
        return args.rtx_exclude
    if gpu == "a100":
        return args.a100_exclude
    raise AssertionError(gpu)


def smoke_command(case: MatrixCase, args: argparse.Namespace) -> list[str]:
    cmd = [
        "uv",
        "run",
        "scripts/smoke_tests.py",
        "--engine",
        case.engine,
        "--mode",
        case.mode,
        "--sample-count",
        str(args.sample_count),
        "--documents-root",
        str(args.documents_root),
        "--smoke-list",
        str(args.smoke_list),
        "--out-dir",
        str(args.out_dir / case.gpu),
        "--max-tokens",
        str(args.max_tokens),
        "--workers",
        str(args.workers),
        "--in-flight",
        str(args.in_flight),
        "--parallel-tunnel",
        "1",
        "--hpc-gres",
        gpu_gres(args, case.gpu),
        "--hpc-mem",
        args.hpc_mem,
        "--hpc-cpus",
        str(args.hpc_cpus),
        "--hpc-time",
        args.hpc_time,
        "--disk-timeout",
        str(args.disk_timeout),
        "--tunnel-timeout",
        str(args.tunnel_timeout),
    ]
    if args.from_file:
        cmd += ["--from-file", str(args.from_file)]
    if args.force_prepare:
        cmd.append("--force-prepare")
    if args.no_cleanup_on_failure:
        cmd.append("--no-cleanup-on-failure")
    exclude = gpu_exclude(args, case.gpu)
    if case.mode == "tunnel" and exclude:
        cmd += ["--hpc-exclude", exclude]
    return cmd


def run_case(case: MatrixCase, args: argparse.Namespace) -> MatrixResult:
    reason = unsupported_reason(case)
    if reason:
        print(f"\n=== {case.label} SKIP: {reason} ===", flush=True)
        return MatrixResult(case, "skip", 0, 0.0, reason, [])

    cmd = smoke_command(case, args)
    if args.dry_run:
        print(f"\n=== {case.label} DRY-RUN ===", flush=True)
        print("$ " + " ".join(cmd), flush=True)
        return MatrixResult(case, "dry-run", 0, 0.0, "", cmd)

    print(f"\n=== matrix:{case.label} ===", flush=True)
    if case.mode == "disk":
        print(
            f"note: disk mode does not request a Slurm GPU; gpu={case.gpu} is "
            "a matrix label only",
            flush=True,
        )
    print("$ " + " ".join(cmd), flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(cmd, start_new_session=True)
    ACTIVE_PROCS.add(proc)
    try:
        rc = proc.wait()
    finally:
        ACTIVE_PROCS.discard(proc)
    elapsed = time.monotonic() - started
    status = "pass" if rc == 0 else "fail"
    print(
        f"=== matrix:{case.label} {status.upper()} exit={rc} "
        f"elapsed={elapsed:.1f}s ===",
        flush=True,
    )
    return MatrixResult(case, status, rc, elapsed, "", cmd)


def terminate_active_processes() -> None:
    for proc in list(ACTIVE_PROCS):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def write_report(results: list[MatrixResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=(
                "engine",
                "mode",
                "gpu",
                "status",
                "rc",
                "elapsed_s",
                "reason",
                "command",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "engine": result.case.engine,
                    "mode": result.case.mode,
                    "gpu": result.case.gpu,
                    "status": result.status,
                    "rc": result.rc,
                    "elapsed_s": f"{result.elapsed_s:.1f}",
                    "reason": result.reason,
                    "command": " ".join(result.command),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        default="all",
        help="comma-separated engine list or all",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help="comma-separated mode list or all",
    )
    parser.add_argument(
        "--gpus",
        default="all",
        help="comma-separated GPU profile list or all. Valid profiles: rtx,a100",
    )
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    parser.add_argument("--from-file", type=Path)
    parser.add_argument(
        "--smoke-list", type=Path, default=Path("data/samples/smoke-documents.txt")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/smoke-matrix"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/smoke-matrix/report.tsv")
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-flight", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--rtx-gres", default=GPU_GRES["rtx"])
    parser.add_argument("--a100-gres", default=GPU_GRES["a100"])
    parser.add_argument("--rtx-exclude", default=GPU_EXCLUDE["rtx"])
    parser.add_argument("--a100-exclude", default=GPU_EXCLUDE["a100"])
    parser.add_argument("--hpc-mem", default="64G")
    parser.add_argument("--hpc-cpus", type=int, default=8)
    parser.add_argument("--hpc-time", default="02:00:00")
    parser.add_argument("--disk-timeout", type=int, default=600)
    parser.add_argument("--tunnel-timeout", type=int, default=1800)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--no-cleanup-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.engines = split_csv(args.engines, ENGINES, "engine")
    args.modes = split_csv(args.modes, MODES, "mode")
    args.gpus = split_csv(args.gpus, GPU_PROFILES, "GPU profile")
    if args.sample_count < 1:
        raise SystemExit("--sample-count must be >= 1")
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    cases = [
        MatrixCase(engine, mode, gpu)
        for engine in args.engines
        for mode in args.modes
        for gpu in args.gpus
    ]
    print(f"matrix cases: {len(cases)}", flush=True)

    results: list[MatrixResult] = []
    try:
        if args.parallel == 1:
            for case in cases:
                results.append(run_case(case, args))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel
            ) as pool:
                future_to_case = {
                    pool.submit(run_case, case, args): case for case in cases
                }
                for future in concurrent.futures.as_completed(future_to_case):
                    results.append(future.result())
    except KeyboardInterrupt:
        terminate_active_processes()
        raise
    finally:
        if results:
            write_report(results, args.report)
            print(f"\nreport: {args.report}", flush=True)

    failures = [result.case.label for result in results if result.status == "fail"]
    skips = [result.case.label for result in results if result.status == "skip"]
    print("\n=== smoke matrix summary ===", flush=True)
    if skips:
        print("skipped: " + ", ".join(skips), flush=True)
    if failures:
        print("failed:  " + ", ".join(failures), flush=True)
        return 1
    print("passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
