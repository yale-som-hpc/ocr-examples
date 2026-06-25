#!/usr/bin/env python3
"""List or cancel OCR example Slurm service jobs.

This helper is intentionally conservative: cleanup only targets job names used
by this repository's tunneled OCR service launchers.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass


DEFAULT_HOST = os.environ.get("HPC_HOST", "hpc.som.yale.edu")
DEFAULT_USER = os.environ.get("HPC_USER") or os.environ.get("USER", "")
DEFAULT_KEY = os.path.expanduser(os.environ.get("HPC_KEY", ""))

OCR_JOB_PATTERNS = (
    re.compile(r"^docling-serve-"),
    re.compile(r"^vllm-serve-"),
    re.compile(r"^unlimited-ocr-"),
    re.compile(r"^docling_serve$"),
    re.compile(r"^vllm_serve$"),
    re.compile(r"^vllm_serve_app$"),
    re.compile(r"^sglang_serve$"),
)


@dataclass(frozen=True)
class SlurmJob:
    job_id: str
    name: str
    state: str
    elapsed: str
    reason: str


def ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}" if user else host


def ssh_base_args(ssh_key: str) -> list[str]:
    args: list[str] = []
    if ssh_key:
        args.extend(["-i", ssh_key, "-o", "IdentitiesOnly=yes"])
    return args


def run_ssh(
    args: argparse.Namespace, remote_cmd: str
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        *ssh_base_args(args.key),
        "-T",
        "-o",
        "BatchMode=yes",
        ssh_target(args.user, args.host),
        remote_cmd,
    ]
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def parse_squeue(output: str) -> list[SlurmJob]:
    jobs: list[SlurmJob] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 4)
        if len(parts) != 5:
            continue
        jobs.append(SlurmJob(*parts))
    return jobs


def is_ocr_job(job: SlurmJob) -> bool:
    return any(pattern.search(job.name) for pattern in OCR_JOB_PATTERNS)


def list_jobs(args: argparse.Namespace) -> list[SlurmJob]:
    remote_cmd = r'squeue -h -u "$USER" -o "%i\t%j\t%T\t%M\t%R"'
    proc = run_ssh(args, remote_cmd)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return parse_squeue(proc.stdout)


def print_jobs(jobs: list[SlurmJob]) -> None:
    print("JOBID\tNAME\tSTATE\tTIME\tNODELIST(REASON)")
    for job in jobs:
        marker = "ocr" if is_ocr_job(job) else "-"
        print(
            f"{job.job_id}\t{job.name}\t{job.state}\t{job.elapsed}\t"
            f"{job.reason}\t{marker}"
        )


def cleanup_jobs(args: argparse.Namespace) -> int:
    jobs = [job for job in list_jobs(args) if is_ocr_job(job)]
    if not jobs:
        print("No OCR example Slurm jobs found.")
        return 0

    print_jobs(jobs)
    if args.dry_run:
        print("Dry run: not cancelling jobs.")
        return 0

    job_ids = [job.job_id for job in jobs]
    remote_cmd = "scancel " + " ".join(shlex.quote(job_id) for job_id in job_ids)
    proc = run_ssh(args, remote_cmd)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode == 0:
        print("Cancelled OCR example Slurm job(s): " + ", ".join(job_ids))
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "cleanup"))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="with status, show only OCR example jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with cleanup, list matching jobs without cancelling them",
    )
    args = parser.parse_args()

    if args.command == "status":
        jobs = list_jobs(args)
        if args.ocr_only:
            jobs = [job for job in jobs if is_ocr_job(job)]
        print_jobs(jobs)
        return 0
    return cleanup_jobs(args)


if __name__ == "__main__":
    raise SystemExit(main())
