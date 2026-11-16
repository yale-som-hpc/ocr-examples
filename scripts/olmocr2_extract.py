# /// script
# requires-python = ">=3.11"
# dependencies = [
# ]
# ///
"""Run olmOCR-2 as one of the OCR engines on selected document PDFs.

The --use-hpc backend runs olmOCR-2 through a containerized vLLM Slurm service
and SSH tunnel.

Writes ocr/olmocr2.{txt,json} per document. The sidecar records engine, version,
host, render scale, max_tokens, and pdf_sha256 — downstream consumers discover it
alongside any other engines under ocr/.

By default this skips documents that already have a usable ocr/pypdf.txt, since
pypdf produced clean text and olmOCR-2 buys almost nothing for those. Pass
--include-text-native to run olmOCR-2 on every selected document regardless.

Usage:
  uv run --script scripts/olmocr2_extract.py --document <guid> --use-hpc
  uv run --script scripts/olmocr2_extract.py --from-file /tmp/document_list.txt --use-hpc
  uv run --script scripts/olmocr2_extract.py --all-local --use-hpc --workers 4
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_provenance import (  # noqa: E402
    ENGINE_OLMOCR2,
    ENGINE_PYPDF,
    ocr_sidecar_path,
    ocr_text_path,
    utc_now_iso,
    write_ocr_result,
)

HPC_MODEL = "allenai/olmOCR-2-7B-1025"
HPC_IMAGE = "docker://vllm/vllm-openai:v0.23.0"
ENGINE_VERSION = "7B-1025"
GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document", action="append", default=[])
    p.add_argument("--from-file", type=Path)
    p.add_argument(
        "--all-local",
        action="store_true",
        help="run on every document dir under --documents-root that has document.pdf",
    )
    p.add_argument("--documents-root", type=Path, default=Path("data/documents"))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--include-text-native",
        action="store_true",
        help="Also run olmOCR-2 on documents whose first-pass was text-native. "
        "Default is to skip them (pypdf already extracted clean text).",
    )

    # HPC backend
    p.add_argument(
        "--use-hpc",
        action="store_true",
        help="Run OCR on the SOM HPC cluster via vLLM HTTP.",
    )
    p.add_argument(
        "--workers", type=int, default=1, help="(--use-hpc) parallel Slurm jobs / GPUs"
    )
    p.add_argument(
        "--in-flight",
        type=int,
        default=1,
        help="(--use-hpc) concurrent PDFs per worker",
    )
    p.add_argument(
        "--hpc-client",
        default=str(
            Path(__file__).resolve().parent.parent / "hpc/client/vllm_http_client.py"
        ),
        help="(--use-hpc) path to vllm_http_client.py",
    )
    p.add_argument(
        "--hpc-gres",
        default="gpu:rtx8000:1",
        help="(--use-hpc) Slurm GRES; defaults to one RTX 8000",
    )
    p.add_argument(
        "--hpc-exclude",
        default="c001",
        help="(--use-hpc) Slurm node exclude list, e.g. c001",
    )
    p.add_argument(
        "--hpc-mem", default="64G", help="(--use-hpc) Slurm memory request per worker"
    )
    p.add_argument(
        "--hpc-cpus",
        type=int,
        default=8,
        help="(--use-hpc) Slurm CPU request per worker",
    )
    p.add_argument(
        "--hpc-time",
        default="02:00:00",
        help="(--use-hpc) Slurm time limit per worker",
    )

    args = p.parse_args()
    if not args.use_hpc:
        sys.exit("this repository runs GPU OCR on HPC; pass --use-hpc")

    documents = list(args.document)
    if args.from_file:
        documents += [
            line.strip()
            for line in args.from_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if args.all_local and args.documents_root.exists():
        documents += [
            child.name
            for child in args.documents_root.iterdir()
            if child.is_dir()
            and GUID_RE.match(child.name)
            and (child / "document.pdf").exists()
        ]
    invalid = [value for value in documents if not GUID_RE.fullmatch(value)]
    if invalid:
        sys.exit(f"invalid document GUID(s): {', '.join(invalid)}")
    documents = sorted({value.lower() for value in documents})
    if not documents:
        sys.exit("no document GUIDs supplied")

    # Filter out documents that already have usable pypdf text — saves ~30s of
    # model-load overhead if every remaining document is already done. A present
    # ocr/pypdf.txt means pypdf cleared the threshold; running olmOCR-2 on top
    # buys almost nothing.
    if not args.include_text_native:
        filtered = []
        skip_text_native = 0
        skip_no_pypdf_probe = 0
        for document in documents:
            document_dir = args.documents_root / document
            pypdf_present = ocr_text_path(document_dir, ENGINE_PYPDF).exists()
            pypdf_sidecar = ocr_sidecar_path(document_dir, ENGINE_PYPDF)
            if pypdf_present:
                skip_text_native += 1
                continue
            if not pypdf_sidecar.exists():
                skip_no_pypdf_probe += 1
            filtered.append(document)
        if skip_text_native:
            print(
                f"skipping {skip_text_native} documents with usable ocr/pypdf.txt (pass --include-text-native to override)",
                flush=True,
            )
        if skip_no_pypdf_probe:
            print(
                f"note: {skip_no_pypdf_probe} documents have no pypdf probe — running olmOCR-2 on them anyway",
                flush=True,
            )
        documents = filtered
        if not documents:
            print("no documents left to process after gating; done", flush=True)
            return

    # ---- HPC backend: build a TSV pdf-list and shell out to vllm_http_client.py ----
    jobs_to_run: list[
        tuple[str, Path, Path, Path]
    ] = []  # (document, pdf, sidecar, txt)
    for document in documents:
        document_dir = args.documents_root / document
        pdf_path = document_dir / "document.pdf"
        txt_path = ocr_text_path(document_dir, ENGINE_OLMOCR2)
        sidecar_path = ocr_sidecar_path(document_dir, ENGINE_OLMOCR2)
        if not pdf_path.exists():
            print(f"  ! missing PDF for {document}", flush=True)
            continue
        if txt_path.exists() and sidecar_path.exists() and not args.force:
            continue
        # Ensure ocr/ exists before HPC starts writing.
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_to_run.append(
            (document, pdf_path.resolve(), sidecar_path, txt_path.resolve())
        )
    if not jobs_to_run:
        print("no documents left to process; done", flush=True)
        return

    # Write TSV: <input_pdf>\t<output_md>. vllm_http_client.py writes the
    # text directly; we synthesize sidecars after the subprocess returns.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tsv",
        delete=False,
        encoding="utf-8",
    ) as fh:
        tsv_path = Path(fh.name)
        for _, in_p, _, out_p in jobs_to_run:
            fh.write(f"{in_p}\t{out_p}\n")
    # `--out-dir` is required by the client but unused when every line
    # supplies its own output path; point it at a throwaway scratch dir.
    scratch_dir = Path(tempfile.mkdtemp(prefix="olmocr2_hpc_"))

    cmd = [
        "uv",
        "run",
        "--with",
        "openai>=1.40",
        "--with",
        "httpx>=0.27",
        "--with",
        "pypdfium2>=4.30",
        "--with",
        "pillow>=11",
        args.hpc_client,
        "--pdf-list",
        str(tsv_path),
        "--out-dir",
        str(scratch_dir),
        "--workers",
        str(args.workers),
        "--in-flight",
        str(args.in_flight),
        "--gres",
        args.hpc_gres,
        "--mem",
        args.hpc_mem,
        "--cpus-per-task",
        str(args.hpc_cpus),
        "--time",
        args.hpc_time,
        "--max-tokens",
        str(args.max_tokens),
        "--model",
        HPC_MODEL,
        "--slurm-script",
        "hpc/slurm/vllm_serve_apptainer.slurm",
        "--image",
        HPC_IMAGE,
    ]
    if args.hpc_exclude:
        cmd.extend(["--exclude", args.hpc_exclude])
    if args.force:
        cmd.append("--force")
    print(
        f"running HPC OCR on {len(jobs_to_run)} document(s) via "
        f"{args.workers} worker(s) × {args.in_flight} in-flight",
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
            pass  # leftover files from no-pages PDFs etc; not worth chasing
    run_finished = utc_now_iso()

    # Synthesize sidecars for the .txt files the HPC client just wrote.
    # We don't have per-document timing (the client only logs aggregate
    # throughput); the run-level start/finish bracket every document.
    host_label = f"hpc-vllm-{args.hpc_gres.replace(':', '-')}"
    written = 0
    for document, pdf_p, sidecar_p, txt_p in jobs_to_run:
        if not txt_p.exists():
            continue
        text = txt_p.read_text(encoding="utf-8", errors="replace")
        document_dir = args.documents_root / document
        write_ocr_result(
            document_dir,
            ENGINE_OLMOCR2,
            text,
            provenance={
                "engine_version": ENGINE_VERSION,
                "method": "vlm-vision",
                "host": host_label,
                "started_at": run_started,
                "finished_at": run_finished,
                "params": {
                    "scale": 1.5,  # HPC client default
                    "max_tokens": args.max_tokens,
                    "model": HPC_MODEL,
                    "image": HPC_IMAGE,
                    "workers": args.workers,
                    "in_flight": args.in_flight,
                    "gres": args.hpc_gres,
                    "note": "per-document timing not recorded; bracket spans the full run",
                },
            },
        )
        written += 1
    print(
        f"\nHPC backend rc={rc}; wrote {written}/{len(jobs_to_run)} (.txt + .json)",
        flush=True,
    )
    sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()
