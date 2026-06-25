# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-vlm>=0.3.11; platform_system == 'Darwin' and platform_machine == 'arm64'",
#     "pypdfium2>=4.30",
# ]
# ///
"""Run olmOCR-2 as one of the OCR engines on selected document PDFs.

Two backends:
  - default: local MLX (mlx-community/olmOCR-2-7B-1025-4bit). Good for laptop
    OCR of a handful of documents. ~30-60s/PDF.
  - --use-hpc: outsource to the Yale SOM HPC vLLM cluster via the HTTP-tunneled
    pipeline in hpc/client/vllm_http_client.py. Good for bulk runs.
    Throughput ~120 PDFs/min on 4 A100 workers, ~9 min for the full corpus.

Writes ocr/olmocr2.{txt,json} per document. The sidecar records engine, version,
host, render scale, max_tokens, and pdf_sha256 — downstream consumers discover it
alongside any other engines under ocr/.

By default this skips documents that already have a usable ocr/pypdf.txt, since
pypdf produced clean text and olmOCR-2 buys almost nothing for those. Pass
--include-text-native to run olmOCR-2 on every selected document regardless.

Usage:
  uv run --script scripts/olmocr2_extract.py --document <guid> [--document <guid>]...
  uv run --script scripts/olmocr2_extract.py --from-file /tmp/document_list.txt
  uv run --script scripts/olmocr2_extract.py --all-local
  uv run --script scripts/olmocr2_extract.py --all-local --use-hpc --workers 4
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_provenance import (  # noqa: E402
    ENGINE_OLMOCR2,
    ENGINE_PYPDF,
    ocr_sidecar_path,
    ocr_text_path,
    utc_now_iso,
    write_ocr_result,
)

MODEL = "mlx-community/olmOCR-2-7B-1025-4bit"
HPC_MODEL = "allenai/olmOCR-2-7B-1025"
HPC_IMAGE = "docker://vllm/vllm-openai:v0.23.0"
ENGINE_VERSION = "7B-1025"
GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

PROMPT = (
    "Below is one page of a document. Reproduce its content as plain "
    "text + markdown, preserving every detail faithfully:\n"
    "- Tables with all rows and columns\n"
    "- Labels, headings, stamps, handwriting, marks, and symbols where visible\n"
    "- Math/equations as LaTeX where appropriate\n"
    "- Legends, scale definitions, narrative remarks, and footnotes — full text\n"
    "Do not summarize, paraphrase, or omit information. Skip pure page-number/header/footer chrome."
)


def render_pages(pdf_path: Path, scale: float = 2.0):
    doc = pdfium.PdfDocument(str(pdf_path))
    images = []
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()
        images.append(pil)
        page.close()
    doc.close()
    return images


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
    p.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="pdfium render scale (local MLX path). HPC path uses its own default of 1.5.",
    )
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
        help="Outsource OCR to the SOM HPC cluster via vLLM HTTP. "
        "Faster (~120 PDFs/min on 4 A100) than local MLX (~30-60s/PDF).",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="(--use-hpc) parallel Slurm jobs / GPUs"
    )
    p.add_argument(
        "--in-flight",
        type=int,
        default=24,
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
        default="gpu:1",
        help="(--use-hpc) Slurm GRES; 'gpu:1' = any GPU, 'gpu:a100:1' = A100 only",
    )
    p.add_argument(
        "--hpc-exclude",
        default="",
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
        help="(--use-hpc) Slurm time limit per worker. Full corpus "
        "is ~20 min on 4 A100; 2h is generous headroom.",
    )

    args = p.parse_args()

    documents = list(args.document)
    if args.from_file:
        documents += [
            line.strip()
            for line in args.from_file.read_text().splitlines()
            if line.strip()
        ]
    if args.all_local and args.documents_root.exists():
        documents += [
            child.name
            for child in args.documents_root.iterdir()
            if child.is_dir()
            and GUID_RE.match(child.name)
            and (child / "document.pdf").exists()
        ]
    documents = sorted({s.lower() for s in documents if GUID_RE.match(s)})
    if not documents:
        sys.exit("no valid document GUIDs supplied")

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
    if args.use_hpc:
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

    # ---- Local MLX backend ----
    # Imports deferred so --use-hpc works without mlx-vlm installed.
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"loading {MODEL}…", flush=True)
    t0 = time.time()
    model, processor = load(MODEL)
    config = load_config(MODEL)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    written = skipped = errors = 0
    for document in documents:
        document_dir = args.documents_root / document
        pdf_path = document_dir / "document.pdf"
        txt_path = ocr_text_path(document_dir, ENGINE_OLMOCR2)
        sidecar_path = ocr_sidecar_path(document_dir, ENGINE_OLMOCR2)
        if not pdf_path.exists():
            print(f"  ! missing PDF for {document}", flush=True)
            errors += 1
            continue
        if txt_path.exists() and sidecar_path.exists() and not args.force:
            print(f"  skip {document} (ocr/olmocr2.* exists)", flush=True)
            skipped += 1
            continue
        t_document = time.time()
        started = utc_now_iso()
        try:
            images = render_pages(pdf_path, scale=args.scale)
        except Exception as exc:
            print(f"  ! render error {document}: {exc}", flush=True)
            errors += 1
            continue
        page_texts = []
        for idx, img in enumerate(images, 1):
            formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
            t_pg = time.time()
            result = generate(
                model,
                processor,
                formatted,
                image=[img],
                max_tokens=args.max_tokens,
                verbose=False,
            )
            text = result.text if hasattr(result, "text") else str(result)
            page_texts.append(f"<!-- page {idx} -->\n{text.strip()}\n")
            print(
                f"    {document} pg{idx}/{len(images)}  "
                f"chars={len(text):,}  {time.time() - t_pg:.1f}s",
                flush=True,
            )
        joined = "\n\n".join(page_texts)
        elapsed = round(time.time() - t_document, 3)
        write_ocr_result(
            document_dir,
            ENGINE_OLMOCR2,
            joined,
            provenance={
                "engine_version": ENGINE_VERSION,
                "method": "vlm-vision",
                "host": "local-mlx",
                "pdf_pages": len(images),
                "started_at": started,
                "finished_at": utc_now_iso(),
                "elapsed_seconds": elapsed,
                "params": {
                    "scale": args.scale,
                    "max_tokens": args.max_tokens,
                    "model": MODEL,
                },
            },
        )
        print(
            f"  ok {document}  pages={len(images)}  "
            f"chars={len(joined):,}  {elapsed:.1f}s",
            flush=True,
        )
        written += 1

    print(f"\nwritten={written} skipped={skipped} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
