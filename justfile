# show available recipes
default:
    @just --list

# check local command-line tools used by the examples
check:
    @command -v uv >/dev/null && echo "uv: $(command -v uv)" || echo "uv: missing"
    @command -v ssh >/dev/null && echo "ssh: $(command -v ssh)" || echo "ssh: missing"
    @echo "HPC container runtime check: ssh hpc 'module load apptainer >/dev/null 2>&1 || true; apptainer --version'"

# download public OCRmyPDF sample documents and write data/samples/manifest.txt
samples:
    uv run scripts/download_sample_documents.py

# copy public PDFs into data/documents/<fake-guid>/document.pdf for engine scripts
prepare-documents:
    uv run scripts/prepare_sample_documents.py

# sync this repo to ~/ocr-examples on HPC without touching remote caches/results
sync-hpc:
    rsync -az --delete --exclude .uv-cache --exclude results --exclude __pycache__ --exclude '*.pyc' ./ hpc:~/ocr-examples/

# show current Slurm queue entries, marking OCR example service jobs
hpc-status *args:
    uv run scripts/hpc_jobs.py status {{args}}

# cancel only OCR example Slurm service jobs; pass --dry-run to preview
hpc-cleanup *args:
    uv run scripts/hpc_jobs.py cleanup {{args}}

# run the document-oriented OCR engine orchestrator directly
documents-process *args:
    uv run --script scripts/documents_process.py {{args}}

# pypdf/docling extraction against public sample documents
engine-extract engines="pypdf" *args:
    uv run --script scripts/documents_process.py --from-file data/samples/documents.txt --stages ocr --engines {{quote(engines)}} {{args}}

# local/HPC olmOCR-2 extraction against public sample documents
engine-olmocr2 *args:
    uv run --script scripts/olmocr2_extract.py --from-file data/samples/documents.txt {{args}}

# local/HPC DeepSeek-OCR-2 extraction against public sample documents
engine-deepseek *args:
    uv run --script scripts/deepseek_ocr_extract.py --from-file data/samples/documents.txt {{args}}

# local/HPC GLM-OCR extraction against public sample documents
engine-glm *args:
    uv run --script scripts/glm_ocr_extract.py --from-file data/samples/documents.txt {{args}}

# HPC Unlimited-OCR extraction against public sample documents
engine-unlimited *args:
    uv run --script scripts/unlimited_ocr_extract.py --from-file data/samples/documents.txt {{args}}

# Syntax-check scripts that do not need OCR engines or Python dependencies
test:
    uv run python -c 'import ast, pathlib; files=("scripts/download_sample_documents.py","scripts/prepare_sample_documents.py","scripts/ocr_engine_disk.py","scripts/ocr_provenance.py","scripts/hpc_jobs.py","scripts/documents_extract.py","scripts/olmocr2_extract.py","scripts/deepseek_ocr_extract.py","scripts/glm_ocr_extract.py","scripts/unlimited_ocr_extract.py","scripts/documents_process.py","hpc/client/vllm_http_client.py","hpc/client/docling_http_client.py","hpc/client/unlimited_ocr_client.py"); [ast.parse(pathlib.Path(f).read_text(), filename=f) for f in files]'
    bash -n hpc/slurm/vllm_serve_apptainer.slurm hpc/slurm/docling_serve.slurm hpc/slurm/sglang_serve.slurm hpc/bin/bootstrap.sh hpc/bin/sync.sh
