# OCR Engines

These are the OCR engines covered by this example repository. The canonical
engine slugs match `scripts/ocr_provenance.py`.

| Engine | Disk/local path | No persistent HPC disk path |
| --- | --- | --- |
| `pypdf` | Local PDF text extraction with `pypdf` | Not needed; run where the trusted PDF already lives |
| `docling` | Docling CLI on a local/HPC PDF | `docling-serve` through SSH tunnel |
| `olmocr2` | MLX VLM on Apple Silicon | vLLM OpenAI-compatible server through SSH tunnel |
| `deepseek_ocr` | MLX VLM on Apple Silicon | vLLM OpenAI-compatible server through SSH tunnel |
| `glm_ocr` | MLX VLM on Apple Silicon | vLLM OpenAI-compatible server through SSH tunnel |
| `unlimited_ocr` | No local disk backend | SGLang server through SSH tunnel |

## Prepare public sample documents

The sample-document scripts operate on deterministic sample document IDs and expect
each PDF at:

```text
data/documents/<document-id>/document.pdf
```

Create deterministic fake document directories from the public sample PDFs:

```sh
just samples
just prepare-documents
```

That writes `data/samples/documents.txt`, which can be passed to every OCR engine
script with `--from-file`.

## Disk-backed/local smoke tests

These read PDFs from disk and write outputs to disk. Use them only for public
sample PDFs or data that is approved for the storage location.

```sh
uv run --script scripts/documents_process.py \
  --from-file data/samples/documents.txt \
  --stages ocr \
  --engines pypdf \
  --force

uv run --script scripts/documents_process.py \
  --from-file data/samples/documents.txt \
  --stages ocr \
  --engines pypdf,docling \
  --force

uv run --script scripts/olmocr2_extract.py \
  --from-file data/samples/documents.txt \
  --include-text-native \
  --force

uv run --script scripts/deepseek_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --include-text-native \
  --force

uv run --script scripts/glm_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --include-text-native \
  --force
```

Notes:

- `documents_extract.py --no-docling` tests `pypdf` only.
- `documents_extract.py` without `--no-docling` tests local Docling fallback
  on scans.
- `unlimited_ocr` has no local disk backend in these examples; it is an HPC
  SGLang engine.

## No persistent HPC disk smoke tests

The tunneled OCR implementation drives these from the trusted local side:

- Local side reads PDFs from disk.
- It opens SSH to the login node and starts `srun` on a compute node.
- The compute node serves HTTP on a random high port.
- A second SSH connection tunnels localhost to that compute-node port.
- Requests carry page images/PDF uploads in memory.
- Results come back over HTTP and are written on the local side.

PDF bytes and OCR output stay off persistent HPC disk in this workflow. The
service Slurm scripts point request/runtime temp directories at private per-job
directories under `/dev/shm`; model weights, Python dependencies, and container
images may still be cached on HPC scratch/local storage, but those caches should
not contain document bytes or OCR output.

The current OCR commands against the public sample documents are:

```sh
# Use the local SSH alias and the remote repo path synced to the cluster.
export HPC_HOST=hpc
export HPC_USER=
export HPC_KEY=
export HPC_REMOTE_DIR=ocr-examples

# docling
uv run --script scripts/documents_process.py \
  --from-file data/samples/documents.txt \
  --stages ocr \
  --engines pypdf,docling \
  --use-hpc \
  --hpc-workers docling=1 --hpc-in-flight docling=1 \
  --force

# olmOCR-2
uv run --script scripts/olmocr2_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 2 --hpc-gres gpu:a100:1 \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# DeepSeek-OCR-2
uv run --script scripts/deepseek_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 2 --hpc-gres gpu:a100:1 \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# GLM-OCR
uv run --script scripts/glm_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 2 --hpc-gres gpu:a100:1 \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# Baidu Unlimited-OCR
uv run --script scripts/unlimited_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 1 --hpc-gres gpu:a100:1 \
  --hpc-client hpc/client/unlimited_ocr_client.py \
  --include-text-native --force
```

For private PDFs, keep the `--pdf-list` on the trusted local side. Do not copy
the PDFs or OCR outputs to shared HPC storage.
