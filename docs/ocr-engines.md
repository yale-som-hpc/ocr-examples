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

## Engine and runtime links

| Engine/runtime | Upstream links |
| --- | --- |
| `pypdf` | [documentation](https://pypdf.readthedocs.io/), [GitHub](https://github.com/py-pdf/pypdf) |
| `docling` | [documentation](https://docling-project.github.io/docling/), [GitHub](https://github.com/docling-project/docling), [Docling Serve](https://github.com/docling-project/docling-serve) |
| `olmocr2` | [olmOCR GitHub](https://github.com/allenai/olmocr), [Hugging Face model](https://huggingface.co/allenai/olmOCR-2-7B-1025), [paper](https://arxiv.org/abs/2510.19817) |
| `deepseek_ocr` | [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2), [Hugging Face model](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2), [paper](https://arxiv.org/abs/2601.20552) |
| `glm_ocr` | [Hugging Face model](https://huggingface.co/zai-org/GLM-OCR), [Z.ai Hugging Face org](https://huggingface.co/zai-org), [Z.ai GitHub org](https://github.com/zai-org) |
| `unlimited_ocr` | [Baidu Unlimited-OCR GitHub](https://github.com/baidu/Unlimited-OCR), [Hugging Face model](https://huggingface.co/baidu/Unlimited-OCR) |
| `vllm` | [documentation](https://docs.vllm.ai/), [GitHub](https://github.com/vllm-project/vllm), [project site](https://vllm.ai/) |
| `sglang` | [documentation](https://docs.sglang.ai/), [GitHub](https://github.com/sgl-project/sglang), [project site](https://sglang.io/) |
| `mlx-vlm` | [GitHub](https://github.com/Blaizzy/mlx-vlm), [mlx-community models](https://huggingface.co/mlx-community) |

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

## Single-command smoke tests

Before tunnel smoke tests, sync the code to the HPC login node:

```sh
export HPC_HOST=hpc.som.yale.edu
export HPC_REMOTE_DIR=ocr-examples
just sync-hpc
```

Run one engine/mode smoke test:

```sh
just smoke pypdf disk
just smoke docling tunnel
just smoke olmocr2 tunnel
just smoke deepseek_ocr tunnel
just smoke glm_ocr tunnel
just smoke unlimited_ocr tunnel
```

Run every supported smoke test serially:

```sh
just smoke-all
```

By default, that command requests `gpu:1` for the RTX-capable tunnel engines
and `gpu:a100:1` for `unlimited_ocr`.

Run tunnel smoke tests in parallel only when the GPU partition has room:

```sh
just smoke-all --parallel-tunnel 3
```

Bound slow backends explicitly:

```sh
just smoke-all --disk-timeout 600 --tunnel-timeout 1800
```

On SOM HPC, prefer RTX 8000 for the engines that support it:

```sh
just smoke docling tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001
just smoke olmocr2 tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001
just smoke deepseek_ocr tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001
just smoke glm_ocr tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001
```

Baidu Unlimited-OCR currently needs A100 on SOM HPC:

```sh
just smoke unlimited_ocr tunnel --hpc-gres gpu:a100:1
```

The full all-engine tunnel matrix can be forced onto A100 for validation when
that is explicitly needed. Do not use this as the default example; it increases
pressure on the A100 nodes:

```sh
just smoke all tunnel --hpc-gres gpu:a100:1 --workers 1 --in-flight 1 --parallel-tunnel 1
```

The full matrix runs disk mode for `pypdf`, `docling`, `olmocr2`,
`deepseek_ocr`, and `glm_ocr`; it runs tunnel mode for `docling`, `olmocr2`,
`deepseek_ocr`, `glm_ocr`, and `unlimited_ocr`. It skips `pypdf` tunnel and
`unlimited_ocr` disk because those backends do not exist in this repo.

## Disk-backed/local smoke tests

These read PDFs from disk and write outputs to disk. Use them only for public
sample PDFs or data that is approved for the storage location. The `olmocr2`,
`deepseek_ocr`, and `glm_ocr` local disk commands use MLX and require Apple
Silicon.

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
# Use the canonical login host and the remote repo path synced to the cluster.
# You must be on the Yale VPN to reach hpc.som.yale.edu.
export HPC_HOST=hpc.som.yale.edu
export HPC_USER=
export HPC_KEY=
export HPC_REMOTE_DIR=ocr-examples

just sync-hpc

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
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# DeepSeek-OCR-2
uv run --script scripts/deepseek_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# GLM-OCR
uv run --script scripts/glm_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# Baidu Unlimited-OCR
uv run --script scripts/unlimited_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 1 --hpc-gres gpu:a100:1 \
  --hpc-client hpc/client/unlimited_ocr_client.py \
  --include-text-native --force
```

Why the split: Docling, olmOCR-2, DeepSeek-OCR-2, and GLM-OCR pass tunnel
smoke tests on RTX 8000 when `c001` is excluded. Unlimited-OCR starts on RTX
but fails the first SGLang request with `cudaErrorNoKernelImageForDevice` in
the fused-MoE path, so the working example requests A100.

For private PDFs, keep the document layout and document-id list on the trusted
local side. The high-level wrappers create the lower-level `--pdf-list` files
locally before they start the tunnel clients. Do not copy the PDFs or OCR
outputs to shared HPC storage.
