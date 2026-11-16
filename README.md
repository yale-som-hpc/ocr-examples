# OCR examples for Yale SOM HPC

Small examples for running OCR on the HPC in ways that are effective, polite to
other users, and usable for both ordinary batch jobs and sensitive data
workflows. We highly recommend your AI agent load the
[HPC-related skills](https://github.com/yale-som-hpc/claude-code-marketplace)
before launching jobs on the SOM HPC. We expect most users to customize these
scripts for their own document layout and output paths.

## Contents

- [Setup and public samples](#install-prerequisites)
- [Run the integrated engines](#running-the-examples)
- [HPC batch and tunneled workflows](#polite-hpc-batch-ocr)
- [Integrated engine reference](#integrated-engine-reference)
- [Specialist engine examples](#specialist-engine-examples)

## Orientation

The integrated engine scripts have two workflows:

| Workflow | Use when | Data location |
| --- | --- | --- |
| Disk-backed OCR | Public PDFs, sample PDFs, or files approved for the current filesystem | PDFs and OCR outputs live under `data/documents/<guid>/` or another approved `--documents-root` |
| Tunneled HPC OCR | Documents that should not be copied to persistent HPC storage | The trusted local machine reads PDFs; a Slurm compute-node service receives requests over SSH tunnel; OCR output is written locally |

Important vocabulary:

- `trusted local machine`: the machine where private or controlled PDFs may live
- `hpc.som.yale.edu`: the canonical SOM HPC SSH host; you must be on the Yale VPN to reach it
- `compute node`: the Slurm-allocated host where containerized OCR services run
- `document layout`: `<documents-root>/<guid>/document.pdf`
- `--from-file`: high-level wrapper input containing one UUID/GUID document ID per line
- `--pdf-list`: lower-level HTTP-client input containing direct PDF paths, optionally as `input_pdf<TAB>output_md`

If you are an AI assistant, read `README.md`, `justfile`, and the relevant script `--help` before changing commands. Start with public
samples, one worker, and one engine. Do not increase concurrency or request A100
GPUs unless the user explicitly asks or the engine requires it.

## What is Here

| Area | Purpose |
| --- | --- |
| `README.md` | Human and AI-agent orientation, safe first commands, and data-layout rules |
| `examples/` | Public sample catalogs, expected OCR checks, and lower-level `--pdf-list` example |
| `scripts/` | High-level wrappers, sample preparation, smoke tests, OCR output validation, and Slurm cleanup helpers |
| `hpc/client/` | Low-level SSH tunnel clients for Docling and vLLM services |
| `hpc/slurm/` | Containerized Slurm service scripts run on compute nodes |
| `third_party/ocrmypdf/` | Vendored OCRmyPDF text fixtures used by correctness smoke tests |
| `justfile` | Convenience recipes for setup, smoke tests, linting, sync, and cleanup |

## Install prerequisites

These examples use [`uv`](https://docs.astral.sh/uv/) to run Python scripts with
inline dependencies and [`just`](https://just.systems/) for convenience recipes.
The HPC GPU services run in Apptainer containers; users should not need to build
a repo-local Python environment or install OCR libraries by hand.

On the trusted local machine:

```sh
uv --version
just --version
ssh hpc.som.yale.edu true
```

On the HPC login node:

```sh
module load uv
module load apptainer
```

Check what the scripts can see:

```sh
just check
```

`just check` uses `${HPC_HOST:-hpc.som.yale.edu}`. If you have an SSH alias
named `hpc`, set `HPC_HOST=hpc` for these examples.

## Public sample documents

The integrated sample set is public PDFs from the OCRmyPDF test resources. They cover
text-native PDF extraction, simple typewriter text, two columns, rotated/skewed
pages, slanted labels, French diacritics, multipage input, and scanner/PDF edge
cases. The source catalog is `examples/sample-documents.tsv`. The specialist
examples use a separate Library of Congress handwritten letter catalog at
`examples/specialist-samples.tsv` and render its first page to PNG with Poppler.

Download them when you are ready to test on HPC:

```sh
just samples
just prepare-documents
just specialist-samples
just specialist-images
```

That writes:

- `data/samples/pdfs/*.pdf`
- `data/samples/images/lincoln-owens-1837.png` when `just specialist-images` runs
- `data/samples/manifest.txt`
- `data/samples/specialist-manifest.txt`
- `data/samples/documents.txt`
- `data/documents/<fake-guid>/document.pdf`

Smoke tests validate OCR output against `examples/expected-ocr.json`. Some
checks use vendored upstream text fixtures in `third_party/ocrmypdf/`; others
use conservative visible phrases from the public sample PDFs. These checks are
intended to catch blank output, backend failure text, wrong documents, and
obvious OCR failures. They are not a full OCR quality benchmark.

## OCR Engine Set

The integrated OCR engine set is:

- [`pypdf`](https://github.com/py-pdf/pypdf): local PDF text extraction
- [`docling`](https://github.com/docling-project/docling): local/HPC fallback OCR, plus [`docling-serve`](https://github.com/docling-project/docling-serve) over SSH tunnel
- [`olmocr2`](https://github.com/allenai/olmocr): HPC vLLM using [`allenai/olmOCR-2-7B-1025`](https://huggingface.co/allenai/olmOCR-2-7B-1025)
- [`deepseek_ocr`](https://github.com/deepseek-ai/DeepSeek-OCR-2): HPC vLLM using [`deepseek-ai/DeepSeek-OCR-2`](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [`glm_ocr`](https://huggingface.co/zai-org/GLM-OCR): HPC vLLM using [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR)

Runtime: [`vLLM`](https://github.com/vllm-project/vllm). See
the integrated engine reference below for detailed smoke commands and upstream links.

The [specialist examples](#specialist-engine-examples) add CHURRO, Kraken 7.1,
Chandra OCR 2, and Infinity Parser 2.
They use each project's own CLI or API and are not yet included in the
integrated `just smoke-all` matrix.

## Running the examples

Prepare the public documents:

```sh
just samples
just prepare-documents
```

Run disk-backed CPU examples inside an HPC Slurm CPU allocation for public PDFs
or files approved for HPC storage:

```sh
just engine-extract pypdf,docling --force
```

Docling is a fallback for pages with little embedded text. Text-native samples
such as `tagged.pdf` may produce only `pypdf.txt` in this recipe. Use the
Docling smoke command below to force a Docling result on that sample.

Run GPU recipes on the trusted client; their OCR models run in HPC Slurm
services. The recipes include `--use-hpc` and default to one RTX 8000 worker,
one in-flight PDF, and exclusion of c001:

```sh
just engine-olmocr2 --include-text-native --force
just engine-deepseek --include-text-native --force
just engine-glm --include-text-native --force
```

Those scripts tunnel the service back over SSH.
Run `just sync-hpc` after changing this repo locally so the login node has the
same `hpc/`, `scripts/`, and `examples/` files. The sync recipe uses
`${HPC_HOST:-hpc.som.yale.edu}` and `${HPC_REMOTE_DIR:-ocr-examples}` and
excludes local `data/`, `results/`, `tmp/`, `.parle/`, and cache directories.
If a run is interrupted, inspect and clean up matching service jobs:

```sh
just hpc-status --ocr-only
just hpc-cleanup --dry-run
just hpc-cleanup
```

## Polite HPC batch OCR

The higher-level engine wrappers use a deterministic document layout:

```text
<documents-root>/<document-id>/document.pdf
```

Their `--from-file` option expects one `document-id` per line, not arbitrary
PDF paths. The public sample setup creates that layout and writes
`data/samples/documents.txt` for you. For your own approved-on-disk PDFs,
create the same layout under a project-controlled `--documents-root`, then
write a document-id list:

```sh
# Example document-id list for an existing document layout.
find /path/to/document-layout -mindepth 2 -maxdepth 2 -name document.pdf \
  | sed 's#/document.pdf$##' \
  | xargs -n1 basename \
  | sort > document-ids.txt
```

Then pass that list with `--from-file` and set `--documents-root` to the layout:

```sh
uv run --script scripts/documents_process.py \
  --from-file document-ids.txt \
  --documents-root /path/to/document-layout \
  --stages ocr \
  --engines pypdf,docling
```

For GPU engines, keep concurrency explicit:

```sh
uv run --script scripts/olmocr2_extract.py \
  --from-file document-ids.txt \
  --documents-root /path/to/document-layout \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G
```

Start with one worker and a small `--in-flight` value. Increase only after a
small run succeeds. Avoid running OCR on login nodes; the HPC paths in this
repo use Slurm services on compute nodes.

In the November 2026 SOM HPC smoke runs, CPU disk checks passed for pypdf and
Docling; the RTX 8000 tunnel checks passed for Docling, olmOCR-2,
DeepSeek-OCR-2, and GLM-OCR with `--hpc-exclude c001`.
GLM-OCR uses a moving vLLM nightly image; rerun its smoke check when that image
changes.

## No persistent HPC disk workflow

For PII or DUA-covered documents, do not copy input files to HPC project, home, or
scratch storage unless your agreement explicitly allows it.

Use the tunneled engine workflows when you need the compute node to see bytes
only over the tunnel and return OCR text in the HTTP response. The trusted local
side binds the SSH tunnel to `127.0.0.1`; the compute-node service listens on a
Slurm-allocated node port, requires a per-job API key, and points request/runtime
temporary directories at `/dev/shm`:

```sh
uv run --script scripts/olmocr2_extract.py \
  --from-file /secure/local/document-list.txt \
  --documents-root /secure/local/document-layout \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G \
  --include-text-native
```

See the [tunneled smoke tests](#no-persistent-hpc-disk-smoke-tests) below for
matching commands for Docling, DeepSeek-OCR-2, and GLM-OCR.

Do not use a normal `sbatch` output log for sensitive OCR text; Slurm stdout and
stderr logs are files on HPC disk. These tunneled commands write OCR output on
the trusted side.

## Input List Formats

Most example wrappers use document-id lists:

```text
2c232430-2f36-5840-8a5a-9c02d279ca84
ae3903c0-da53-5893-81c9-43abb13cdf9d
```

Blank lines and lines beginning with `#` are ignored.

The lower-level HTTP clients under `hpc/client/` use `--pdf-list` instead.
That file accepts either one input PDF path per line or `input_pdf<TAB>output_md`
when you need user-determined output paths. The high-level wrappers generate
those `--pdf-list` files internally.

## Integrated engine reference

The canonical engine slugs match `scripts/ocr_provenance.py`.

| Engine | HPC disk path | No persistent HPC disk path |
| --- | --- | --- |
| `pypdf` | PDF text extraction on HPC CPU | Run where the trusted PDF already lives |
| `docling` | Docling CLI on HPC CPU | `docling-serve` through SSH tunnel |
| `olmocr2` | — | vLLM OpenAI-compatible server through SSH tunnel |
| `deepseek_ocr` | — | vLLM OpenAI-compatible server through SSH tunnel |
| `glm_ocr` | — | vLLM OpenAI-compatible server through SSH tunnel |

### Engine and runtime links

| Engine/runtime | Upstream links |
| --- | --- |
| `pypdf` | [documentation](https://pypdf.readthedocs.io/), [GitHub](https://github.com/py-pdf/pypdf) |
| `docling` | [documentation](https://docling-project.github.io/docling/), [GitHub](https://github.com/docling-project/docling), [Docling Serve](https://github.com/docling-project/docling-serve) |
| `olmocr2` | [olmOCR GitHub](https://github.com/allenai/olmocr), [Hugging Face model](https://huggingface.co/allenai/olmOCR-2-7B-1025), [paper](https://arxiv.org/abs/2510.19817) |
| `deepseek_ocr` | [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2), [Hugging Face model](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2), [paper](https://arxiv.org/abs/2601.20552) |
| `glm_ocr` | [Hugging Face model](https://huggingface.co/zai-org/GLM-OCR), [Z.ai Hugging Face org](https://huggingface.co/zai-org), [Z.ai GitHub org](https://github.com/zai-org) |
| `vllm` | [documentation](https://docs.vllm.ai/), [GitHub](https://github.com/vllm-project/vllm), [project site](https://vllm.ai/) |

### Single-command smoke tests

Before tunnel smoke tests, sync the code to the HPC login node:

```sh
export HPC_HOST=hpc.som.yale.edu
export HPC_REMOTE_DIR=ocr-examples
just sync-hpc
```

Run `just smoke pypdf disk` inside an HPC Slurm CPU allocation. Run tunnel
smoke tests on the trusted client; each OCR service runs on an HPC compute node:

```sh
just smoke pypdf disk
just smoke docling tunnel
just smoke olmocr2 tunnel
just smoke deepseek_ocr tunnel
just smoke glm_ocr tunnel
```

Run every HPC tunnel smoke test serially:

```sh
just smoke-all
```

By default, that command requests one RTX 8000 and excludes c001.

Run the HPC tunnel engine/GPU matrix:

```sh
just smoke-matrix
```

That enumerates tunnel `ENGINE-GPU` cells for `rtx`, writes
`results/smoke-matrix/report.tsv`, and skips cells that do not exist in this
repo such as `pypdf-tunnel-*`. A matrix cell passes only if the engine command exits successfully
and expected-content validation passes.

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
just smoke docling tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G
just smoke olmocr2 tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G
just smoke deepseek_ocr tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G
just smoke glm_ocr tunnel --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G
```

The tunnel matrix covers `docling`, `olmocr2`, `deepseek_ocr`, and `glm_ocr`.
CPU disk checks for `pypdf` and `docling` run separately in
an HPC Slurm allocation.

### Disk-backed HPC CPU smoke tests

Run these inside a Slurm CPU allocation. They read PDFs from HPC storage and
write outputs there; use only public PDFs or data approved for that storage.
For example, from the HPC login node:

```sh
cd ~/ocr-examples
srun --partition=cpunormal --cpus-per-task=4 --mem=24G \
  --time=00:20:00 --pty bash
```

Run the following commands in the allocated shell:

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

```

Notes:

- `documents_extract.py --no-docling` tests `pypdf` only.
- `documents_extract.py` without `--no-docling` tests local Docling fallback
  on scans.
- The GPU engines run through HPC Slurm services in the tunnel workflow.

### No persistent HPC disk smoke tests

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
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# DeepSeek-OCR-2
uv run --script scripts/deepseek_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

# GLM-OCR
uv run --script scripts/glm_ocr_extract.py \
  --from-file data/samples/documents.txt \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 --hpc-mem 32G \
  --hpc-client hpc/client/vllm_http_client.py \
  --include-text-native --force

```

Docling, olmOCR-2, DeepSeek-OCR-2, and GLM-OCR use HPC compute nodes with
`c001` excluded.

For private PDFs, keep the document layout and document-id list on the trusted
local side. The high-level wrappers create the lower-level `--pdf-list` files
locally before they start the tunnel clients. Do not copy the PDFs or OCR
outputs to shared HPC storage.

## Specialist engine examples

These are small examples for four additional projects. They are separate
from `just smoke-all`: that matrix covers the engines with integrated
`data/documents/<guid>/ocr/` wrappers and SSH tunnel clients. Commands below use
the projects' own CLIs and write to `results/specialist/`. The HPC
examples use **public samples only**; they keep no private document bytes on
the cluster.

### Samples and HPC setup

The HPC setup below creates `data/samples/images/lincoln-owens-1837.png`, cropped to the
manuscript on the first page of a
[Library of Congress 1837 handwritten letter](https://www.loc.gov/static/classroom-materials/abraham-lincoln/documents/letter_owens.pdf),
cataloged separately in `examples/specialist-samples.tsv`. The existing
`c02-22.pdf` is historical print, `linn.pdf` covers two columns, and
`graph.pdf` covers slanted chart labels. For
math, forms, and tables, use `tagged.pdf` for a basic table, or substitute a
public document from the [Chandra example gallery](https://github.com/datalab-to/chandra#examples).
The Lincoln PDF includes a text layer only for Library of Congress framing,
so the rendered image is the actual handwriting test.

To work on the cluster, set the host and remote directory, sync code, and
prepare only public samples on the HPC login node. This assumes `uv` and
Poppler's `pdftoppm` are available there. Run OCR itself inside a Slurm
allocation:

```sh
export HPC_HOST="${HPC_HOST:-hpc.som.yale.edu}"   # use hpc if that is your SSH alias
export HPC_REMOTE_DIR="${HPC_REMOTE_DIR:-ocr-examples}"
just sync-hpc
ssh "$HPC_HOST" "cd ~/$HPC_REMOTE_DIR && \
  command -v pdftoppm && \
  uv run scripts/download_sample_documents.py && \
  uv run scripts/download_sample_documents.py \
    --catalog examples/specialist-samples.tsv \
    --manifest data/samples/specialist-manifest.txt && \
  uv run scripts/prepare_specialist_samples.py"
ssh -t "$HPC_HOST" "cd ~/$HPC_REMOTE_DIR && \
  srun --partition=gpunormal --gres=gpu:rtx8000:1 \
    --cpus-per-task=4 --mem=32G --time=00:45:00 --pty bash"
```

The following GPU commands run **inside that allocated compute-node shell**.
Keep each project's Python dependencies isolated in scratch. Check `hostname`
and `nvidia-smi` before starting. These commands were run on SOM HPC in
November 2026; model downloads and Python environments belong on scratch.
The cluster's `uv` 0.6.6 is too old for CHURRO's installer, so install a newer
`uv` into scratch once and use it for the GPU examples:

```sh
export UV_TOOL_DIR=/gpfs/scratch60/$USER/ocr-example-tools
export UV_TOOL_BIN_DIR=$UV_TOOL_DIR/bin
export UV_CACHE_DIR=/gpfs/scratch60/$USER/.cache/uv
export XDG_CACHE_HOME=/gpfs/scratch60/$USER/.cache
export XDG_DATA_HOME=/gpfs/scratch60/$USER/.local/share
export HF_HOME=/gpfs/scratch60/$USER/.cache/huggingface
uv tool install --python 3.12 uv
export PATH=$UV_TOOL_BIN_DIR:$HOME/.local/bin:$PATH
```

### CHURRO: historical handwriting

[CHURRO](https://github.com/stanford-oval/Churro) targets historical printed
and handwritten sources. Its quick start specifies Python 3.12+, `uv`, and the
`stanford-oval/churro-3B` model. In the compute-node shell:

```sh
mkdir -p results/specialist/churro
uv tool install --python 3.12 churro-ocr
churro-ocr install hf
uv pip install --python "$UV_TOOL_DIR/churro-ocr/bin/python" accelerate
churro-ocr transcribe \
  --image data/samples/images/lincoln-owens-1837.png \
  --backend hf --model stanford-oval/churro-3B \
  --output results/specialist/churro/lincoln-owens-1837.xml
```

CHURRO writes `HistoricalDocument` XML. On the HPC test it recognized the
`Springfield` date and `Friend Mary` salutation, with reading-order mistakes
in the lower half. The CHURRO model weights use a Qwen research license; check
the model terms before use beyond research.

### Kraken 7.1: historical line recognition

[Kraken 7.1](https://github.com/mittagessen/kraken/releases/tag/7.1) adds
PP-OCRv6 based multilingual recognition models trained on historical material.
Its stock baseline segmenter locates lines; a downloaded recognition model
transcribes them. The medium model gives the best reported accuracy of the
three released sizes. This single-page command ran in an HPC `cpunormal`
allocation (`--cpus-per-task=4 --mem=16G`):

```sh
uv run --python 3.12 --with 'kraken>=7.1,<7.2' kraken get 10.5281/zenodo.21788410
mkdir -p results/specialist/kraken
uv run --python 3.12 --with 'kraken>=7.1,<7.2' kraken \
  -i data/samples/images/lincoln-owens-1837.png \
  results/specialist/kraken/lincoln-owens-1837.txt \
  segment -bl ocr -m medium.safetensors
```

`kraken get` prints the installed model filename. Use that filename with `-m`
if it differs. For layout and confidence review, run the same pipeline with
`-x` before `-i` to produce PAGE XML. The model is a general base model;
fine-tuning or a manuscript-specific model may be more accurate. The HPC run
recognized `Friend Mary` but misread the date and several lines.

### Chandra OCR 2: forms, math, tables, layout

[Chandra OCR 2](https://github.com/datalab-to/chandra) is strongest on
structured documents such as forms, tables, math and mixed layouts. Use the
existing two-column and table samples first. In a GPU compute-node shell with
a supported PyTorch/CUDA environment:

```sh
uv venv /gpfs/scratch60/$USER/chandra-example-venv --python 3.12
uv pip install --python /gpfs/scratch60/$USER/chandra-example-venv/bin/python 'chandra-ocr[hf]'
mkdir -p results/specialist/chandra
/gpfs/scratch60/$USER/chandra-example-venv/bin/chandra \
  data/samples/pdfs/linn.pdf results/specialist/chandra --method hf --no-images
/gpfs/scratch60/$USER/chandra-example-venv/bin/chandra \
  data/samples/pdfs/tagged.pdf results/specialist/chandra --method hf --no-images
```

The CLI writes Markdown, HTML, and metadata beneath a directory for each
input. Its default vLLM method needs a running server; `--method hf` uses a
local model. The open weights carry a [modified OpenRAIL-M license](https://github.com/datalab-to/chandra#commercial-usage)
with commercial limits.

Validate both HPC outputs:

```sh
uv run scripts/validate_ocr_output.py --engine chandra_ocr_2 \
  --filename linn.pdf --output results/specialist/chandra/linn/linn.md
uv run scripts/validate_ocr_output.py --engine chandra_ocr_2 \
  --filename tagged.pdf --output results/specialist/chandra/tagged/tagged.md
```

### Infinity Parser 2: complex document parsing

[Infinity-Parser2](https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser2)
has Pro and Flash models and a `parser` CLI. The Flash model fits on one HPC
RTX 8000 using the local Transformers backend. In the GPU compute-node shell:

```sh
uv venv /gpfs/scratch60/$USER/infinity-example-venv --python 3.12
uv pip install --python /gpfs/scratch60/$USER/infinity-example-venv/bin/python \
  infinity_parser2 torch torchvision accelerate
mkdir -p results/specialist/infinity-parser2
/gpfs/scratch60/$USER/infinity-example-venv/bin/parser \
  data/samples/pdfs/linn.pdf -o results/specialist/infinity-parser2 \
  --pages 1 --backend transformers \
  --model-name infly/Infinity-Parser2-Flash \
  --model-cache-dir /gpfs/scratch60/$USER/.cache/infinity_parser2 \
  --max-pixels 1000000 --batch-size 1
```

The two-column page tests reading order. The CLI otherwise defaults to an
offline vLLM engine with the larger Pro model, so specify both backend and
model as above for a single RTX 8000. The pixel cap is needed because the
default image size exhausted RTX 8000 memory in the HPC test. The successful
run wrote `results/specialist/infinity-parser2/linn.pdf/result.md` and passed
the repository's output validator.

```sh
uv run scripts/validate_ocr_output.py --engine infinity_parser_2 \
  --filename linn.pdf \
  --output results/specialist/infinity-parser2/linn.pdf/result.md
```

### HPC results on the public examples

| Engine | Node | Sample | Result |
| --- | --- | --- | --- |
| CHURRO 3B | RTX 8000 | 1837 Lincoln letter | XML output; opening date and salutation recognized, lower-page reading order imperfect |
| Kraken 7.1 medium | CPU | 1837 Lincoln letter | Text output; salutation recognized, date and several lines misread |
| Chandra OCR 2 | RTX 8000 | Two-column `linn.pdf` and table `tagged.pdf` | Readable Markdown with column order and table cells intact; both validations passed |
| Infinity Parser 2 Flash | RTX 8000 | Two-column `linn.pdf` | Readable Markdown with headings and column order intact; output validation passed |

These are single-document functional checks, not comparative accuracy scores.
The different examples exercise each engine's intended document type.

## Contributing

Please send a pull request! Bug fixes and new OCR engines are greatly appreciated, in particular.

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org/> 
