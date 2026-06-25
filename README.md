# OCR examples for Yale SOM HPC

Small examples for running OCR on the HPC in ways that are effective, polite to
other users, and usable for both ordinary batch jobs and sensitive data workflows. We highly recommend your AI agent load the [HPC-related skills](https://github.com/yale-som-hpc/claude-code-marketplace) before launching jobs on the SOM HPC. We don't expect you to use these scripts directly, but rather to customize them for your use case. 

## What is here

| Path | Purpose |
| --- | --- |
| `examples/sample-documents.tsv` | Curated public OCRmyPDF test PDFs to download for validation |
| `examples/engine-backends.tsv` | The six OCR engines and their disk/tunnel backends |
| `docs/ocr-engines.md` | Copy-paste OCR engine smoke-test commands |
| `scripts/download_sample_documents.py` | Download sample PDFs and write `data/samples/manifest.txt` |
| `scripts/prepare_sample_documents.py` | Copy sample PDFs into the `data/documents/<guid>/document.pdf` layout |
| `scripts/ocr_engine_disk.py` | Disk-backed smoke runner for the OCR engine set |
| `examples/manifest.example.txt` | Example manifest format |
| `justfile` | Convenience recipes |

## Install prerequisites

These examples use `uv` to run Python scripts with inline dependencies. The HPC
GPU services run in Apptainer containers; users should not need to build a
repo-local Python environment or install OCR libraries by hand.

On the trusted local machine:

```sh
uv --version
ssh hpc true
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

## Public sample documents

The sample set is ten public PDFs from the OCRmyPDF test resources. They cover
simple typewriter text, two columns, rotated/skewed pages, slanted labels,
French diacritics, multipage input, and scanner/PDF edge cases. The source
catalog is `examples/sample-documents.tsv`.

Download them when you are ready to test on HPC:

```sh
just samples
just prepare-documents
```

That writes:

- `data/samples/pdfs/*.pdf`
- `data/samples/manifest.txt`
- `data/samples/documents.txt`
- `data/documents/<fake-guid>/document.pdf`

## OCR Engine Set

The current OCR engine set is:

- [`pypdf`](https://github.com/py-pdf/pypdf): local PDF text extraction
- [`docling`](https://github.com/docling-project/docling): local/HPC fallback OCR, plus [`docling-serve`](https://github.com/docling-project/docling-serve) over SSH tunnel
- [`olmocr2`](https://github.com/allenai/olmocr): local MLX or HPC vLLM using [`allenai/olmOCR-2-7B-1025`](https://huggingface.co/allenai/olmOCR-2-7B-1025)
- [`deepseek_ocr`](https://github.com/deepseek-ai/DeepSeek-OCR-2): DeepSeek-OCR-2, local MLX or HPC vLLM using [`deepseek-ai/DeepSeek-OCR-2`](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [`glm_ocr`](https://huggingface.co/zai-org/GLM-OCR): GLM-OCR, local MLX or HPC vLLM using [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR)
- [`unlimited_ocr`](https://github.com/baidu/Unlimited-OCR): Baidu Unlimited-OCR, HPC SGLang using [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)

Runtime links: [`vLLM`](https://github.com/vllm-project/vllm),
[`SGLang`](https://github.com/sgl-project/sglang), and
[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm). See
`docs/ocr-engines.md` for disk-backed and no-persistent-HPC-disk smoke
commands, plus more upstream documentation links.

## Running the examples

Prepare the public documents:

```sh
just samples
just prepare-documents
```

Run disk-backed examples for files that are approved for the current storage
location:

```sh
just engine-extract pypdf,docling --force
just engine-olmocr2 --include-text-native --force
just engine-deepseek --include-text-native --force
just engine-glm --include-text-native --force
```

Use `--use-hpc` on the engine scripts for the containerized GPU services.
Those scripts launch Slurm jobs politely, tunnel the service back over SSH, and
clean up matching service jobs with `just hpc-cleanup` if something is
interrupted.

## Polite HPC batch OCR

Create a manifest with one input path per line:

```sh
find /path/to/pdfs -type f \( -iname '*.pdf' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.tif' \) \
  | sort > manifest.txt
```

For PDFs that are approved to reside on the storage where the script runs, pass
that manifest with `--from-file` and set an output path appropriate for the
project:

```sh
uv run --script scripts/documents_process.py \
  --from-file manifest.txt \
  --documents-root /path/to/document-layout \
  --stages ocr \
  --engines pypdf,docling
```

For GPU engines, keep concurrency explicit:

```sh
uv run --script scripts/olmocr2_extract.py \
  --from-file manifest.txt \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001
```

Start with one worker and a small `--in-flight` value. Increase only after a
small run succeeds. Avoid running OCR on login nodes; the HPC paths in this
repo use Slurm services on compute nodes.

On SOM HPC, the RTX 8000 path has been tested for Docling, olmOCR-2,
DeepSeek-OCR-2, and GLM-OCR with `--hpc-exclude c001`. Baidu Unlimited-OCR
currently requires `--hpc-gres gpu:a100:1`; the current Baidu/SGLang wheel
fails on RTX 8000 during the MoE request path.

## No persistent HPC disk workflow

For PII or DUA-covered documents, do not copy input files to HPC project, home, or
scratch storage unless your agreement explicitly allows it.

Use the tunneled engine workflows when you need the compute node to see bytes
only over the tunnel and return OCR text in the HTTP response. The services bind
to `127.0.0.1` on the compute node and use `/dev/shm` for request/runtime
temporary files:

```sh
uv run --script scripts/olmocr2_extract.py \
  --from-file /secure/local/document-list.txt \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 \
  --include-text-native
```

See `docs/ocr-engines.md` for the matching commands for Docling,
DeepSeek-OCR-2, GLM-OCR, and Unlimited-OCR.

Do not use a normal `sbatch` output log for sensitive OCR text; Slurm stdout and
stderr logs are files on HPC disk. These tunneled commands write OCR output on
the trusted side.

## Manifest format

Manifest files are plain text with one document path per line:

```text
/path/to/documents/file_001.pdf
/path/to/documents/page_001.png
```

Blank lines and lines beginning with `#` are ignored.

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

