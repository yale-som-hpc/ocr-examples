# OCR examples for Yale SOM HPC

Small examples for running OCR on the HPC in ways that are effective, polite to
other users, and usable for both ordinary batch jobs and sensitive data
workflows. We highly recommend your AI agent load the
[HPC-related skills](https://github.com/yale-som-hpc/claude-code-marketplace)
before launching jobs on the SOM HPC. We expect most users to customize these
scripts for their own document layout and output paths.

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
location. The `olmocr2`, `deepseek_ocr`, and `glm_ocr` local commands use
MLX and require Apple Silicon; use their `--use-hpc` tunnel mode from Linux
or other non-MLX machines.

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
  --documents-root /secure/local/document-layout \
  --use-hpc --workers 1 --in-flight 2 \
  --hpc-gres gpu:rtx8000:1 --hpc-exclude c001 \
  --include-text-native
```

See `docs/ocr-engines.md` for the matching commands for Docling,
DeepSeek-OCR-2, GLM-OCR, and Unlimited-OCR.

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
