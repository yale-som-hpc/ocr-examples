Source: vllm-project/vllm PR #42765 (sudhir-mcw fork)
Commit: 658f301489da42390e840836849232571d734d5c
Fixes: GLM-OCR producing degenerate output (issue #42016)
Bind-mount this file over vllm/model_executor/layers/rotary_embedding/mrope.py
in the docker://vllm/vllm-openai:nightly container.
