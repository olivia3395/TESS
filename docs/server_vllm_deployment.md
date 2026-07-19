# Server vLLM Deployment Notes

The local pipeline works with `--backend mock` and does not require a model server.
For real primitive inference, run an OpenAI-compatible chat-completion server and switch only the sampled inference stage to the real backend.

## What Runs On The Server

Run a vLLM OpenAI-compatible API server that exposes a chat completion endpoint such as:

```text
http://<server-host>:8000/v1/chat/completions
```

The repo code calls the OpenAI-compatible Python client from `OpenAICompatibleClient`.
GT cache, flat gate cache, grouped gate cache, and audit generation remain local Python/cache steps and do not need model-server changes.

## Example vLLM Server Command

Adjust model path/name, host, port, tensor parallelism, dtype, and GPU settings for your server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_OR_LOCAL_PATH> \
  --served-model-name <MODEL_NAME_FOR_CLIENT> \
  --host 0.0.0.0 \
  --port 8000
```

Use the model name from `--served-model-name` as the repo `--model` argument.
For local vLLM deployments, `--api-key EMPTY` is usually sufficient unless your server enforces authentication.

## Repo CLI Changes For Real Inference

Start with a tiny sampled-inference smoke test:

```bash
python -m primitive_inference_rc2.sampled_inference \
  --root /home/lyc/workspace/TESS-RC2 \
  --dataset fnspid \
  --split train \
  --primitive distribution_shift \
  --backend openai-compatible \
  --base-url http://<server-host>:8000/v1 \
  --api-key EMPTY \
  --model <MODEL_NAME_FOR_CLIENT> \
  --prompt-source auto \
  --num-samples 2 \
  --limit 2 \
  --seed 42
```

After the smoke output parses correctly, run a slightly larger limit before any full split:

```bash
python -m primitive_inference_rc2.run_primitive_pipeline \
  --root /home/lyc/workspace/TESS-RC2 \
  --dataset fnspid \
  --primitives distribution_shift \
  --splits train \
  --backend openai-compatible \
  --prompt-source auto \
  --num-samples 4 \
  --limit 10 \
  --seed 42 \
  --stages sampled gate grouped audit \
  --overwrite
```

Then remove `--limit` only when server throughput, prompt formatting, parsing, and costs are understood.

## What Should Not Need Code Changes

- GT primitive cache building
- flat gate cache merging
- grouped gate cache generation
- audit report generation
- prompt-template discovery for `legacy/primitive_inference/prompt_templates`

## Things That May Need Adjustment

- `--model` must match the vLLM served model name.
- `--base-url` must point to `/v1`, not `/v1/chat/completions`.
- Some models need a specific chat template configured in vLLM.
- If completions include extra text, update primitive parsing carefully in `primitive_specs.py`.
- If the server requires auth, replace `--api-key EMPTY` with the real key or environment-injected value.

