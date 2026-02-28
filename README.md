# llm-forge

Universal LLM fine-tuning pipeline: **SFT → DPO → Eval → Export → Serving**.

One YAML config, one CLI command. Auto-detects hardware and selects optimal training strategy.

## Features

- **SFT & DPO training** with Unsloth (2-5x speedup) or HuggingFace Transformers
- **Hardware auto-detection** — automatically selects QLoRA / LoRA / FSDP / DeepSpeed
- **Config inheritance** — compose experiments from reusable base/model/strategy configs
- **Evaluation** — batch inference, accuracy, F1, confusion matrix, Markdown reports
- **Export** — merge LoRA, GGUF quantization (q4_k_m/q8_0/f16), Ollama Modelfile
- **Serving** — llama.cpp, llama-cpp-python, Ollama, vLLM backends
- **HuggingFace Hub** — push models/adapters directly

## Installation

```bash
pip install -e .

# With optional backends
pip install -e ".[unsloth]"     # Unsloth (recommended for single GPU)
pip install -e ".[vllm]"        # vLLM serving
pip install -e ".[llamacpp]"    # llama.cpp serving
pip install -e ".[eval]"        # Evaluation plots (seaborn, matplotlib)
pip install -e ".[all]"         # Everything
```

## Quick Start

### 1. Train (SFT)

```bash
forge train configs/examples/cam-sft.yaml
```

With overrides:
```bash
forge train configs/examples/cam-sft.yaml learning_rate=1e-4 epochs=5
```

### 2. Evaluate

```bash
forge eval --model ./outputs/cam-sft/lora --test-data data/test.csv
```

### 3. Export to GGUF

```bash
forge export --model ./outputs/cam-sft/lora --format gguf --quant q4_k_m
```

### 4. Serve

```bash
forge serve --model ./outputs/model-q4_k_m.gguf --port 8080
```

## Config System

Configs use YAML inheritance. An experiment config can inherit from base, model, strategy, and task configs:

```yaml
# configs/examples/cam-sft.yaml
inherit:
  - base
  - models/qwen2.5-3b

task: sft

dataset:
  path: data/cam_intents.csv
  text_column: phrase
  label_columns: [domain, skill]

training:
  epochs: 3
  learning_rate: 2e-4
```

### Hardware Auto-Detection

Set `strategy: auto` (default) and llm-forge detects your GPU and selects:

| Hardware | Strategy |
|----------|----------|
| 1 GPU, <12 GB | QLoRA |
| 1 GPU, 12-24 GB | LoRA |
| 1 GPU, 24-48 GB | Full finetune |
| 2-4 GPUs, <24 GB | FSDP + QLoRA |
| 8+ GPUs, 40+ GB | FSDP Full |

### Supported Models

Pre-configured in `configs/models/`:
- `qwen2.5-3b` — Qwen 2.5 3B Instruct
- `llama3.2-1b` — Llama 3.2 1B Instruct
- `mistral-7b` — Mistral 7B Instruct v0.3

Any HuggingFace model works — just set `model.name` in your config.

## CLI Reference

```
forge train <config> [overrides...]   Train SFT or DPO
forge eval --model <path> --test-data <path>   Evaluate model
forge export --model <path> --format <gguf|merged|hub>   Export model
forge serve --model <path> --backend <llamacpp|vllm>   Start server
```

## Project Structure

```
src/llm_forge/
  cli.py              CLI entrypoint
  config.py           YAML config loader with inheritance
  hardware.py         GPU detection & strategy selection
  model_loader.py     Unified model loading
  validation.py       Config validation
  data/               Dataset loading, formatting, splitting
  training/           SFT, DPO, distributed training
  evaluation/         Batch inference, metrics, reports
  export/             LoRA merge, GGUF, HuggingFace Hub
  serving/            llama.cpp, vLLM servers
configs/
  base.yaml           Shared defaults
  models/             Model-specific configs
  strategies/         Training strategy configs
  tasks/              Task-specific configs (sft, dpo, eval)
  examples/           Ready-to-use experiment configs
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
