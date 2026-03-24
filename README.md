<p align="center">
  <h1 align="center">Pulsar AI</h1>
  <p align="center"><strong>The only platform that trains, deploys, evaluates, and improves LLMs in one closed loop</strong></p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-brightgreen.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/react-19-61dafb.svg" alt="React 19">
  <img src="https://img.shields.io/badge/tests-81%20suites-success.svg" alt="Tests: 81 suites">
  <img src="https://img.shields.io/badge/k8s-ready-326ce5.svg" alt="Kubernetes ready">
</p>

---

## Why Pulsar AI?

Most teams use 5-7 separate tools to fine-tune, deploy, monitor, and improve their LLMs. Pulsar AI replaces them all with a single self-hosted platform that closes the loop automatically:

```
Train (SFT/DPO) --> Deploy --> Agents work --> Traces collected --> Auto-generate training data --> Train again
```

**Your models get better every day, automatically.** No manual data labeling. No pipeline duct tape. No vendor lock-in.

## The Problem We Solve

| Pain Point | How teams solve it today | How Pulsar AI solves it |
|---|---|---|
| Fine-tuning requires ML expertise | Hire specialists, use notebooks | Visual experiment wizard, hardware auto-detection, one-click training |
| Agent traces are wasted | Log to Langfuse/Datadog, never use for training | Traces auto-convert to SFT/DPO datasets with one click |
| Too many tools in the stack | ClearML + LangSmith + W&B + custom scripts | Single platform: train, eval, deploy, monitor, improve |
| No governance for LLM ops | Manual approvals, no audit trail | Built-in RBAC, approval gates, full audit log |
| Deployment is fragile | Custom Docker + K8s manifests per model | One-click export to GGUF/vLLM/Ollama + production K8s manifests |
| Can't compare model versions | Spreadsheets, ad-hoc scripts | Built-in benchmarks with leaderboard, radar charts, cost/latency metrics |

## Competitive Landscape

### Pulsar AI vs. the alternatives

| Capability | Pulsar AI | Langfuse | ClearML | W&B | LangSmith | OpenJarvis |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SFT / DPO / GRPO training** | Yes | -- | Plugin | -- | -- | -- |
| **Hardware auto-detection** | Yes | -- | -- | -- | -- | -- |
| **Visual DAG pipelines (30+ nodes)** | Yes | -- | Yes | -- | -- | -- |
| **Agent framework (ReAct)** | Yes | -- | -- | -- | -- | Yes |
| **MCP / A2A protocols** | Yes | -- | -- | -- | -- | Partial |
| **LLM-as-Judge evaluation** | Yes | -- | -- | Yes | Yes | -- |
| **Agent eval suites + regression** | Yes | -- | -- | -- | Yes | -- |
| **Experiment tracking** | Yes | -- | Yes | Yes | -- | -- |
| **Trace-to-training pipeline** | Yes | -- | -- | -- | -- | -- |
| **GGUF / vLLM / Ollama export** | Yes | -- | -- | -- | -- | -- |
| **Real-time GPU monitoring** | Yes | -- | Yes | Yes | -- | -- |
| **Benchmark leaderboard** | Yes | -- | -- | Yes | -- | -- |
| **Prompt versioning + lab** | Yes | Yes | -- | -- | Yes | -- |
| **RBAC + governance + audit** | Yes | Yes | Yes | Yes | -- | -- |
| **SSO (OIDC) + MFA (TOTP)** | Yes | Yes | Yes | Enterprise | -- | -- |
| **Multi-tenant isolation** | Yes | Yes | Yes | Yes | -- | -- |
| **Kubernetes-ready** | Yes | Yes | Yes | Cloud | Cloud | -- |
| **Self-hosted, no cloud dependency** | Yes | Yes | Yes | -- | -- | Yes |
| **Open source** | Apache 2.0 | MIT | Apache 2.0 | -- | -- | MIT |
| **Single binary / single machine** | Yes | -- | -- | -- | -- | Yes |

### What makes Pulsar AI fundamentally different

**Langfuse** is excellent for tracing but has no training capabilities. You observe your LLMs but can't improve them from the same platform.

**ClearML** handles experiment tracking and pipelines but lacks agent frameworks, LLM-specific evaluation, and the closed-loop trace-to-training pipeline.

**Weights & Biases** offers powerful experiment tracking and evaluation but is cloud-dependent, closed-source, and doesn't include training or agent deployment.

**LangSmith** provides tracing and evaluation for LangChain apps but is tightly coupled to the LangChain ecosystem and offers no fine-tuning capabilities.

**OpenJarvis** focuses on agent orchestration but has no training, evaluation, or monitoring infrastructure.

**Pulsar AI is the only platform where traces from production agents automatically become training data for the next model iteration.** This closed-loop architecture means your models improve continuously without manual intervention.

## Platform Overview

### 18 Integrated Modules

```
+---------------------+    +------------------+    +-------------------+
|   DATA & TRAINING   |    |   ORCHESTRATION  |    |   AGENTS & EVAL   |
|                     |    |                  |    |                   |
|  Datasets           |    |  Workflow Builder |    |  Agent Chat       |
|  New Experiment     |    |  30+ Node Types  |    |  Agent Eval       |
|  Experiments        |    |  Agent Office    |    |  Traces           |
|  Prompt Lab         |    |  Lifecycle Graph |    |  OpenClaw Runtime |
+---------------------+    +------------------+    +-------------------+

+---------------------+    +------------------+    +-------------------+
|   DEPLOYMENT        |    |   MONITORING     |    |   GOVERNANCE      |
|                     |    |                  |    |                   |
|  Compute Manager    |    |  Real-time GPU   |    |  Workspaces       |
|  Benchmarks         |    |  Dashboard       |    |  Approval Gates   |
|  Model Export       |    |  Cost Tracking   |    |  Audit Log        |
|  vLLM / Ollama      |    |  Alert System    |    |  Admin Panel      |
+---------------------+    +------------------+    +-------------------+
```

### The Closed Loop in Action

```
1. TRAIN      pulsar train configs/sft.yaml        Fine-tune with SFT/DPO/GRPO
                  |
2. EVALUATE   Agent Eval + Benchmarks               LLM-as-Judge, pass/fail suites
                  |
3. EXPORT     GGUF, vLLM, Ollama, HuggingFace      One-click model packaging
                  |
4. DEPLOY     OpenClaw sandboxed runtime             Agents with MCP/A2A tools
                  |
5. OBSERVE    Traces + Monitoring                    Every agent action is logged
                  |
6. IMPROVE    Trace -> SFT/DPO dataset               One click: traces become training data
                  |
      +---------> Back to step 1. Models get better every cycle.
```

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/VasilyKolbenev/PulsarAI.git
cd PulsarAI
docker compose up
```

Open **http://localhost:8888** -- the full platform starts automatically.

### pip (development)

```bash
pip install -e ".[all]"

# Or pick modules:
pip install -e ".[unsloth]"      # 2-5x faster training on single GPU
pip install -e ".[vllm]"         # vLLM inference serving
pip install -e ".[ui]"           # Web dashboard
pip install -e ".[postgres]"     # PostgreSQL backend
pip install -e ".[redis]"        # Redis caching
pip install -e ".[s3]"           # S3 artifact storage
```

### First training run

```bash
# Fine-tune with SFT
pulsar train configs/examples/cam-sft.yaml

# Evaluate
pulsar eval --model ./outputs/cam-sft/lora --test-data data/test.csv

# Export to GGUF for local inference
pulsar export --model ./outputs/cam-sft/lora --format gguf --quant q4_k_m

# Launch the web dashboard
pulsar ui
```

## Hardware Auto-Detection

Set `strategy: auto` (default) and Pulsar picks the optimal training approach:

| Hardware | Strategy | Memory Usage |
|---|---|---|
| 1 GPU, < 12 GB | QLoRA (4-bit) | ~6 GB |
| 1 GPU, 12-24 GB | LoRA | ~14 GB |
| 1 GPU, 24-48 GB | Full fine-tune | ~30 GB |
| 2-4 GPUs, < 24 GB each | FSDP + QLoRA | Distributed |
| 8+ GPUs, 40+ GB each | FSDP Full | Distributed |

## Architecture

```
                           +-------------------+
                           |    React 19 UI    |
                           |  18 pages, TailwindCSS, Framer Motion
                           +--------+----------+
                                    |
                           +--------v----------+
                           |     FastAPI        |
                           |  REST + SSE + WS   |
                           |  JWT + API Keys    |
                           +--------+----------+
                                    |
     +----------+----------+--------+--------+----------+-----------+
     |          |          |        |        |          |           |
+----v---+ +---v----+ +---v---+ +--v----+ +-v------+ +-v-------+ +-v--------+
|Training| |Pipeline| |Agent  | |Eval   | |Export  | |OpenClaw | |Benchmark |
|Engine  | |DAG     | |Runtime| |Engine | |Registry| |Runtime  | |System    |
|SFT,DPO | |30 nodes| |ReAct  | |Judge  | |GGUF   | |Sandbox  | |Leaderboard|
|GRPO    | |YAML    | |MCP,A2A| |A/B    | |vLLM   | |Govern.  | |Compare   |
+--------+ +--------+ +-------+ +-------+ +--------+ +---------+ +----------+
     |          |          |        |        |          |           |
     +----------+----------+--------+--------+----------+-----------+
                                    |
                    +---------------+---------------+
                    |               |               |
             +------v-----+  +-----v------+  +-----v------+
             | SQLite/     |  | Redis      |  | S3/MinIO   |
             | PostgreSQL  |  | (optional) |  | (optional) |
             +-------------+  +------------+  +------------+
```

## Enterprise Features

### Security & Authentication
- **JWT tokens** with refresh rotation and blacklist
- **SSO/OIDC** -- Azure AD, Google, Okta, Keycloak, Auth0
- **MFA/TOTP** with QR provisioning and backup codes
- **Brute-force protection** -- account lockout after 5 failed attempts
- **Security headers** -- CSP, HSTS, X-Frame-Options, rate limiting
- **Password policy** -- 12+ chars, complexity requirements

### Multi-Tenancy & Governance
- **Workspace isolation** -- users see only their own data
- **Role-based access** -- admin, manager, member, viewer
- **Approval gates** -- review and approve model deployments
- **Full audit log** -- every action tracked with user, IP, timestamp
- **Admin panel** -- user management, system health, configuration

### Production Deployment
- **Kubernetes-ready** -- 11 manifests: Deployment, HPA, PDB, NetworkPolicy, Ingress
- **PostgreSQL** for production persistence (SQLite for dev)
- **Redis** for caching and session management
- **S3/MinIO** for artifact storage
- **Docker** with non-root user, health checks, graceful shutdown
- **Nginx** reverse proxy with TLS termination

## Project Structure

```
src/pulsar_ai/
  cli.py                CLI entry point (pulsar command)
  config.py             YAML config with inheritance
  hardware.py           GPU detection and strategy selection
  training/             SFT, DPO, GRPO, distributed (FSDP), embedding, reranker
  evaluation/           Batch inference, metrics, LLM-as-Judge, agent eval
  export/               LoRA merge, GGUF quantization, HuggingFace Hub
  serving/              llama.cpp, vLLM, Ollama backends
  agent/                ReAct runtime, tools, memory, MCP/A2A protocols
  openclaw/             Sandboxed agent execution, NemoClaw deployment
  pipeline/             DAG engine, 30+ node types, closed-loop steps
  benchmark/            Model benchmarks, leaderboard, comparison
  prompts/              Prompt templates, versioning, testing
  observability/        Tracing, logging, system metrics
  storage/              SQLite/PostgreSQL, schema migrations, stores
  ui/                   FastAPI routes, auth, middleware, admin

ui/
  src/pages/            18 React pages (Dashboard through Admin)
  src/components/       Shared components, layout, Agent Office (3D)
  src/api/              Typed API client
  src/hooks/            Custom React hooks (SSE, metrics, auth)

k8s/                    11 Kubernetes manifests
docs/                   User guide + deployment guide (Russian)
configs/                Training configs with YAML inheritance
tests/                  81 test suites
```

## CLI Reference

```
pulsar train <config> [overrides...]       Fine-tune with SFT, DPO, or GRPO
pulsar eval --model <path> --test-data <p> Evaluate model quality
pulsar export --model <path> --format <f>  Export: gguf, merged, hub
pulsar serve --model <path> --backend <b>  Start inference server (vLLM, llama.cpp)
pulsar agent serve                         Launch agent runtime
pulsar ui                                  Open web dashboard (port 8888)
```

## Documentation

| Document | Description |
|---|---|
| [User Guide](docs/user-guide.md) | Complete platform walkthrough (18 sections) |
| [Deployment Guide](docs/deployment-guide.md) | Docker, Kubernetes, SSL, OIDC, backups |
| [Contributing](CONTRIBUTING.md) | Development setup and guidelines |

## Contributing

```bash
git clone https://github.com/VasilyKolbenev/PulsarAI.git
cd PulsarAI
pip install -e ".[dev]"
pytest tests/
```

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

Code style: type hints on all functions, Google-style docstrings, PEP 8 naming, 100-char line limit.

## License

[Apache License 2.0](LICENSE)

Copyright 2025-2026 Pulsar AI Contributors
