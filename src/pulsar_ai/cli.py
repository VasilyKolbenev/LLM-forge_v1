"""CLI entrypoint for Pulsar AI: pulsar train/eval/export/serve."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger("pulsar_ai")


def setup_logging(verbose: bool = False) -> None:
    """Configure rich logging handler.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _parse_overrides(overrides: tuple[str, ...]) -> dict[str, str]:
    """Parse key=value CLI overrides into a dict.

    Args:
        overrides: Tuple of "key=value" strings from CLI.

    Returns:
        Dict of parsed overrides.
    """
    result = {}
    for item in overrides:
        if "=" not in item:
            console.print(f"[red]Invalid override (expected key=value): {item}[/red]")
            sys.exit(1)
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.version_option(package_name="pulsar-ai")
def main(verbose: bool) -> None:
    """Pulsar AI: Universal LLM fine-tuning pipeline.

    Supports SFT, DPO, evaluation, GGUF export, and serving.
    Auto-detects hardware and selects optimal training strategy.
    """
    setup_logging(verbose)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("overrides", nargs=-1)
@click.option(
    "--task",
    type=click.Choice(["sft", "dpo", "grpo", "embedding", "reranker", "classification", "auto"]),
    default="auto",
    help="Training task (default: auto-detect from config).",
)
@click.option(
    "--base-model",
    type=click.Path(),
    default=None,
    help="Path to SFT adapter for DPO training.",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume training from checkpoint directory.",
)
def train(
    config_path: str,
    overrides: tuple[str, ...],
    task: str,
    base_model: Optional[str],
    resume: Optional[str],
) -> None:
    """Run training (SFT or DPO).

    \b
    Examples:
        pulsar train experiments/cam-sft.yaml
        pulsar train experiments/cam-dpo.yaml --task dpo --base-model ./outputs/cam-sft
        pulsar train experiments/cam-sft.yaml learning_rate=1e-4 epochs=5
    """
    from pulsar_ai.config import load_config

    cli_overrides = _parse_overrides(overrides)
    if base_model:
        cli_overrides["sft_adapter_path"] = base_model

    config = load_config(config_path, cli_overrides=cli_overrides)

    if resume:
        config["resume_from_checkpoint"] = resume
        logger.info("Resuming from checkpoint: %s", resume)

    # Determine task
    if task == "auto":
        task = config.get("task", "sft")

    # Validate config
    from pulsar_ai.validation import validate_config

    errors = validate_config(config, task=task)
    if errors:
        for err in errors:
            console.print(f"[red]Config error:[/red] {err}")
        sys.exit(1)

    _show_config_summary(config, task)

    ui_store = None
    ui_experiment_id = None
    progress = None

    # Mirror CLI runs to UI ExperimentStore so active training is visible in Web UI.
    try:
        from pulsar_ai.ui.experiment_store import ExperimentStore

        ui_store = ExperimentStore()
        run_name = config.get("name") or Path(config_path).stem
        run_name = f"[CLI] {run_name}"
        ui_experiment_id = ui_store.create(name=run_name, config=config, task=task)
        ui_store.update_status(ui_experiment_id, "running")
        ui_store.add_metrics(
            ui_experiment_id,
            {
                "step": 0,
                "epoch": 0.0,
                "event": "started",
                "time": datetime.now().isoformat(),
            },
        )

        from pulsar_ai.ui.progress import ProgressCallback

        progress = ProgressCallback(
            job_id=f"cli-{ui_experiment_id}",
            experiment_id=ui_experiment_id,
        )
        logger.info("CLI run synced to UI ExperimentStore: %s", ui_experiment_id)
    except Exception:
        logger.debug("UI ExperimentStore sync unavailable for CLI run", exc_info=True)

    try:
        if task == "sft":
            from pulsar_ai.training.sft import train_sft

            results = train_sft(config, progress=progress)
        elif task == "dpo":
            from pulsar_ai.training.dpo import train_dpo

            results = train_dpo(config, progress=progress)
        elif task == "grpo":
            try:
                from pulsar_ai.training.grpo import train_grpo
            except ImportError:
                console.print(
                    "[red]GRPO requires TRL >= 0.14.[/red]" " Install: pip install 'trl>=0.14,<1.0'"
                )
                sys.exit(1)
            results = train_grpo(config, progress=progress)
        elif task == "embedding":
            try:
                from pulsar_ai.training.embedding import train_embedding
            except ImportError:
                console.print(
                    "[red]Embedding requires sentence-transformers >= 3.0.[/red]"
                    " Install: pip install 'pulsar-ai[embedding]'"
                )
                sys.exit(1)
            results = train_embedding(config, progress=progress)
        elif task == "reranker":
            try:
                from pulsar_ai.training.reranker import train_reranker
            except ImportError:
                console.print(
                    "[red]Reranker requires sentence-transformers >= 3.0.[/red]"
                    " Install: pip install 'pulsar-ai[embedding]'"
                )
                sys.exit(1)
            results = train_reranker(config, progress=progress)
        elif task == "classification":
            try:
                from pulsar_ai.training.classification import train_classification
            except ImportError:
                console.print(
                    "[red]Classification requires scikit-learn.[/red]"
                    " Install: pip install 'pulsar-ai[classification]'"
                )
                sys.exit(1)
            results = train_classification(config, progress=progress)
        else:
            console.print(f"[red]Unknown task: {task}[/red]")
            sys.exit(1)

        if ui_store and ui_experiment_id:
            ui_store.update_status(ui_experiment_id, "completed")
            artifacts = {k: v for k, v in results.items() if isinstance(v, str)}
            if artifacts:
                ui_store.set_artifacts(ui_experiment_id, artifacts)
            if "training_loss" in results:
                ui_store.add_metrics(
                    ui_experiment_id,
                    {
                        "step": results.get("global_steps", 0),
                        "loss": results["training_loss"],
                        "event": "metric",
                    },
                )
            if "eval_results" in results:
                ui_store.set_eval_results(ui_experiment_id, results["eval_results"])

        _show_training_results(results)
    except Exception:
        if ui_store and ui_experiment_id:
            ui_store.update_status(ui_experiment_id, "failed")
        raise
    finally:
        if progress is not None:
            try:
                from pulsar_ai.ui.progress import cleanup_queue

                cleanup_queue(progress.job_id)
            except Exception:
                logger.debug("Failed to cleanup CLI progress queue", exc_info=True)


@main.command(name="eval")
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model or adapter directory.",
)
@click.option(
    "--test-data",
    type=click.Path(exists=True),
    required=True,
    help="Path to test dataset.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Config file for eval settings.",
)
@click.option("--batch-size", type=int, default=8, help="Inference batch size.")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output directory for eval report.",
)
def evaluate(
    model: str,
    test_data: str,
    config_path: Optional[str],
    batch_size: int,
    output: Optional[str],
) -> None:
    """Evaluate a trained model on test data.

    \b
    Examples:
        pulsar eval --model ./outputs/cam-sft/lora --test-data data/test.csv
        pulsar eval --model ./outputs/cam-sft/lora --test-data data/test.csv --output reports/
    """
    from pulsar_ai.config import load_config

    if config_path:
        config = load_config(config_path, auto_hardware=False)
    else:
        config = {}

    config["model_path"] = model
    config["test_data_path"] = test_data
    config.setdefault("evaluation", {})["batch_size"] = batch_size
    if output:
        config.setdefault("output", {})["eval_dir"] = output

    from pulsar_ai.evaluation.runner import run_evaluation

    results = run_evaluation(config)
    _show_eval_results(results)


@main.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model or adapter directory.",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["gguf", "merged", "hub", "awq", "gptq"]),
    default="gguf",
    help="Export format.",
)
@click.option(
    "--quant",
    type=click.Choice(["q4_k_m", "q8_0", "f16"]),
    default="q4_k_m",
    help="Quantization level for GGUF.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for exported model.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Config file for export settings.",
)
def export(
    model: str,
    export_format: str,
    quant: str,
    output: Optional[str],
    config_path: Optional[str],
) -> None:
    """Export model to production format.

    \b
    Examples:
        pulsar export --model ./outputs/cam-sft/lora --format gguf --quant q4_k_m
        pulsar export --model ./outputs/cam-sft/lora --format merged --output ./merged/
        pulsar export --model ./outputs/cam-sft/lora --format hub
    """
    from pulsar_ai.config import load_config

    if config_path:
        config = load_config(config_path, auto_hardware=False)
    else:
        config = {}

    config["model_path"] = model
    config.setdefault("export", {}).update(
        {
            "format": export_format,
            "quantization": quant,
        }
    )
    if output:
        config["export"]["output_path"] = output

    if export_format == "gguf":
        from pulsar_ai.export.gguf import export_gguf

        result = export_gguf(config)
    elif export_format == "merged":
        from pulsar_ai.export.merged import export_merged

        result = export_merged(config)
    elif export_format == "hub":
        from pulsar_ai.export.hub import push_to_hub

        result = push_to_hub(config)
    elif export_format == "awq":
        try:
            from pulsar_ai.export.awq import export_awq
        except ImportError:
            console.print("[red]AWQ requires autoawq.[/red] Install: pip install 'pulsar-ai[awq]'")
            sys.exit(1)
        result = export_awq(config)
    elif export_format == "gptq":
        try:
            from pulsar_ai.export.gptq import export_gptq
        except ImportError:
            console.print(
                "[red]GPTQ requires auto-gptq.[/red] Install: pip install 'pulsar-ai[gptq]'"
            )
            sys.exit(1)
        result = export_gptq(config)
    else:
        console.print(f"[red]Unknown format: {export_format}[/red]")
        sys.exit(1)

    console.print(Panel(f"[green]Export complete:[/green] {result.get('output_path', 'done')}"))


@main.command()
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model file (GGUF) or directory.",
)
@click.option("--port", type=int, default=8080, help="Server port.")
@click.option(
    "--backend",
    type=click.Choice(["llamacpp", "vllm", "sglang"]),
    default="llamacpp",
    help="Serving backend.",
)
@click.option("--host", default="0.0.0.0", help="Server host.")
def serve(model: str, port: int, backend: str, host: str) -> None:
    """Start model serving.

    \b
    Examples:
        pulsar serve --model ./outputs/model.gguf --port 8080
        pulsar serve --model ./outputs/cam-sft --backend vllm --port 8000
    """
    console.print(Panel(f"Starting [bold]{backend}[/bold] server on {host}:{port}"))

    if backend == "llamacpp":
        from pulsar_ai.serving.llamacpp import start_server

        start_server(model_path=model, host=host, port=port)
    elif backend == "vllm":
        from pulsar_ai.serving.vllm import start_server

        start_server(model_path=model, host=host, port=port)
    elif backend == "sglang":
        try:
            from pulsar_ai.serving.sglang import start_server as start_sglang
        except ImportError:
            console.print(
                "[red]SGLang not installed.[/red] Install: pip install 'pulsar-ai[sglang]'"
            )
            sys.exit(1)
        start_sglang(model_path=model, host=host, port=port)


@main.command()
@click.argument("name")
@click.option(
    "--task",
    type=click.Choice(["sft", "dpo", "grpo", "embedding", "reranker", "classification"]),
    default="sft",
    help="Training task.",
)
@click.option(
    "--model",
    type=click.Choice(["qwen2.5-3b", "llama3.2-1b", "mistral-7b"]),
    default="qwen2.5-3b",
    help="Base model.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: ./outputs/<name>).",
)
def init(name: str, task: str, model: str, output_dir: Optional[str]) -> None:
    """Create a new experiment config.

    \b
    Examples:
        pulsar init my-classifier
        pulsar init my-classifier --task dpo --model llama3.2-1b
    """
    import yaml

    if output_dir is None:
        output_dir = f"./outputs/{name}"

    config: dict = {
        "inherit": ["base", f"models/{model}"],
        "task": task,
        "dataset": {
            "path": f"data/{name}.csv",
            "format": "csv",
            "text_column": "text",
            "label_columns": ["label"],
            "test_size": 0.15,
        },
        "training": {
            "epochs": 3,
            "learning_rate": 2e-4,
        },
        "output": {
            "dir": output_dir,
        },
    }

    if task == "dpo":
        config["inherit"].append("tasks/dpo")
        config["sft_adapter_path"] = f"./outputs/{name}-sft/lora"
        config["dpo"] = {
            "pairs_path": f"./outputs/{name}-sft/dpo_pairs.jsonl",
            "beta": 0.1,
        }
        config["training"]["epochs"] = 2
        config["training"]["learning_rate"] = 5e-5

    config_dir = Path("configs/experiments")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(
        Panel(
            f"[green]Created:[/green] {config_path}\n"
            f"[dim]Edit dataset.path and label_columns, then run:[/dim]\n"
            f"  pulsar train {config_path}",
            title=f"New Experiment: {name}",
        )
    )


@main.command()
def info() -> None:
    """Show detected hardware and recommended strategy.

    \b
    Examples:
        pulsar info
    """
    from pulsar_ai.hardware import detect_hardware

    hw = detect_hardware()

    table = Table(title="Hardware Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("GPUs", str(hw.num_gpus))
    table.add_row("GPU Name", hw.gpu_name)
    table.add_row("VRAM per GPU", f"{hw.vram_per_gpu_gb:.1f} GB")
    table.add_row("Total VRAM", f"{hw.total_vram_gb:.1f} GB")
    table.add_row("Compute Capability", f"{hw.compute_capability[0]}.{hw.compute_capability[1]}")
    table.add_row("BF16 Supported", str(hw.bf16_supported))
    table.add_row("Recommended Strategy", f"[bold]{hw.strategy}[/bold]")
    table.add_row("Recommended Batch Size", str(hw.recommended_batch_size))
    table.add_row("Recommended Grad Accum", str(hw.recommended_gradient_accumulation))

    console.print(table)


# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚
# Agent subgroup
# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚


@main.group()
def agent() -> None:
    """Agent system commands: init, test, serve."""


@agent.command(name="init")
@click.argument("name")
@click.option(
    "--model-url",
    default="http://localhost:8080/v1",
    help="Model server URL.",
)
@click.option(
    "--model-name",
    default="default",
    help="Model name on the server.",
)
def agent_init(name: str, model_url: str, model_name: str) -> None:
    """Create a new agent config.

    \b
    Examples:
        pulsar agent init my-assistant
        pulsar agent init code-helper --model-url http://localhost:11434/v1
    """
    import yaml

    config = {
        "inherit": ["agents/base"],
        "agent": {
            "name": name,
            "system_prompt": (
                f"You are {name}, a helpful AI assistant. " "Use the available tools when needed."
            ),
        },
        "model": {
            "base_url": model_url,
            "name": model_name,
            "timeout": 120,
        },
        "tools": [
            {"name": "search_files", "module": "pulsar_ai.agent.builtin_tools"},
            {"name": "read_file", "module": "pulsar_ai.agent.builtin_tools"},
            {"name": "calculate", "module": "pulsar_ai.agent.builtin_tools"},
        ],
        "memory": {
            "max_tokens": 4096,
            "strategy": "sliding_window",
        },
        "guardrails": {
            "max_iterations": 15,
            "max_tokens": 8192,
        },
    }

    config_dir = Path("configs/agents")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(
        Panel(
            f"[green]Created:[/green] {config_path}\n"
            f"[dim]Edit the config, then run:[/dim]\n"
            f"  pulsar agent test {config_path}",
            title=f"New Agent: {name}",
        )
    )


@agent.command(name="test")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--native-tools", is_flag=True, help="Use native tool calling instead of ReAct.")
def agent_test(config_path: str, native_tools: bool) -> None:
    """Interactive REPL to test an agent.

    \b
    Examples:
        pulsar agent test configs/agents/my-assistant.yaml
        pulsar agent test configs/agents/my-assistant.yaml --native-tools
    """
    from pulsar_ai.agent.loader import load_agent_config
    from pulsar_ai.agent.base import BaseAgent
    from pulsar_ai.agent.builtin_tools import get_default_registry
    from pulsar_ai.validation import validate_agent_config

    config = load_agent_config(config_path)

    errors = validate_agent_config(config)
    if errors:
        for err in errors:
            console.print(f"[red]Config error:[/red] {err}")
        sys.exit(1)

    if native_tools:
        config.setdefault("agent", {})["native_tools"] = True

    tools = get_default_registry()
    agent_instance = BaseAgent.from_config(config, tools=tools)

    agent_name = config.get("agent", {}).get("name", "agent")
    console.print(
        Panel(
            f"Agent [bold]{agent_name}[/bold] loaded with "
            f"{len(tools)} tools.\n"
            f"Type your message and press Enter. Type 'quit' to exit.",
            title="Agent REPL",
        )
    )

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        try:
            answer = agent_instance.run(user_input)
            console.print(f"[bold green]{agent_name}:[/bold green] {answer}\n")
        except ConnectionError as e:
            console.print(f"[red]Connection error:[/red] {e}")
            console.print("[dim]Is your model server running?[/dim]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            logger.exception("Agent error")


@agent.command(name="serve")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", type=int, default=8081, help="Server port.")
def agent_serve(config_path: str, host: str, port: int) -> None:
    """Start agent REST API server.

    \b
    Examples:
        pulsar agent serve configs/agents/my-assistant.yaml
        pulsar agent serve configs/agents/my-assistant.yaml --port 9000
    """
    from pulsar_ai.agent.loader import load_agent_config
    from pulsar_ai.agent.server import start_agent_server
    from pulsar_ai.validation import validate_agent_config

    config = load_agent_config(config_path)

    errors = validate_agent_config(config)
    if errors:
        for err in errors:
            console.print(f"[red]Config error:[/red] {err}")
        sys.exit(1)

    agent_name = config.get("agent", {}).get("name", "agent")
    console.print(
        Panel(
            f"Starting agent [bold]{agent_name}[/bold] server\n"
            f"Endpoint: http://{host}:{port}/v1/agent/chat\n"
            f"Health:   http://{host}:{port}/v1/agent/health",
            title="Agent Server",
        )
    )

    start_agent_server(config, host=host, port=port)


# ── Agent Eval subgroup ──────────────────────────────────────────


@main.group(name="agent-eval")
def agent_eval_group() -> None:
    """Agent evaluation commands."""


@agent_eval_group.command(name="run")
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--agent-config", type=click.Path(exists=True), required=True, help="Agent config YAML.")
@click.option(
    "--scoring",
    type=click.Choice(["judge", "exact", "contains"]),
    default="exact",
    help="Scoring strategy.",
)
def eval_run(suite_path: str, agent_config: str, scoring: str) -> None:
    """Run an agent evaluation suite.

    \b
    Examples:
        pulsar agent-eval run configs/eval-suites/basic-agent-suite.yaml --agent-config configs/agents/my-assistant.yaml
    """
    import yaml as _yaml

    from pulsar_ai.evaluation.agent_eval import AgentEvaluator, load_suite_from_yaml
    from pulsar_ai.evaluation.agent_eval_store import AgentEvalStore

    suite = load_suite_from_yaml(suite_path)

    with open(agent_config, "r", encoding="utf-8") as f:
        config = _yaml.safe_load(f)

    console.print(
        Panel(
            f"Suite: [bold]{suite.name}[/bold] ({len(suite.cases)} cases)\n"
            f"Scoring: {scoring}",
            title="Agent Eval",
        )
    )

    evaluator = AgentEvaluator(config)
    report = evaluator.run_suite(suite, scoring=scoring)

    store = AgentEvalStore()
    report_id = store.save_report(report)

    table = Table(title=f"Eval Report: {report_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Suite", report.suite_name)
    table.add_row("Model", report.model_name)
    table.add_row("Success Rate", f"{report.success_rate * 100:.1f}%")
    table.add_row("Avg Score", f"{report.avg_score:.4f}")
    table.add_row("Avg Latency", f"{report.avg_latency_ms:.0f}ms")
    table.add_row("Total Tokens", str(report.total_tokens))
    table.add_row("Total Cost", f"${report.total_cost:.4f}")
    table.add_row("Tools Accuracy", f"{report.tools_accuracy * 100:.1f}%")
    console.print(table)

    # Per-case results
    cases_table = Table(title="Case Results")
    cases_table.add_column("Case ID", style="cyan")
    cases_table.add_column("Pass", style="green")
    cases_table.add_column("Score")
    cases_table.add_column("Latency")
    cases_table.add_column("Tools Match")
    cases_table.add_column("Error", style="red")

    for r in report.results:
        pass_str = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
        tools_str = "[green]yes[/green]" if r.tools_match else "[red]no[/red]"
        cases_table.add_row(
            r.case_id,
            pass_str,
            f"{r.score:.2f}",
            f"{r.latency_ms}ms",
            tools_str,
            r.error or "",
        )

    console.print(cases_table)
    console.print(f"\n[dim]Report saved: {report_id}[/dim]")


@agent_eval_group.command(name="list")
def eval_list() -> None:
    """List saved evaluation reports.

    \b
    Examples:
        pulsar agent-eval list
    """
    from pulsar_ai.evaluation.agent_eval_store import AgentEvalStore

    store = AgentEvalStore()
    reports = store.list_reports(limit=50)

    if not reports:
        console.print("[dim]No evaluation reports found.[/dim]")
        return

    table = Table(title="Evaluation Reports")
    table.add_column("ID", style="cyan")
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Timestamp")
    table.add_column("Success Rate", style="green")
    table.add_column("Avg Score")
    table.add_column("Latency")

    for r in reports:
        success_pct = f"{r['success_rate'] * 100:.1f}%"
        table.add_row(
            r["id"],
            r["suite_name"],
            r["model_name"],
            r["timestamp"],
            success_pct,
            f"{r['avg_score']:.4f}",
            f"{r['avg_latency_ms']:.0f}ms",
        )

    console.print(table)


@agent_eval_group.command(name="compare")
@click.argument("report_a")
@click.argument("report_b")
def eval_compare(report_a: str, report_b: str) -> None:
    """Compare two evaluation reports.

    \b
    Examples:
        pulsar agent-eval compare abc123 def456
    """
    from pulsar_ai.evaluation.agent_eval_store import AgentEvalStore

    store = AgentEvalStore()

    try:
        comparison = store.get_comparison(report_a, report_b)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    table = Table(title="Report Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("A", style="yellow")
    table.add_column("B", style="yellow")
    table.add_column("Delta")

    # Fetch both reports for absolute values
    a = store.get_report(report_a)
    b = store.get_report(report_b)

    if a and b:
        def _delta_str(delta: float, higher_is_better: bool = True) -> str:
            if abs(delta) < 0.0001:
                return "[dim]--[/dim]"
            arrow = "[green]+[/green]" if (delta > 0) == higher_is_better else "[red]-[/red]"
            return f"{arrow}{abs(delta):.4f}"

        table.add_row(
            "Model",
            comparison["model_a"],
            comparison["model_b"],
            "",
        )
        table.add_row(
            "Success Rate",
            f"{a['success_rate'] * 100:.1f}%",
            f"{b['success_rate'] * 100:.1f}%",
            _delta_str(comparison["success_delta"]),
        )
        table.add_row(
            "Avg Score",
            f"{a['avg_score']:.4f}",
            f"{b['avg_score']:.4f}",
            _delta_str(comparison["score_delta"]),
        )
        table.add_row(
            "Avg Latency",
            f"{a['avg_latency_ms']:.0f}ms",
            f"{b['avg_latency_ms']:.0f}ms",
            _delta_str(comparison["latency_delta"], higher_is_better=False),
        )
        table.add_row(
            "Total Cost",
            f"${a['total_cost']:.4f}",
            f"${b['total_cost']:.4f}",
            _delta_str(comparison["cost_delta"], higher_is_better=False),
        )

    console.print(table)
    winner = comparison["winner"]
    if winner == "tie":
        console.print("\n[dim]Result: Tie[/dim]")
    else:
        console.print(f"\n[bold green]Winner: {winner}[/bold green]")


# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚
# Web UI command
# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚


@main.command(name="ui")
@click.option("--host", default="0.0.0.0", help="Server host.")
@click.option("--port", type=int, default=8888, help="Server port.")
def ui(host: str, port: int) -> None:
    """Start the Web UI dashboard.

    \b
    Examples:
        pulsar ui
        pulsar ui --port 9000
    """
    from pulsar_ai.ui.app import start_ui_server

    console.print(
        Panel(
            f"Starting [bold]Pulsar AI UI[/bold]\n"
            f"Dashboard: http://{host}:{port}\n"
            f"API docs:  http://{host}:{port}/docs",
            title="Web UI",
        )
    )

    start_ui_server(host=host, port=port)


# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚
# Pipeline subgroup
# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚


@main.group()
def pipeline() -> None:
    """Pipeline orchestrator: run multi-step training pipelines."""


@pipeline.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
def pipeline_run(config_path: str) -> None:
    """Run a pipeline from YAML config.

    \b
    Examples:
        pulsar pipeline run configs/pipelines/example.yaml
    """
    import yaml

    with open(config_path, encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)

    name = pipeline_config.get("pipeline", {}).get("name", "unnamed")
    steps = pipeline_config.get("steps", [])

    console.print(
        Panel(
            f"Pipeline: [bold]{name}[/bold]\n" f"Steps: {len(steps)}",
            title="Pipeline Run",
        )
    )

    from pulsar_ai.pipeline.executor import PipelineExecutor

    executor = PipelineExecutor(pipeline_config)

    try:
        outputs = executor.run()
        console.print(f"\n[green]Pipeline '{name}' completed successfully![/green]")

        table = Table(title="Step Results")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Key Outputs")

        for step_name, result in outputs.items():
            keys = ", ".join(
                f"{k}={v}"
                for k, v in result.items()
                if isinstance(v, (str, int, float)) and k != "output_dir"
            )[:80]
            table.add_row(step_name, "completed", keys or "-")

        console.print(table)
    except RuntimeError as e:
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        sys.exit(1)


@pipeline.command(name="list")
@click.option("--name", default=None, help="Filter by pipeline name.")
def pipeline_list(name: str | None) -> None:
    """List past pipeline runs.

    \b
    Examples:
        pulsar pipeline list
        pulsar pipeline list --name full-pipeline
    """
    from pulsar_ai.pipeline.tracker import PipelineTracker

    runs = PipelineTracker.list_runs(pipeline_name=name)

    if not runs:
        console.print("[dim]No pipeline runs found.[/dim]")
        return

    table = Table(title="Pipeline Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Pipeline", style="green")
    table.add_column("Status")
    table.add_column("Started", style="dim")
    table.add_column("Steps")

    for run in runs:
        status = run.get("status", "unknown")
        style = "green" if status == "completed" else "red" if status == "failed" else "yellow"
        steps = run.get("steps", {})
        step_summary = (
            f"{sum(1 for s in steps.values() if s.get('status') == 'completed')}/{len(steps)}"
        )

        table.add_row(
            run.get("run_id", "?"),
            run.get("pipeline", "?"),
            f"[{style}]{status}[/{style}]",
            run.get("started_at", "?")[:19],
            step_summary,
        )

    console.print(table)


# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚
# Experiment tracking & comparison
# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚


@main.group()
def runs() -> None:
    """Experiment run tracking: list, compare, show."""


@runs.command(name="list")
@click.option("--project", default=None, help="Filter by project name.")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--limit", type=int, default=20, help="Max runs to show.")
def runs_list(project: str | None, status: str | None, limit: int) -> None:
    """List tracked experiment runs.

    \b
    Examples:
        pulsar runs list
        pulsar runs list --status completed --limit 10
    """
    from pulsar_ai.tracking import list_runs

    results = list_runs(project=project, status=status, limit=limit)
    if not results:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Experiment Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Loss")
    table.add_column("Backend", style="dim")

    for run in results:
        status_val = run.get("status", "?")
        style = (
            "green" if status_val == "completed" else "red" if status_val == "failed" else "yellow"
        )
        loss = run.get("results", {}).get("training_loss")
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "-"
        duration = run.get("duration_s")
        dur_str = f"{duration:.0f}s" if duration else "-"

        table.add_row(
            run.get("run_id", "?")[:12],
            run.get("name", "?")[:30],
            f"[{style}]{status_val}[/{style}]",
            dur_str,
            loss_str,
            run.get("backend", "?"),
        )

    console.print(table)


@runs.command(name="show")
@click.argument("run_id")
def runs_show(run_id: str) -> None:
    """Show details of a specific run.

    \b
    Examples:
        pulsar runs show abc123def456
    """
    from pulsar_ai.tracking import get_run

    run = get_run(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]{run.get('name', '?')}[/bold]\n"
            f"Status: {run.get('status')}\n"
            f"Duration: {run.get('duration_s', 0):.1f}s\n"
            f"Backend: {run.get('backend')}\n"
            f"Started: {run.get('started_at', '?')[:19]}",
            title=f"Run {run_id}",
        )
    )

    # Results
    results = run.get("results", {})
    if results:
        table = Table(title="Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in results.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            elif isinstance(v, (str, int)):
                table.add_row(k, str(v))
        console.print(table)

    # Environment
    env = run.get("environment", {})
    if env:
        table = Table(title="Environment")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="dim")
        for k, v in env.items():
            if k != "packages":
                table.add_row(k, str(v)[:60])
        console.print(table)


@runs.command(name="compare")
@click.argument("run_ids", nargs=-1, required=True)
def runs_compare(run_ids: tuple[str, ...]) -> None:
    """Compare experiment runs side by side.

    \b
    Examples:
        pulsar runs compare abc123 def456
        pulsar runs compare run1 run2 run3
    """
    from pulsar_ai.tracking import compare_runs

    result = compare_runs(list(run_ids))
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        sys.exit(1)

    # Config differences
    config_diff = result.get("config_diff", {})
    if config_diff:
        table = Table(title="Config Differences")
        table.add_column("Parameter", style="cyan")
        for name in result.get("run_names", []):
            table.add_column(name[:20], style="green")

        for key, values in config_diff.items():
            if not key.startswith("_"):
                table.add_row(key, *[str(v)[:20] for v in values])
        console.print(table)

    # Metrics comparison
    metrics = result.get("metrics_comparison", {})
    if metrics:
        table = Table(title="Metrics Comparison")
        table.add_column("Metric", style="cyan")
        for name in result.get("run_names", []):
            table.add_column(name[:20], style="green")

        for key, values in metrics.items():
            table.add_row(
                key,
                *[f"{v:.4f}" if isinstance(v, float) else str(v) for v in values],
            )
        console.print(table)


# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚
# HPO sweep
# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("sweep_config_path", type=click.Path(exists=True))
@click.option("--n-trials", type=int, default=None, help="Override number of trials.")
@click.option("--name", default=None, help="Study name.")
def sweep(
    config_path: str,
    sweep_config_path: str,
    n_trials: int | None,
    name: str | None,
) -> None:
    """Run hyperparameter optimization sweep.

    \b
    Examples:
        pulsar sweep configs/experiments/sft.yaml configs/sweeps/lr-search.yaml
        pulsar sweep configs/experiments/sft.yaml configs/sweeps/full.yaml --n-trials 30
    """
    from pulsar_ai.hpo.sweep import SweepRunner, load_sweep_config

    sweep_config = load_sweep_config(sweep_config_path)
    sweep_conf = sweep_config.get("hpo", sweep_config)

    console.print(
        Panel(
            f"Sweep config: {sweep_config_path}\n"
            f"Base config: {config_path}\n"
            f"Trials: {n_trials or sweep_conf.get('n_trials', 10)}\n"
            f"Metric: {sweep_conf.get('metric', 'training_loss')}\n"
            f"Direction: {sweep_conf.get('direction', 'minimize')}\n"
            f"Search space: {len(sweep_conf.get('search_space', {}))} parameters",
            title="HPO Sweep",
        )
    )

    runner = SweepRunner(
        base_config_path=config_path,
        sweep_config=sweep_config,
        study_name=name,
    )

    results = runner.run(n_trials=n_trials)

    console.print("\n[green]Sweep completed![/green]")
    console.print(f"Best trial: #{results['best_trial']}")
    console.print(f"Best value: {results['best_value']:.6f}")

    table = Table(title="Best Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for k, v in results["best_params"].items():
        table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
    console.print(table)


# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚
# Model registry
# РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚РІвЂќР‚


@main.group()
def registry() -> None:
    """Model registry: register, list, promote, compare."""


@registry.command(name="list")
@click.option("--name", default=None, help="Filter by model name.")
@click.option("--status", default=None, help="Filter by status.")
def registry_list(name: str | None, status: str | None) -> None:
    """List registered models.

    \b
    Examples:
        pulsar registry list
        pulsar registry list --name customer-intent --status production
    """
    from pulsar_ai.registry import ModelRegistry

    reg = ModelRegistry()
    models = reg.list_models(name=name, status=status)

    if not models:
        console.print("[dim]No models registered.[/dim]")
        return

    table = Table(title="Model Registry")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Base Model")
    table.add_column("Created", style="dim")

    for m in models:
        status_val = m.get("status", "?")
        style = (
            "green"
            if status_val == "production"
            else "yellow" if status_val == "staging" else "dim"
        )
        table.add_row(
            m["id"],
            m["name"],
            str(m["version"]),
            m.get("task", ""),
            f"[{style}]{status_val}[/{style}]",
            m.get("base_model", "")[:25],
            m.get("created_at", "?")[:10],
        )

    console.print(table)


@registry.command(name="register")
@click.argument("name")
@click.option("--model-path", required=True, help="Path to model/adapter.")
@click.option("--task", default="sft", help="Training task.")
@click.option("--base-model", default="", help="Base model name.")
@click.option("--tag", multiple=True, help="Tags (repeatable).")
def registry_register(
    name: str,
    model_path: str,
    task: str,
    base_model: str,
    tag: tuple[str, ...],
) -> None:
    """Register a model in the registry.

    \b
    Examples:
        pulsar registry register customer-intent --model-path ./outputs/sft/lora --base-model qwen2.5-3b
    """
    from pulsar_ai.registry import ModelRegistry

    reg = ModelRegistry()
    entry = reg.register(
        name=name,
        model_path=model_path,
        task=task,
        base_model=base_model,
        tags=list(tag),
    )
    console.print(f"[green]Registered:[/green] {entry['id']} ({entry['model_path']})")


@registry.command(name="promote")
@click.argument("model_id")
@click.argument(
    "status",
    type=click.Choice(["staging", "production", "archived"]),
)
def registry_promote(model_id: str, status: str) -> None:
    """Promote a model to a new status.

    \b
    Examples:
        pulsar registry promote customer-intent-v2 production
    """
    from pulsar_ai.registry import ModelRegistry

    reg = ModelRegistry()
    entry = reg.update_status(model_id, status)
    if entry:
        console.print(f"[green]{model_id}[/green] РІвЂ вЂ™ {status}")
    else:
        console.print(f"[red]Model not found: {model_id}[/red]")


def _show_config_summary(config: dict, task: str) -> None:
    """Display config summary panel.

    Args:
        config: Resolved config dict.
        task: Training task name.
    """
    table = Table(title=f"Training Config - {task.upper()}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    model_name = config.get("model", {}).get("name", "unknown")
    strategy = config.get("_detected_strategy", config.get("strategy", "unknown"))
    training = config.get("training", {})

    table.add_row("Model", model_name)
    table.add_row("Strategy", strategy)
    table.add_row("Learning Rate", str(training.get("learning_rate", "-")))
    table.add_row("Epochs", str(training.get("epochs", "-")))
    table.add_row("Batch Size", str(training.get("batch_size", "-")))
    table.add_row(
        "Gradient Accum",
        str(training.get("gradient_accumulation", "-")),
    )
    table.add_row(
        "Max Seq Length",
        str(training.get("max_seq_length", "-")),
    )

    hw = config.get("_hardware", {})
    if hw:
        table.add_row(
            "GPU",
            f"{hw.get('num_gpus', '?')}x {hw.get('gpu_name', '?')} "
            f"({hw.get('vram_per_gpu_gb', '?')} GB)",
        )

    console.print(table)


# ---------------------------------------------------------------
# Recipe Hub
# ---------------------------------------------------------------


@main.group()
def recipes() -> None:
    """Browse and run recipe templates."""


@recipes.command(name="list")
@click.option("--task", "task_type", default=None, help="Filter by task.")
@click.option("--tag", default=None, help="Filter by tag.")
@click.option(
    "--difficulty",
    default=None,
    type=click.Choice(["beginner", "intermediate", "advanced"]),
    help="Filter by difficulty.",
)
def recipes_list(
    task_type: Optional[str],
    tag: Optional[str],
    difficulty: Optional[str],
) -> None:
    """List available recipes.

    \b
    Examples:
        pulsar recipes list
        pulsar recipes list --task sft --difficulty beginner
    """
    from pulsar_ai.recipes import RecipeRegistry

    registry = RecipeRegistry()
    items = registry.list_recipes(task_type=task_type, tag=tag, difficulty=difficulty)

    if not items:
        console.print("[dim]No recipes found.[/dim]")
        return

    table = Table(title="Recipes")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Difficulty")
    table.add_column("Hardware", style="dim")
    table.add_column("Description")

    for r in items:
        diff = r.get("difficulty", "?")
        diff_style = {
            "beginner": "green",
            "intermediate": "yellow",
            "advanced": "red",
        }.get(diff, "white")
        table.add_row(
            r.get("file", "?"),
            r.get("task_type", "?"),
            f"[{diff_style}]{diff}[/{diff_style}]",
            r.get("hardware", "?"),
            (r.get("description", "") or "")[:60],
        )

    console.print(table)


@recipes.command(name="run")
@click.argument("name")
@click.argument("overrides", nargs=-1)
def recipes_run(name: str, overrides: tuple[str, ...]) -> None:
    """Run a recipe by name with optional key=value overrides.

    \b
    Examples:
        pulsar recipes run llama-instruct-sft
        pulsar recipes run dpo-starter training.epochs=5
    """
    from pulsar_ai.recipes import RecipeRegistry
    from pulsar_ai.config import load_config, _set_nested

    registry = RecipeRegistry()
    try:
        config = registry.load_recipe(name)
    except FileNotFoundError:
        console.print(f"[red]Recipe not found: {name}[/red]")
        sys.exit(1)

    parsed = _parse_overrides(overrides)
    for key, value in parsed.items():
        _set_nested(config, key, value)

    task = config.get("task", "sft")
    console.print(
        Panel(
            f"Recipe: [bold]{name}[/bold]\n"
            f"Task: {task}\n"
            f"Model: {config.get('model', {}).get('name', '?')}",
            title="Recipe Run",
        )
    )

    try:
        if task == "sft":
            from pulsar_ai.training.sft import train_sft

            result = train_sft(config)
        elif task == "dpo":
            from pulsar_ai.training.dpo import train_dpo

            result = train_dpo(config)
        elif task == "grpo":
            try:
                from pulsar_ai.training.grpo import train_grpo
            except ImportError:
                console.print(
                    "[red]GRPO requires TRL >= 0.14.[/red]" " Install: pip install 'trl>=0.14,<1.0'"
                )
                sys.exit(1)
            result = train_grpo(config)
        elif task == "embedding":
            try:
                from pulsar_ai.training.embedding import train_embedding
            except ImportError:
                console.print(
                    "[red]Embedding requires sentence-transformers >= 3.0.[/red]"
                    " Install: pip install 'pulsar-ai[embedding]'"
                )
                sys.exit(1)
            result = train_embedding(config)
        elif task == "reranker":
            try:
                from pulsar_ai.training.reranker import train_reranker
            except ImportError:
                console.print(
                    "[red]Reranker requires sentence-transformers >= 3.0.[/red]"
                    " Install: pip install 'pulsar-ai[embedding]'"
                )
                sys.exit(1)
            result = train_reranker(config)
        elif task == "classification":
            try:
                from pulsar_ai.training.classification import train_classification
            except ImportError:
                console.print(
                    "[red]Classification requires scikit-learn.[/red]"
                    " Install: pip install 'pulsar-ai[classification]'"
                )
                sys.exit(1)
            result = train_classification(config)
        else:
            console.print(f"[red]Unknown task: {task}[/red]")
            sys.exit(1)

        console.print("[green]Recipe completed successfully![/green]")
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (str, int, float)):
                    console.print(f"  {k}: {v}")
    except Exception as exc:
        console.print(f"[red]Recipe failed:[/red] {exc}")
        logger.exception("Recipe run error for %s", name)
        sys.exit(1)


def _show_training_results(results: dict) -> None:
    """Display training results panel.

    Args:
        results: Dict with training results.
    """
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


@main.command()
@click.option("--from-traces", is_flag=True, help="Build dataset from stored traces.")
@click.option("--days", type=int, default=7, help="Collect traces from last N days.")
@click.option("--min-rating", type=float, default=0, help="Minimum trace rating.")
@click.option(
    "--task",
    type=click.Choice(["sft", "dpo"]),
    default="dpo",
    help="Training task.",
)
@click.option("--model", type=click.Path(), required=True, help="Base model path.")
@click.option("--output", type=click.Path(), default="./outputs/retrain", help="Output directory.")
def retrain(
    from_traces: bool,
    days: int,
    min_rating: float,
    task: str,
    model: str,
    output: str,
) -> None:
    """Retrain model from collected traces (closed-loop).

    \b
    Examples:
        pulsar retrain --from-traces --model ./models/base --task dpo
        pulsar retrain --from-traces --days 14 --min-rating 0.7 --model ./models/base
    """
    from pulsar_ai.pipeline.closed_loop_steps import (
        step_collect_traces,
        step_build_dataset,
    )

    if not from_traces:
        console.print("[yellow]Use --from-traces to build dataset from stored traces.[/yellow]")
        sys.exit(1)

    # 1. Collect traces
    console.print(Panel(f"Collecting traces (last {days} days, min_rating={min_rating})..."))
    collect_result = step_collect_traces({"days": days, "min_rating": min_rating, "limit": 500})
    trace_count = collect_result["count"]
    console.print(f"  Found [bold]{trace_count}[/bold] traces")

    if trace_count == 0:
        console.print("[yellow]No traces found. Nothing to retrain on.[/yellow]")
        return

    # 2. Build dataset
    console.print(Panel(f"Building {task} dataset from {trace_count} traces..."))
    dataset_result = step_build_dataset(
        {
            "trace_ids": collect_result["trace_ids"],
            "format": task,
            "output_dir": f"{output}/data",
            "name": f"retrain-{task}",
            "quality_filter": {"dedup": True, "min_length": 10},
        }
    )
    console.print(
        f"  Dataset: [bold]{dataset_result['path']}[/bold] "
        f"({dataset_result['num_examples']} examples)"
    )

    if dataset_result["num_examples"] == 0:
        console.print("[yellow]Empty dataset generated. Nothing to train on.[/yellow]")
        return

    # 3. Train
    console.print(Panel(f"Training {task} on [bold]{model}[/bold]..."))
    train_config = {
        "task": task,
        "model": {"name": model},
        "dataset": {"path": dataset_result["path"]},
        "output": {"dir": output},
    }

    from pulsar_ai.pipeline.steps import dispatch_step

    try:
        train_result = dispatch_step("training", train_config)
    except Exception as e:
        console.print(f"[red]Training failed:[/red] {e}")
        sys.exit(1)

    # 4. Show results
    table = Table(title="Retrain Results")
    table.add_column("Step", style="cyan")
    table.add_column("Result", style="green")
    table.add_row("Traces collected", str(trace_count))
    table.add_row("Dataset examples", str(dataset_result["num_examples"]))
    table.add_row("Dataset path", dataset_result["path"])
    table.add_row("Task", task)
    table.add_row("Output", train_result.get("adapter_dir", output))
    console.print(table)
    console.print("[green]Retrain pipeline complete![/green]")


def _show_eval_results(results: dict) -> None:
    """Display evaluation results panel.

    Args:
        results: Dict with evaluation results.
    """
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        if key == "per_class" and isinstance(value, dict):
            continue
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)

    # Per-class breakdown if available
    per_class = results.get("per_class")
    if per_class and isinstance(per_class, dict):
        cls_table = Table(title="Per-Class Accuracy")
        cls_table.add_column("Class", style="cyan")
        cls_table.add_column("Accuracy", style="green")
        cls_table.add_column("Count", style="yellow")

        for cls_name, cls_data in sorted(per_class.items()):
            if isinstance(cls_data, dict):
                acc = cls_data.get("accuracy", 0)
                count = cls_data.get("count", 0)
                cls_table.add_row(cls_name, f"{acc:.2%}", str(count))

        console.print(cls_table)


# ── OpenClaw commands ────────────────────────────────────────────


@main.group()
def openclaw() -> None:
    """OpenClaw runtime management."""


@openclaw.command()
def health() -> None:
    """Check OpenClaw runtime health."""
    from pulsar_ai.openclaw.adapter import OpenClawAdapter

    adapter = OpenClawAdapter()
    result = adapter.health_check()
    status = result.get("status", "unknown")
    color = "green" if status == "healthy" else "red"
    console.print(f"OpenClaw status: [{color}]{status}[/{color}]")
    if result.get("error"):
        console.print(f"  Error: {result['error']}")


@openclaw.command(name="sessions")
def list_sessions() -> None:
    """List active OpenClaw sessions."""
    from pulsar_ai.openclaw.adapter import OpenClawAdapter

    adapter = OpenClawAdapter()
    sessions = adapter.list_sessions()

    if not sessions:
        console.print("[dim]No active sessions[/dim]")
        return

    table = Table(title="OpenClaw Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Agent", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Created", style="dim")

    for s in sessions:
        table.add_row(
            s.session_id,
            s.agent_name,
            s.model,
            s.status,
            s.created_at,
        )
    console.print(table)


@openclaw.command(name="deploy")
@click.argument("agent_config", type=click.Path(exists=True))
@click.option("--sandbox/--no-sandbox", default=True, help="Enable NemoClaw sandbox.")
def deploy(agent_config: str, sandbox: bool) -> None:
    """Deploy agent via OpenClaw (with optional NemoClaw sandbox).

    \b
    Examples:
        pulsar openclaw deploy configs/agents/my-agent.yaml
        pulsar openclaw deploy configs/agents/my-agent.yaml --no-sandbox
    """
    import yaml

    from pulsar_ai.openclaw.adapter import OpenClawAdapter
    from pulsar_ai.openclaw.nemoclaw import NemoClawManager, SandboxPolicy

    with open(agent_config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    adapter = OpenClawAdapter()

    if sandbox:
        manager = NemoClawManager(adapter)
        policy = SandboxPolicy()
        deployment = manager.deploy(config.get("agent", config), policy)
        console.print(
            Panel(
                f"[green]Deployed:[/green] {deployment.deployment_id}\n"
                f"Session: {deployment.session_id}\n"
                f"Sandbox: enabled\n"
                f"Status: {deployment.status}",
                title="NemoClaw Deployment",
            )
        )
    else:
        session = adapter.create_session(config.get("agent", config))
        console.print(
            Panel(
                f"[green]Created:[/green] {session.session_id}\n"
                f"Agent: {session.agent_name}\n"
                f"Status: {session.status}",
                title="OpenClaw Session",
            )
        )
