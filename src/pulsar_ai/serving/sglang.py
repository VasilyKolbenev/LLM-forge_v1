"""SGLang server for high-throughput model serving."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def start_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    mem_fraction: Optional[float] = None,
) -> None:
    """Start SGLang OpenAI-compatible server.

    Args:
        model_path: Path to model directory (HF format, not GGUF).
        host: Server host.
        port: Server port.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        mem_fraction: GPU memory fraction to use.

    Raises:
        ImportError: If sglang is not installed.
        ValueError: If model format is not supported.
    """
    model_dir = Path(model_path)
    if model_dir.suffix == ".gguf" or (model_dir.is_file() and model_dir.name.endswith(".gguf")):
        raise ValueError(
            "SGLang requires HuggingFace model format, not GGUF. "
            "Use 'pulsar serve --backend llamacpp' for GGUF models, "
            "or 'pulsar export --format merged' first."
        )

    args = _build_server_args(model_path, host, port, tensor_parallel_size, mem_fraction)

    logger.info(
        "Starting SGLang server on %s:%d (model: %s)",
        host,
        port,
        model_path,
    )
    logger.info(
        "OpenAI-compatible API: http://%s:%d/v1/chat/completions",
        host,
        port,
    )

    # Try Python API first, fall back to subprocess
    try:
        _run_via_python_api(args)
    except ImportError:
        raise ImportError(
            "sglang is not installed. " "Install with: pip install 'pulsar-ai[sglang]'"
        )


def _build_server_args(
    model_path: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    mem_fraction: Optional[float] = None,
) -> list[str]:
    """Build CLI argument list for SGLang server.

    Args:
        model_path: Path to model directory.
        host: Server host.
        port: Server port.
        tensor_parallel_size: Number of GPUs.
        mem_fraction: GPU memory fraction.

    Returns:
        List of CLI argument strings.
    """
    args = [
        "--model-path",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(tensor_parallel_size),
    ]

    if mem_fraction is not None:
        args.extend(["--mem-fraction-static", str(mem_fraction)])

    return args


def _run_via_python_api(args: list[str]) -> None:
    """Start SGLang server via Python API.

    Args:
        args: CLI argument list.

    Raises:
        ImportError: If sglang is not installed.
    """
    from sglang.srt.server import launch_server
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs.from_cli_args(args)
    launch_server(server_args)


def _run_via_subprocess(args: list[str]) -> None:
    """Start SGLang server via subprocess fallback.

    Args:
        args: CLI argument list.
    """
    cmd = ["python", "-m", "sglang.launch_server"] + args
    logger.info("Running: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped")
