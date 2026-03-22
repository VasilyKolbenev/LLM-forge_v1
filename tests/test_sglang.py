"""Tests for SGLang serving module."""

from unittest.mock import patch

import pytest

from pulsar_ai.serving.sglang import (
    _build_server_args,
    start_server,
)


class TestBuildServerArgs:
    """Tests for _build_server_args."""

    def test_default_args(self) -> None:
        args = _build_server_args(
            "/models/my-model", "0.0.0.0", 8000, 1
        )
        assert args == [
            "--model-path",
            "/models/my-model",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--tp-size",
            "1",
        ]

    def test_custom_port_and_tp(self) -> None:
        args = _build_server_args(
            "/models/llama", "127.0.0.1", 9090, 4
        )
        assert "--port" in args
        assert args[args.index("--port") + 1] == "9090"
        assert args[args.index("--tp-size") + 1] == "4"

    def test_mem_fraction_included(self) -> None:
        args = _build_server_args(
            "/models/m", "0.0.0.0", 8000, 1,
            mem_fraction=0.85,
        )
        assert "--mem-fraction-static" in args
        assert args[args.index("--mem-fraction-static") + 1] == "0.85"

    def test_mem_fraction_omitted_when_none(self) -> None:
        args = _build_server_args(
            "/models/m", "0.0.0.0", 8000, 1,
            mem_fraction=None,
        )
        assert "--mem-fraction-static" not in args


class TestStartServer:
    """Tests for start_server validation."""

    def test_rejects_gguf_file(self) -> None:
        with pytest.raises(
            ValueError, match="SGLang requires HuggingFace"
        ):
            start_server("/models/model.gguf")

    def test_rejects_gguf_suffix(self) -> None:
        with pytest.raises(ValueError, match="not GGUF"):
            start_server("path/to/weights.gguf")

    def test_raises_import_error_without_sglang(self) -> None:
        with patch(
            "pulsar_ai.serving.sglang._run_via_python_api",
            side_effect=ImportError("No module named 'sglang'"),
        ):
            with pytest.raises(
                ImportError, match="sglang is not installed"
            ):
                start_server("/models/hf-model")
