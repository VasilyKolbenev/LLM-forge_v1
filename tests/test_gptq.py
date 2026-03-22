"""Tests for GPTQ export module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pulsar_ai.export.gptq import (
    _get_gptq_config,
    export_gptq,
)


class TestExportGptqValidation:
    """Tests for export_gptq input validation."""

    def test_missing_model_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="model_path is required"):
            export_gptq({"export": {"format": "gptq"}})

    def test_empty_config_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="model_path is required"):
            export_gptq({})


class TestGetGptqConfig:
    """Tests for _get_gptq_config defaults and validation."""

    def test_defaults(self) -> None:
        result = _get_gptq_config({})
        assert result["bits"] == 4
        assert result["group_size"] == 128
        assert result["desc_act"] is False

    def test_custom_values(self) -> None:
        result = _get_gptq_config({
            "bits": 8,
            "group_size": 64,
            "desc_act": True,
        })
        assert result["bits"] == 8
        assert result["group_size"] == 64
        assert result["desc_act"] is True

    def test_invalid_bits_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid GPTQ bits=5"):
            _get_gptq_config({"bits": 5})

    def test_invalid_bits_16_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid GPTQ bits=16"):
            _get_gptq_config({"bits": 16})

    def test_invalid_group_size_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match="Invalid GPTQ group_size=256"
        ):
            _get_gptq_config({"group_size": 256})

    def test_invalid_group_size_16_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match="Invalid GPTQ group_size=16"
        ):
            _get_gptq_config({"group_size": 16})

    def test_all_valid_bits(self) -> None:
        for bits in (2, 3, 4, 8):
            result = _get_gptq_config({"bits": bits})
            assert result["bits"] == bits

    def test_all_valid_group_sizes(self) -> None:
        for gs in (32, 64, 128):
            result = _get_gptq_config({"group_size": gs})
            assert result["group_size"] == gs


class TestExportGptqEndToEnd:
    """Tests for export_gptq mocked end-to-end."""

    @patch("pulsar_ai.export.gptq._quantize_gptq")
    @patch("pulsar_ai.export.gptq._dir_size_mb")
    @patch("pulsar_ai.export.awq._load_calibration_data")
    def test_export_gptq_success(
        self,
        mock_cal: MagicMock,
        mock_size: MagicMock,
        mock_quant: MagicMock,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "model-gptq"
        mock_cal.return_value = ["sample text"]
        mock_quant.return_value = str(output_dir)
        mock_size.return_value = 2000.0

        config = {
            "model_path": str(tmp_path / "base-model"),
            "export": {
                "format": "gptq",
                "output_path": str(output_dir),
                "gptq": {
                    "bits": 4,
                    "group_size": 128,
                },
            },
        }

        result = export_gptq(config)

        assert result["output_path"] == str(output_dir)
        assert result["gptq_config"]["bits"] == 4
        assert result["file_size_mb"] == 2000.0
        mock_quant.assert_called_once()
