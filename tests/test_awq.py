"""Tests for AWQ export module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pulsar_ai.export.awq import (
    _get_awq_config,
    _load_calibration_data,
    export_awq,
)


class TestExportAwqValidation:
    """Tests for export_awq input validation."""

    def test_missing_model_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="model_path is required"):
            export_awq({"export": {"format": "awq"}})

    def test_empty_config_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="model_path is required"):
            export_awq({})


class TestGetAwqConfig:
    """Tests for _get_awq_config defaults and presets."""

    def test_default_preset_w4_g128(self) -> None:
        result = _get_awq_config({})
        assert result["w_bit"] == 4
        assert result["q_group_size"] == 128
        assert result["zero_point"] is True
        assert result["version"] == "GEMM"

    def test_preset_w4_g128_explicit(self) -> None:
        result = _get_awq_config({"quant_config": "w4-g128"})
        assert result["w_bit"] == 4
        assert result["q_group_size"] == 128

    def test_preset_w4_g64(self) -> None:
        result = _get_awq_config({"quant_config": "w4-g64"})
        assert result["w_bit"] == 4
        assert result["q_group_size"] == 64

    def test_custom_config_values(self) -> None:
        result = _get_awq_config({
            "quant_config": "custom",
            "w_bit": 8,
            "q_group_size": 32,
        })
        assert result["w_bit"] == 8
        assert result["q_group_size"] == 32


class TestLoadCalibrationData:
    """Tests for _load_calibration_data."""

    def test_dummy_fallback_when_no_data(self) -> None:
        config: dict = {
            "export": {"awq": {"calibration_data": "auto"}},
        }
        result = _load_calibration_data(config)
        assert len(result) == 128
        assert "quick brown fox" in result[0]

    def test_dummy_fallback_with_custom_samples(self) -> None:
        config: dict = {
            "export": {
                "awq": {
                    "calibration_data": "auto",
                    "calibration_samples": 16,
                }
            },
        }
        result = _load_calibration_data(config)
        assert len(result) == 16

    def test_loads_from_jsonl_file(self, tmp_path: Path) -> None:
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text(
            '{"text": "sample one"}\n'
            '{"text": "sample two"}\n'
        )
        config = {
            "export": {
                "awq": {
                    "calibration_data": str(cal_file),
                    "text_column": "text",
                    "calibration_samples": 10,
                }
            },
        }
        result = _load_calibration_data(config)
        assert result == ["sample one", "sample two"]

    def test_loads_from_training_dataset_auto(
        self, tmp_path: Path
    ) -> None:
        dataset_file = tmp_path / "train.jsonl"
        dataset_file.write_text('{"text": "training sample"}\n')
        config = {
            "export": {"awq": {"calibration_data": "auto"}},
            "dataset": {"path": str(dataset_file)},
        }
        result = _load_calibration_data(config)
        assert result == ["training sample"]


class TestExportAwqEndToEnd:
    """Tests for export_awq mocked end-to-end."""

    @patch("pulsar_ai.export.awq._quantize_awq")
    @patch("pulsar_ai.export.awq._load_calibration_data")
    @patch("pulsar_ai.export.awq._dir_size_mb")
    def test_export_awq_success(
        self,
        mock_size: MagicMock,
        mock_cal: MagicMock,
        mock_quant: MagicMock,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "model-awq"
        mock_cal.return_value = ["sample text"]
        mock_quant.return_value = str(output_dir)
        mock_size.return_value = 1500.0

        config = {
            "model_path": str(tmp_path / "base-model"),
            "export": {
                "format": "awq",
                "output_path": str(output_dir),
                "awq": {"quant_config": "w4-g128"},
            },
        }

        result = export_awq(config)

        assert result["output_path"] == str(output_dir)
        assert result["quant_config"]["w_bit"] == 4
        assert result["file_size_mb"] == 1500.0
        mock_quant.assert_called_once()
