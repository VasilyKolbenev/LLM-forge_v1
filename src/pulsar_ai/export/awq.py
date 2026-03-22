"""AWQ (Activation-aware Weight Quantization) export."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# AWQ preset configurations
_AWQ_PRESETS: dict[str, dict] = {
    "w4-g128": {"w_bit": 4, "q_group_size": 128},
    "w4-g64": {"w_bit": 4, "q_group_size": 64},
}


def export_awq(config: dict) -> dict:
    """Export model to AWQ quantized format.

    Args:
        config: Config dict with model_path and export.awq settings.

    Returns:
        Dict with output_path, quant_config, and file_size_mb.

    Raises:
        ValueError: If model_path is missing.
        ImportError: If autoawq is not installed.
    """
    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("model_path is required for AWQ export")

    export_config = config.get("export", {})
    awq_section = export_config.get("awq", {})
    output_path = export_config.get(
        "output_path",
        str(Path(model_path).parent / "model-awq"),
    )

    quant_config = _get_awq_config(awq_section)
    calibration_data = _load_calibration_data(config)

    logger.info(
        "Starting AWQ quantization: %s -> %s (w%d-g%d)",
        model_path,
        output_path,
        quant_config["w_bit"],
        quant_config["q_group_size"],
    )

    result_path = _quantize_awq(
        model_path, output_path, quant_config, calibration_data
    )

    file_size_mb = _dir_size_mb(result_path)
    logger.info(
        "AWQ export complete: %s (%.1f MB)", result_path, file_size_mb
    )

    return {
        "output_path": result_path,
        "quant_config": quant_config,
        "file_size_mb": round(file_size_mb, 1),
    }


def _get_awq_config(awq_section: dict) -> dict:
    """Extract AWQ quantization parameters from config.

    Args:
        awq_section: The export.awq section of config.

    Returns:
        Dict with w_bit, q_group_size, zero_point, version.
    """
    preset_name = awq_section.get("quant_config", "w4-g128")

    if preset_name in _AWQ_PRESETS:
        params = dict(_AWQ_PRESETS[preset_name])
    else:
        params = {
            "w_bit": awq_section.get("w_bit", 4),
            "q_group_size": awq_section.get("q_group_size", 128),
        }

    params.setdefault("zero_point", True)
    params.setdefault("version", "GEMM")
    return params


def _load_calibration_data(config: dict) -> list[str]:
    """Load calibration data for AWQ quantization.

    Tries in order:
    1. Explicit calibration_data path from config
    2. Training dataset (if calibration_data == "auto")
    3. Dummy fallback data

    Args:
        config: Full config dict.

    Returns:
        List of text samples for calibration.
    """
    export_config = config.get("export", {})
    awq_section = export_config.get("awq", {})
    cal_path = awq_section.get("calibration_data", "auto")
    cal_samples = awq_section.get("calibration_samples", 128)
    text_column = awq_section.get("text_column", "text")

    # Option 1: explicit file path
    if cal_path not in ("auto", None) and Path(cal_path).exists():
        return _read_calibration_file(
            cal_path, text_column, cal_samples
        )

    # Option 2: reuse training dataset
    if cal_path == "auto":
        dataset_path = config.get("dataset", {}).get("path")
        if dataset_path and Path(dataset_path).exists():
            logger.info(
                "Using training dataset for calibration: %s",
                dataset_path,
            )
            return _read_calibration_file(
                dataset_path, text_column, cal_samples
            )

    # Option 3: dummy fallback
    logger.warning(
        "No calibration data found, using dummy data. "
        "Quality may be reduced."
    )
    return [
        "The quick brown fox jumps over the lazy dog."
    ] * cal_samples


def _read_calibration_file(
    path: str, text_column: str, max_samples: int
) -> list[str]:
    """Read calibration samples from a dataset file.

    Args:
        path: Path to dataset file (jsonl, csv, or json).
        text_column: Column name containing text data.
        max_samples: Maximum number of samples to load.

    Returns:
        List of text strings.
    """
    import json

    file_path = Path(path)
    samples: list[str] = []

    if file_path.suffix == ".jsonl":
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                row = json.loads(line)
                if text_column in row:
                    samples.append(str(row[text_column]))
    elif file_path.suffix == ".csv":
        import csv

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break
                if text_column in row:
                    samples.append(str(row[text_column]))
    elif file_path.suffix == ".json":
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for row in data[:max_samples]:
                if isinstance(row, dict) and text_column in row:
                    samples.append(str(row[text_column]))

    if not samples:
        logger.warning(
            "No samples found in %s (column=%s), "
            "falling back to dummy data",
            path,
            text_column,
        )
        return ["Dummy calibration text."] * max_samples

    return samples


def _quantize_awq(
    model_path: str,
    output_path: str,
    quant_config: dict,
    calibration_data: list[str],
) -> str:
    """Run AWQ quantization using autoawq library.

    Args:
        model_path: Path to HuggingFace model.
        output_path: Directory for quantized output.
        quant_config: AWQ quantization parameters.
        calibration_data: Calibration text samples.

    Returns:
        Path to quantized model directory.

    Raises:
        ImportError: If autoawq is not installed.
    """
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError(
            "autoawq is not installed. "
            "Install with: pip install 'pulsar-ai[awq]'"
        )

    from transformers import AutoTokenizer

    try:
        logger.info("Loading model for AWQ quantization: %s", model_path)
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info(
            "Quantizing with w_bit=%d, group_size=%d",
            quant_config["w_bit"],
            quant_config["q_group_size"],
        )
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data,
        )

        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info("AWQ model saved to: %s", output_path)
        return output_path

    except Exception:
        if Path(output_path).exists():
            logger.warning(
                "Cleaning up failed export: %s", output_path
            )
            shutil.rmtree(output_path)
        raise


def _dir_size_mb(path: str) -> float:
    """Calculate total size of directory in MB.

    Args:
        path: Directory path.

    Returns:
        Size in megabytes.
    """
    total = sum(
        f.stat().st_size
        for f in Path(path).rglob("*")
        if f.is_file()
    )
    return total / (1024 * 1024)
