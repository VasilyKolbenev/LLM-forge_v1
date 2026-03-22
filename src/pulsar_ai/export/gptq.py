"""GPTQ (GPT Post-Training Quantization) export."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_BITS = {2, 3, 4, 8}
_VALID_GROUP_SIZES = {32, 64, 128}


def export_gptq(config: dict) -> dict:
    """Export model to GPTQ quantized format.

    Args:
        config: Config dict with model_path and export.gptq settings.

    Returns:
        Dict with output_path, gptq_config, and file_size_mb.

    Raises:
        ValueError: If model_path is missing or params are invalid.
        ImportError: If auto-gptq is not installed.
    """
    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("model_path is required for GPTQ export")

    export_config = config.get("export", {})
    gptq_section = export_config.get("gptq", {})
    output_path = export_config.get(
        "output_path",
        str(Path(model_path).parent / "model-gptq"),
    )

    gptq_config = _get_gptq_config(gptq_section)

    # Reuse AWQ calibration loader
    from pulsar_ai.export.awq import _load_calibration_data

    calibration_data = _load_calibration_data(config)

    logger.info(
        "Starting GPTQ quantization: %s -> %s "
        "(%d-bit, group_size=%d)",
        model_path,
        output_path,
        gptq_config["bits"],
        gptq_config["group_size"],
    )

    result_path = _quantize_gptq(
        model_path, output_path, gptq_config, calibration_data
    )

    file_size_mb = _dir_size_mb(result_path)
    logger.info(
        "GPTQ export complete: %s (%.1f MB)",
        result_path,
        file_size_mb,
    )

    return {
        "output_path": result_path,
        "gptq_config": gptq_config,
        "file_size_mb": round(file_size_mb, 1),
    }


def _get_gptq_config(gptq_section: dict) -> dict:
    """Extract and validate GPTQ quantization parameters.

    Args:
        gptq_section: The export.gptq section of config.

    Returns:
        Dict with bits, group_size, desc_act.

    Raises:
        ValueError: If bits or group_size are invalid.
    """
    bits = gptq_section.get("bits", 4)
    group_size = gptq_section.get("group_size", 128)
    desc_act = gptq_section.get("desc_act", False)

    if bits not in _VALID_BITS:
        raise ValueError(
            f"Invalid GPTQ bits={bits}. "
            f"Must be one of {sorted(_VALID_BITS)}"
        )

    if group_size not in _VALID_GROUP_SIZES:
        raise ValueError(
            f"Invalid GPTQ group_size={group_size}. "
            f"Must be one of {sorted(_VALID_GROUP_SIZES)}"
        )

    return {
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
    }


def _quantize_gptq(
    model_path: str,
    output_path: str,
    gptq_config: dict,
    calibration_data: list[str],
) -> str:
    """Run GPTQ quantization using auto-gptq library.

    Args:
        model_path: Path to HuggingFace model.
        output_path: Directory for quantized output.
        gptq_config: GPTQ quantization parameters.
        calibration_data: Calibration text samples.

    Returns:
        Path to quantized model directory.

    Raises:
        ImportError: If auto-gptq is not installed.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        raise ImportError(
            "auto-gptq is not installed. "
            "Install with: pip install 'pulsar-ai[gptq]'"
        )

    from transformers import AutoTokenizer

    try:
        logger.info(
            "Loading model for GPTQ quantization: %s", model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quantize_config = BaseQuantizeConfig(
            bits=gptq_config["bits"],
            group_size=gptq_config["group_size"],
            desc_act=gptq_config["desc_act"],
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            model_path, quantize_config=quantize_config
        )

        # Tokenize calibration data
        examples = [
            tokenizer(text, return_tensors="pt")
            for text in calibration_data
        ]

        logger.info(
            "Quantizing with %d-bit, group_size=%d, "
            "desc_act=%s (%d calibration samples)",
            gptq_config["bits"],
            gptq_config["group_size"],
            gptq_config["desc_act"],
            len(examples),
        )
        model.quantize(examples)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info("GPTQ model saved to: %s", output_path)
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
