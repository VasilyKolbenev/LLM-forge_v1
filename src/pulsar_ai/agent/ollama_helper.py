"""Ollama integration helpers for model management."""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"


def get_ollama_status() -> dict[str, Any]:
    """Return Ollama server status and available models."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [
            {
                "name": m["name"],
                "size_gb": round(m.get("size", 0) / 1e9, 1),
                "modified_at": m.get("modified_at", ""),
                "family": m.get("details", {}).get("family", "unknown"),
            }
            for m in data.get("models", [])
        ]
        return {"connected": True, "models": models}
    except Exception as e:
        logger.warning("Ollama not available: %s", e)
        return {"connected": False, "models": [], "error": str(e)}


def ensure_ollama_model(model: str = "qwen3.5:4b") -> bool:
    """Check if model is available, pull if not.

    Args:
        model: Ollama model tag to ensure is available.

    Returns:
        True when the model is ready for use.
    """
    status = get_ollama_status()
    if not status["connected"]:
        return False

    model_names = [m["name"] for m in status["models"]]
    # Check exact match or prefix match (e.g., "qwen3.5:4b" matches "qwen3.5:4b-instruct")
    if any(
        model in name or name.startswith(model.split(":")[0])
        for name in model_names
    ):
        return True

    try:
        logger.info("Pulling model %s from Ollama...", model)
        resp = requests.post(
            f"{OLLAMA_BASE}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,  # 10 min for large models
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Failed to pull model %s: %s", model, e)
        return False


def list_ollama_models() -> list[dict[str, Any]]:
    """Return list of available Ollama models."""
    status = get_ollama_status()
    return status.get("models", [])
