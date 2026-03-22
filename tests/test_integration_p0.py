"""Integration tests for P0 features: GRPO, embedding, reranker,
classification, AWQ/GPTQ export, and recipe hub."""

from pathlib import Path

import pandas as pd
import pytest

yaml = pytest.importorskip("yaml")

from pulsar_ai.validation import validate_config
from pulsar_ai.export.awq import _get_awq_config
from pulsar_ai.export.gptq import _get_gptq_config
from pulsar_ai.recipes import RecipeRegistry


# ---------------------------------------------------------------------------
# GRPO integration
# ---------------------------------------------------------------------------


class TestGrpoIntegration:
    """Cross-module validation for GRPO configs."""

    def test_grpo_config_validates(self) -> None:
        config = {
            "task": "grpo",
            "model": {"name": "test-model"},
            "dataset": {"path": "/tmp/fake.jsonl"},
            "grpo": {"num_generations": 4, "max_completion_length": 256},
        }
        errors = validate_config(config, task="grpo")
        assert errors == []

    def test_grpo_invalid_num_generations(self) -> None:
        config = {
            "task": "grpo",
            "model": {"name": "test-model"},
            "dataset": {"path": "/tmp/fake.jsonl"},
            "grpo": {"num_generations": -1},
        }
        errors = validate_config(config, task="grpo")
        assert any("num_generations" in e for e in errors)

    def test_grpo_invalid_max_completion_length(self) -> None:
        config = {
            "task": "grpo",
            "model": {"name": "test-model"},
            "dataset": {"path": "/tmp/fake.jsonl"},
            "grpo": {"max_completion_length": 0},
        }
        errors = validate_config(config, task="grpo")
        assert any("max_completion_length" in e for e in errors)


# ---------------------------------------------------------------------------
# Embedding / Reranker / Classification integration
# ---------------------------------------------------------------------------


class TestEmbeddingIntegration:
    """Cross-module validation for embedding, reranker, classification."""

    def test_embedding_config_validates(self) -> None:
        config = {
            "task": "embedding",
            "model": {"name": "test-model"},
            "dataset": {"path": "/tmp/fake.jsonl"},
        }
        errors = validate_config(config, task="embedding")
        assert errors == []

    def test_reranker_config_validates(self) -> None:
        config = {
            "task": "reranker",
            "model": {"name": "test-model"},
            "dataset": {"path": "/tmp/fake.jsonl"},
        }
        errors = validate_config(config, task="reranker")
        assert errors == []

    def test_classification_config_validates(self) -> None:
        config = {
            "task": "classification",
            "model": {"name": "test-model"},
            "dataset": {"path": "/tmp/fake.jsonl"},
        }
        errors = validate_config(config, task="classification")
        assert errors == []


# ---------------------------------------------------------------------------
# Export integration
# ---------------------------------------------------------------------------


class TestExportIntegration:
    """AWQ/GPTQ config parsing integration tests."""

    def test_awq_config_parses_w4_g128(self) -> None:
        result = _get_awq_config({"quant_config": "w4-g128"})
        assert result["w_bit"] == 4
        assert result["q_group_size"] == 128

    def test_awq_config_parses_w4_g64(self) -> None:
        result = _get_awq_config({"quant_config": "w4-g64"})
        assert result["w_bit"] == 4
        assert result["q_group_size"] == 64

    def test_gptq_config_parses(self) -> None:
        result = _get_gptq_config({"bits": 8, "group_size": 32})
        assert result["bits"] == 8
        assert result["group_size"] == 32

    def test_gptq_invalid_bits(self) -> None:
        with pytest.raises(ValueError, match="Invalid GPTQ bits"):
            _get_gptq_config({"bits": 5})

    def test_gptq_invalid_group_size(self) -> None:
        with pytest.raises(ValueError, match="Invalid GPTQ group_size"):
            _get_gptq_config({"group_size": 100})


# ---------------------------------------------------------------------------
# Recipe integration
# ---------------------------------------------------------------------------


_RECIPES_DIR = Path(__file__).parent.parent / "configs" / "recipes"


class TestRecipeIntegration:
    """Recipe hub integration tests against real shipped recipes."""

    def test_recipe_registry_finds_shipped_recipes(self) -> None:
        registry = RecipeRegistry(_RECIPES_DIR)
        recipes = registry.list_recipes()
        assert len(recipes) >= 15

    def test_all_recipes_have_task_field(self) -> None:
        for path in sorted(_RECIPES_DIR.glob("*.yaml")):
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            assert isinstance(data, dict), f"{path.name} is not a dict"
            assert "task" in data, f"{path.name} missing 'task' key"
