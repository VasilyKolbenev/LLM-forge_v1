"""Edge-case tests for all P0 modules: GRPO, embedding, export, sglang, recipes."""

from pathlib import Path

import pandas as pd
import pytest

from pulsar_ai.training.grpo import (
    reward_format_compliance,
    _load_custom_reward,
)
from pulsar_ai.training.embedding import validate_embedding_dataset
from pulsar_ai.export.gptq import _get_gptq_config, export_gptq
from pulsar_ai.export.awq import export_awq
from pulsar_ai.serving.sglang import start_server
from pulsar_ai.recipes import RecipeRegistry


# ---------------------------------------------------------------------------
# GRPO edge cases
# ---------------------------------------------------------------------------


class TestGrpoEdgeCases:
    """Edge cases for GRPO reward functions and custom reward loading."""

    def test_empty_completions_list(self) -> None:
        scores = reward_format_compliance([], [])
        assert scores == []

    def test_custom_reward_traversal_blocked(self) -> None:
        with pytest.raises(ValueError, match="traversal"):
            _load_custom_reward(Path("../../etc/evil.py"))

    def test_custom_reward_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_custom_reward(Path("/nonexistent/reward.py"))


# ---------------------------------------------------------------------------
# Embedding edge cases
# ---------------------------------------------------------------------------


class TestEmbeddingEdgeCases:
    """Edge cases for embedding dataset validation."""

    def test_empty_dataframe_validation(self) -> None:
        df = pd.DataFrame()
        errors = validate_embedding_dataset(df, schema="triplet")
        assert len(errors) > 0
        assert any("Missing" in e for e in errors)

    def test_unknown_schema_validation(self) -> None:
        df = pd.DataFrame({"col": [1]})
        errors = validate_embedding_dataset(df, schema="unknown_schema")
        assert len(errors) > 0
        assert any("Unknown schema" in e for e in errors)


# ---------------------------------------------------------------------------
# Export edge cases
# ---------------------------------------------------------------------------


class TestExportEdgeCases:
    """Edge cases for GPTQ/AWQ export."""

    def test_gptq_invalid_bits_5(self) -> None:
        with pytest.raises(ValueError, match="Invalid GPTQ bits"):
            _get_gptq_config({"bits": 5})

    def test_gptq_invalid_group_size_100(self) -> None:
        with pytest.raises(ValueError, match="Invalid GPTQ group_size"):
            _get_gptq_config({"group_size": 100})

    def test_awq_missing_model_path(self) -> None:
        with pytest.raises(ValueError, match="model_path is required"):
            export_awq({})

    def test_gptq_missing_model_path(self) -> None:
        with pytest.raises(ValueError, match="model_path is required"):
            export_gptq({})


# ---------------------------------------------------------------------------
# SGLang edge cases
# ---------------------------------------------------------------------------


class TestSglangEdgeCases:
    """Edge cases for SGLang serving module."""

    def test_sglang_rejects_gguf(self) -> None:
        with pytest.raises(ValueError, match="GGUF"):
            start_server("model.gguf")


# ---------------------------------------------------------------------------
# Recipe edge cases
# ---------------------------------------------------------------------------


class TestRecipeEdgeCases:
    """Edge cases for recipe registry."""

    def test_empty_recipes_dir(self, tmp_path: Path) -> None:
        registry = RecipeRegistry(tmp_path)
        assert registry.list_recipes() == []

    def test_malformed_recipe_skipped(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(": : : invalid yaml {{{}}", encoding="utf-8")
        registry = RecipeRegistry(tmp_path)
        recipes = registry.list_recipes()
        assert recipes == []

    def test_nonexistent_recipes_dir(self, tmp_path: Path) -> None:
        fake_dir = tmp_path / "does_not_exist"
        registry = RecipeRegistry(fake_dir)
        assert registry.list_recipes() == []
