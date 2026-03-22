"""Regression tests: ensure existing features still work after P0 additions."""

import inspect

import pytest

from pulsar_ai.validation import validate_config


# ---------------------------------------------------------------------------
# Trainer function signatures
# ---------------------------------------------------------------------------


class TestTrainerSignatures:
    """Verify all trainer entry points keep their (config, progress) signature."""

    def test_train_sft_signature(self) -> None:
        from pulsar_ai.training.sft import train_sft

        sig = inspect.signature(train_sft)
        assert "config" in sig.parameters
        assert "progress" in sig.parameters

    def test_train_dpo_signature(self) -> None:
        from pulsar_ai.training.dpo import train_dpo

        sig = inspect.signature(train_dpo)
        assert "config" in sig.parameters
        assert "progress" in sig.parameters

    def test_train_grpo_signature(self) -> None:
        from pulsar_ai.training.grpo import train_grpo

        sig = inspect.signature(train_grpo)
        assert "config" in sig.parameters
        assert "progress" in sig.parameters

    def test_train_embedding_signature(self) -> None:
        from pulsar_ai.training.embedding import train_embedding

        sig = inspect.signature(train_embedding)
        assert "config" in sig.parameters
        assert "progress" in sig.parameters

    def test_train_reranker_signature(self) -> None:
        from pulsar_ai.training.reranker import train_reranker

        sig = inspect.signature(train_reranker)
        assert "config" in sig.parameters
        assert "progress" in sig.parameters

    def test_train_classification_signature(self) -> None:
        from pulsar_ai.training.classification import train_classification

        sig = inspect.signature(train_classification)
        assert "config" in sig.parameters
        assert "progress" in sig.parameters


# ---------------------------------------------------------------------------
# Config backward compatibility
# ---------------------------------------------------------------------------


class TestConfigBackwardCompat:
    """Ensure validate_config still works for original task types."""

    def test_sft_validation_still_works(self) -> None:
        config = {
            "task": "sft",
            "model": {"name": "test-model"},
            "dataset": {"path": "/tmp/fake.jsonl"},
        }
        errors = validate_config(config, task="sft")
        assert errors == []

    def test_dpo_validation_still_works(self) -> None:
        config = {
            "task": "dpo",
            "model": {"name": "test-model"},
            "sft_adapter_path": "/tmp/adapter",
            "dpo": {"pairs_path": "/tmp/pairs.jsonl"},
        }
        errors = validate_config(config, task="dpo")
        assert errors == []

    def test_unknown_task_no_crash(self) -> None:
        config = {
            "task": "nonexistent_task_xyz",
            "model": {"name": "test-model"},
        }
        # Should not raise; unknown tasks just have no required keys
        errors = validate_config(config, task="nonexistent_task_xyz")
        assert isinstance(errors, list)


# ---------------------------------------------------------------------------
# Export backward compatibility
# ---------------------------------------------------------------------------


class TestExportBackwardCompat:
    """Ensure all export functions remain importable."""

    def test_export_gguf_importable(self) -> None:
        from pulsar_ai.export.gguf import export_gguf

        assert callable(export_gguf)

    def test_export_merged_importable(self) -> None:
        from pulsar_ai.export.merged import export_merged

        assert callable(export_merged)

    def test_export_awq_importable(self) -> None:
        from pulsar_ai.export.awq import export_awq

        assert callable(export_awq)

    def test_export_gptq_importable(self) -> None:
        from pulsar_ai.export.gptq import export_gptq

        assert callable(export_gptq)
