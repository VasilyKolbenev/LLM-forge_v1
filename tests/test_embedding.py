"""Tests for embedding training module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from pulsar_ai.training.embedding import (
    validate_embedding_dataset,
    train_embedding,
)


class TestValidateEmbeddingDataset:
    """Tests for validate_embedding_dataset."""

    def test_valid_triplet(self) -> None:
        df = pd.DataFrame({
            "anchor": ["a", "b"],
            "positive": ["c", "d"],
            "negative": ["e", "f"],
        })
        errors = validate_embedding_dataset(df, schema="triplet")
        assert errors == []

    def test_valid_pair(self) -> None:
        df = pd.DataFrame({
            "sentence1": ["a", "b"],
            "sentence2": ["c", "d"],
            "label": [0.8, 0.2],
        })
        errors = validate_embedding_dataset(df, schema="pair")
        assert errors == []

    def test_missing_triplet_columns(self) -> None:
        df = pd.DataFrame({"anchor": ["a"], "positive": ["b"]})
        errors = validate_embedding_dataset(df, schema="triplet")
        assert len(errors) == 1
        assert "negative" in errors[0]

    def test_missing_pair_columns(self) -> None:
        df = pd.DataFrame({"sentence1": ["a"]})
        errors = validate_embedding_dataset(df, schema="pair")
        assert len(errors) == 1
        assert "sentence2" in errors[0] or "label" in errors[0]

    def test_unknown_schema(self) -> None:
        df = pd.DataFrame({"a": [1]})
        errors = validate_embedding_dataset(df, schema="unknown")
        assert len(errors) == 1
        assert "Unknown schema" in errors[0]

    def test_empty_dataframe_triplet(self) -> None:
        df = pd.DataFrame({"anchor": [], "positive": [], "negative": []})
        errors = validate_embedding_dataset(df, schema="triplet")
        assert errors == []


class TestTrainEmbedding:
    """Tests for train_embedding with mocked internals."""

    @pytest.fixture()
    def mock_results(self) -> dict:
        return {
            "training_loss": 0.35,
            "global_steps": 200,
            "output_dir": "./outputs/embedding",
            "artifact_type": "full_model",
        }

    @pytest.fixture()
    def base_config(self) -> dict:
        return {
            "model": {"name": "test-embedding-model"},
            "training": {"epochs": 1},
            "embedding": {"schema": "triplet"},
            "dataset": {"path": "data.jsonl"},
        }

    def test_train_embedding_returns_results(
        self, base_config: dict, mock_results: dict
    ) -> None:
        mock_tracker = MagicMock()
        mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
        mock_tracker.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "pulsar_ai.tracking.track_experiment",
                return_value=mock_tracker,
            ),
            patch(
                "pulsar_ai.training.embedding._run_embedding_training",
                return_value=mock_results,
            ),
        ):
            results = train_embedding(base_config)

        assert results["training_loss"] == 0.35
        assert results["global_steps"] == 200
        assert results["artifact_type"] == "full_model"
        mock_tracker.log_metrics.assert_called_once()
        mock_tracker.log_artifact.assert_called_once_with(
            "full_model", "./outputs/embedding"
        )

    def test_train_embedding_tracker_receives_metrics(
        self, base_config: dict, mock_results: dict
    ) -> None:
        mock_tracker = MagicMock()
        mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
        mock_tracker.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "pulsar_ai.tracking.track_experiment",
                return_value=mock_tracker,
            ),
            patch(
                "pulsar_ai.training.embedding._run_embedding_training",
                return_value=mock_results,
            ),
        ):
            train_embedding(base_config)

        logged = mock_tracker.log_metrics.call_args[0][0]
        assert "training_loss" in logged
        assert "global_steps" in logged


class TestEmbeddingImportGuard:
    """Test that the module is importable without sentence-transformers."""

    def test_module_importable(self) -> None:
        import pulsar_ai.training.embedding as mod
        assert hasattr(mod, "train_embedding")
        assert hasattr(mod, "validate_embedding_dataset")
