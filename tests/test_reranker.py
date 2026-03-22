"""Tests for reranker training module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from pulsar_ai.training.reranker import (
    validate_reranker_dataset,
    train_reranker,
)


class TestValidateRerankerDataset:
    """Tests for validate_reranker_dataset."""

    def test_valid_dataset(self) -> None:
        df = pd.DataFrame({
            "query": ["what is AI"],
            "document": ["AI is artificial intelligence"],
            "relevance_score": [0.9],
        })
        errors = validate_reranker_dataset(df)
        assert errors == []

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"query": ["q1"], "document": ["d1"]})
        errors = validate_reranker_dataset(df)
        assert len(errors) == 1
        assert "relevance_score" in errors[0]

    def test_all_columns_missing(self) -> None:
        df = pd.DataFrame({"text": ["hello"]})
        errors = validate_reranker_dataset(df)
        assert len(errors) == 1
        assert "query" in errors[0]

    def test_empty_dataframe_valid_columns(self) -> None:
        df = pd.DataFrame({
            "query": [],
            "document": [],
            "relevance_score": [],
        })
        errors = validate_reranker_dataset(df)
        assert errors == []


class TestTrainReranker:
    """Tests for train_reranker with mocked internals."""

    @pytest.fixture()
    def mock_results(self) -> dict:
        return {
            "output_dir": "./outputs/reranker",
            "artifact_type": "full_model",
            "num_samples": 100,
        }

    @pytest.fixture()
    def base_config(self) -> dict:
        return {
            "model": {"name": "test-cross-encoder"},
            "training": {"epochs": 1},
            "dataset": {"path": "data.jsonl"},
        }

    def test_train_reranker_returns_results(
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
                "pulsar_ai.training.reranker._run_reranker_training",
                return_value=mock_results,
            ),
        ):
            results = train_reranker(base_config)

        assert results["output_dir"] == "./outputs/reranker"
        assert results["artifact_type"] == "full_model"
        assert results["num_samples"] == 100
        mock_tracker.log_metrics.assert_called_once()
        mock_tracker.log_artifact.assert_called_once_with(
            "full_model", "./outputs/reranker"
        )

    def test_train_reranker_tracker_receives_metrics(
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
                "pulsar_ai.training.reranker._run_reranker_training",
                return_value=mock_results,
            ),
        ):
            train_reranker(base_config)

        logged = mock_tracker.log_metrics.call_args[0][0]
        assert "num_samples" in logged
