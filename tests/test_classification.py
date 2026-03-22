"""Tests for classification training module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from pulsar_ai.training.classification import (
    detect_labels,
    validate_classification_dataset,
    train_classification,
)


class TestDetectLabels:
    """Tests for detect_labels."""

    def test_correct_mapping(self) -> None:
        df = pd.DataFrame({"label": ["cat", "dog", "cat", "bird"]})
        label_map = detect_labels(df, "label")
        assert label_map == {"bird": 0, "cat": 1, "dog": 2}

    def test_single_label(self) -> None:
        df = pd.DataFrame({"label": ["positive", "positive"]})
        label_map = detect_labels(df, "label")
        assert label_map == {"positive": 0}

    def test_numeric_labels_converted_to_str(self) -> None:
        df = pd.DataFrame({"label": [0, 1, 2, 1]})
        label_map = detect_labels(df, "label")
        assert label_map == {"0": 0, "1": 1, "2": 2}

    def test_custom_column_name(self) -> None:
        df = pd.DataFrame({"sentiment": ["pos", "neg"]})
        label_map = detect_labels(df, "sentiment")
        assert label_map == {"neg": 0, "pos": 1}


class TestValidateClassificationDataset:
    """Tests for validate_classification_dataset."""

    def test_valid_dataset(self) -> None:
        df = pd.DataFrame({
            "text": ["hello", "world"],
            "label": ["pos", "neg"],
        })
        errors = validate_classification_dataset(df, "text", "label")
        assert errors == []

    def test_missing_text_column(self) -> None:
        df = pd.DataFrame({"label": ["pos"]})
        errors = validate_classification_dataset(df, "text", "label")
        assert len(errors) == 1
        assert "text" in errors[0]

    def test_missing_label_column(self) -> None:
        df = pd.DataFrame({"text": ["hello"]})
        errors = validate_classification_dataset(df, "text", "label")
        assert len(errors) == 1
        assert "label" in errors[0]

    def test_missing_both_columns(self) -> None:
        df = pd.DataFrame({"other": [1]})
        errors = validate_classification_dataset(df, "text", "label")
        assert len(errors) == 1
        assert "text" in errors[0] or "label" in errors[0]

    def test_custom_column_names(self) -> None:
        df = pd.DataFrame({
            "review": ["great"],
            "sentiment": ["positive"],
        })
        errors = validate_classification_dataset(
            df, "review", "sentiment"
        )
        assert errors == []


class TestTrainClassification:
    """Tests for train_classification with mocked internals."""

    @pytest.fixture()
    def mock_results(self) -> dict:
        return {
            "training_loss": 0.25,
            "global_steps": 150,
            "output_dir": "./outputs/classification",
            "artifact_type": "full_model",
            "num_labels": 3,
            "label_map": {"bird": 0, "cat": 1, "dog": 2},
        }

    @pytest.fixture()
    def base_config(self) -> dict:
        return {
            "model": {"name": "test-bert"},
            "training": {"epochs": 1},
            "classification": {
                "text_column": "text",
                "label_column": "label",
            },
            "dataset": {"path": "data.csv"},
        }

    def test_train_classification_returns_results(
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
                "pulsar_ai.training.classification._run_classification_training",
                return_value=mock_results,
            ),
        ):
            results = train_classification(base_config)

        assert results["training_loss"] == 0.25
        assert results["global_steps"] == 150
        assert results["artifact_type"] == "full_model"
        assert results["num_labels"] == 3
        assert results["label_map"] == {"bird": 0, "cat": 1, "dog": 2}
        mock_tracker.log_metrics.assert_called_once()
        mock_tracker.log_artifact.assert_called_once_with(
            "full_model", "./outputs/classification"
        )

    def test_train_classification_tracker_receives_metrics(
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
                "pulsar_ai.training.classification._run_classification_training",
                return_value=mock_results,
            ),
        ):
            train_classification(base_config)

        logged = mock_tracker.log_metrics.call_args[0][0]
        assert "training_loss" in logged
        assert "global_steps" in logged
        assert "num_labels" in logged
