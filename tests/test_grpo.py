"""Tests for GRPO training module: rewards, registry, trainer, dataset."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from pulsar_ai.training.grpo import (
    reward_format_compliance,
    reward_length_penalty,
    reward_keyword_match,
    REWARD_REGISTRY,
    get_reward_function,
    _load_custom_reward,
    _load_grpo_dataset,
    train_grpo,
)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


class TestRewardFunctions:
    """Tests for built-in reward functions."""

    def test_format_compliance_valid_json(self) -> None:
        completions = ['{"key": "value"}', "[1, 2, 3]", '"hello"']
        scores = reward_format_compliance(completions, [""] * 3)
        assert scores == [1.0, 1.0, 1.0]

    def test_format_compliance_invalid_json(self) -> None:
        completions = ["not json", "{bad:", ""]
        scores = reward_format_compliance(completions, [""] * 3)
        assert scores == [0.0, 0.0, 0.0]

    def test_format_compliance_mixed(self) -> None:
        completions = ['{"ok": true}', "nope"]
        scores = reward_format_compliance(completions, [""] * 2)
        assert scores == [1.0, 0.0]

    def test_format_compliance_unknown_format(self) -> None:
        scores = reward_format_compliance(
            ["hello"], [""], expected_format="xml"
        )
        assert scores == [0.0]

    def test_length_penalty_within_range(self) -> None:
        text = "a" * 50
        scores = reward_length_penalty(
            [text], [""], min_length=10, max_length=500
        )
        assert scores == [1.0]

    def test_length_penalty_too_short(self) -> None:
        scores = reward_length_penalty(
            ["hi"], [""], min_length=10, max_length=500
        )
        assert len(scores) == 1
        assert 0.0 < scores[0] < 1.0

    def test_length_penalty_too_long(self) -> None:
        text = "a" * 1000
        scores = reward_length_penalty(
            [text], [""], min_length=10, max_length=500
        )
        assert len(scores) == 1
        assert 0.0 < scores[0] < 1.0

    def test_length_penalty_empty(self) -> None:
        scores = reward_length_penalty(
            [""], [""], min_length=10, max_length=500
        )
        assert scores == [0.0]

    def test_length_penalty_exact_min(self) -> None:
        text = "a" * 10
        scores = reward_length_penalty(
            [text], [""], min_length=10, max_length=500
        )
        assert scores == [1.0]

    def test_length_penalty_exact_max(self) -> None:
        text = "a" * 500
        scores = reward_length_penalty(
            [text], [""], min_length=10, max_length=500
        )
        assert scores == [1.0]

    def test_keyword_match_all_present(self) -> None:
        scores = reward_keyword_match(
            ["hello world foo"], [""], keywords=["hello", "world"]
        )
        assert scores == [1.0]

    def test_keyword_match_partial(self) -> None:
        scores = reward_keyword_match(
            ["hello only"], [""], keywords=["hello", "world"]
        )
        assert scores == [0.5]

    def test_keyword_match_none_present(self) -> None:
        scores = reward_keyword_match(
            ["nothing here"], [""], keywords=["hello", "world"]
        )
        assert scores == [0.0]

    def test_keyword_match_no_keywords(self) -> None:
        scores = reward_keyword_match(["text"], [""], keywords=None)
        assert scores == [0.0]

    def test_keyword_match_case_insensitive(self) -> None:
        scores = reward_keyword_match(
            ["HELLO World"], [""], keywords=["hello", "world"]
        )
        assert scores == [1.0]


# ---------------------------------------------------------------------------
# Reward registry
# ---------------------------------------------------------------------------


class TestRewardRegistry:
    """Tests for the reward function registry."""

    def test_get_builtin(self) -> None:
        fn = get_reward_function("format_compliance")
        assert fn is reward_format_compliance

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reward function"):
            get_reward_function("nonexistent_reward")

    def test_all_builtins_registered(self) -> None:
        expected = {"format_compliance", "length_penalty", "keyword_match"}
        assert set(REWARD_REGISTRY.keys()) == expected

    def test_get_each_builtin(self) -> None:
        for name in REWARD_REGISTRY:
            fn = get_reward_function(name)
            assert callable(fn)


# ---------------------------------------------------------------------------
# Custom reward loading
# ---------------------------------------------------------------------------


class TestCustomReward:
    """Tests for custom reward file loading."""

    def test_traversal_blocked(self, tmp_path: Path) -> None:
        bad_path = tmp_path / ".." / "evil.py"
        with pytest.raises(ValueError, match="traversal"):
            _load_custom_reward(bad_path)

    def test_missing_reward_fn_raises(self, tmp_path: Path) -> None:
        py_file = tmp_path / "bad_reward.py"
        py_file.write_text("x = 1\n")
        with pytest.raises(AttributeError, match="reward_fn"):
            _load_custom_reward(py_file)

    def test_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.py"
        with pytest.raises(FileNotFoundError):
            _load_custom_reward(missing)

    def test_valid_custom_reward(self, tmp_path: Path) -> None:
        py_file = tmp_path / "my_reward.py"
        py_file.write_text(
            "def reward_fn(completions, prompts, **kw):\n"
            "    return [1.0] * len(completions)\n"
        )
        fn = _load_custom_reward(py_file)
        assert fn(["a", "b"], ["", ""]) == [1.0, 1.0]

    def test_get_reward_function_with_py_path(self, tmp_path: Path) -> None:
        py_file = tmp_path / "custom.py"
        py_file.write_text(
            "def reward_fn(completions, prompts, **kw):\n"
            "    return [0.5] * len(completions)\n"
        )
        fn = get_reward_function(str(py_file))
        assert fn(["x"], ["p"]) == [0.5]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestGrpoDataset:
    """Tests for _load_grpo_dataset."""

    def test_load_with_prompt_column(self, tmp_path: Path) -> None:
        ds_file = tmp_path / "prompts.jsonl"
        lines = [
            json.dumps({"prompt": "Say hello"}),
            json.dumps({"prompt": "Say bye"}),
        ]
        ds_file.write_text("\n".join(lines))

        config = {"dataset": {"path": str(ds_file)}}
        ds = _load_grpo_dataset(config)
        assert len(ds) == 2
        assert "prompt" in ds.column_names

    def test_load_with_text_column(self, tmp_path: Path) -> None:
        ds_file = tmp_path / "texts.jsonl"
        lines = [json.dumps({"text": "Do something"})]
        ds_file.write_text("\n".join(lines))

        config = {"dataset": {"path": str(ds_file)}}
        ds = _load_grpo_dataset(config)
        assert len(ds) == 1
        assert "prompt" in ds.column_names

    def test_load_with_instruction_column(self, tmp_path: Path) -> None:
        ds_file = tmp_path / "instr.jsonl"
        lines = [json.dumps({"instruction": "Explain X"})]
        ds_file.write_text("\n".join(lines))

        config = {"dataset": {"path": str(ds_file)}}
        ds = _load_grpo_dataset(config)
        assert "prompt" in ds.column_names

    def test_load_with_input_column(self, tmp_path: Path) -> None:
        ds_file = tmp_path / "inputs.jsonl"
        lines = [json.dumps({"input": "What is 2+2?"})]
        ds_file.write_text("\n".join(lines))

        config = {"dataset": {"path": str(ds_file)}}
        ds = _load_grpo_dataset(config)
        assert "prompt" in ds.column_names

    def test_no_prompt_column_raises(self, tmp_path: Path) -> None:
        ds_file = tmp_path / "bad.jsonl"
        lines = [json.dumps({"answer": "42"})]
        ds_file.write_text("\n".join(lines))

        config = {"dataset": {"path": str(ds_file)}}
        with pytest.raises(ValueError, match="No prompt column"):
            _load_grpo_dataset(config)

    def test_missing_path_raises(self) -> None:
        config: dict = {"dataset": {}}
        with pytest.raises(ValueError, match="dataset.path is required"):
            _load_grpo_dataset(config)


# ---------------------------------------------------------------------------
# train_grpo integration (mocked)
# ---------------------------------------------------------------------------


class TestTrainGrpo:
    """Tests for the train_grpo entry point with mocked internals."""

    @pytest.fixture()
    def mock_results(self) -> dict:
        return {
            "training_loss": 0.42,
            "global_steps": 100,
            "reward_mean": 0.75,
            "reward_std": 0.12,
            "vram_peak_gb": 8.5,
            "output_dir": "./outputs/grpo",
            "adapter_dir": "./outputs/grpo/grpo_model",
        }

    @pytest.fixture()
    def base_config(self) -> dict:
        return {
            "model": {"name": "test-model"},
            "training": {"epochs": 1},
            "grpo": {"reward_function": "format_compliance"},
            "dataset": {"path": "data.jsonl"},
        }

    def test_train_grpo_returns_results(
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
                "pulsar_ai.training.grpo._run_grpo_trl",
                return_value=mock_results,
            ),
        ):
            results = train_grpo(base_config)

        assert results["training_loss"] == 0.42
        assert results["reward_mean"] == 0.75
        assert results["global_steps"] == 100
        mock_tracker.log_metrics.assert_called_once()
        mock_tracker.log_artifact.assert_called_once()
        mock_tracker.finish.assert_called_once()

    def test_train_grpo_accepts_progress(
        self, base_config: dict, mock_results: dict
    ) -> None:
        mock_tracker = MagicMock()
        mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
        mock_tracker.__exit__ = MagicMock(return_value=False)
        mock_progress = MagicMock()
        mock_callback = MagicMock()

        with (
            patch(
                "pulsar_ai.tracking.track_experiment",
                return_value=mock_tracker,
            ),
            patch(
                "pulsar_ai.training.grpo._run_grpo_trl",
                return_value=mock_results,
            ) as mock_run,
            patch(
                "pulsar_ai.ui.progress.make_hf_callback",
                return_value=mock_callback,
            ),
        ):
            results = train_grpo(base_config, progress=mock_progress)

        assert results is not None
        # Verify callback was passed to _run_grpo_trl
        call_args = mock_run.call_args
        assert mock_callback in call_args[1].get(
            "callbacks", call_args[0][2] if len(call_args[0]) > 2 else []
        )

    def test_train_grpo_invalid_reward_raises(
        self, base_config: dict
    ) -> None:
        base_config["grpo"]["reward_function"] = "nonexistent"
        with pytest.raises(ValueError, match="Unknown reward function"):
            train_grpo(base_config)

    def test_train_grpo_tracker_receives_metrics(
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
                "pulsar_ai.training.grpo._run_grpo_trl",
                return_value=mock_results,
            ),
        ):
            train_grpo(base_config)

        logged = mock_tracker.log_metrics.call_args[0][0]
        assert "training_loss" in logged
        assert "reward_mean" in logged
        assert "reward_std" in logged
        assert "vram_peak_gb" in logged
