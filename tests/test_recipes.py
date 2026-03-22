"""Tests for the RecipeRegistry: scan, filter, load, and validation."""

from pathlib import Path

import pytest
import yaml

from pulsar_ai.recipes import RecipeRegistry

_SHIPPED_RECIPES_DIR = (
    Path(__file__).parent.parent / "configs" / "recipes"
)


@pytest.fixture()
def recipes_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample recipe files."""
    recipe_a = {
        "meta": {
            "name": "Alpha",
            "task_type": "sft",
            "difficulty": "beginner",
            "hardware": "1x 24GB",
            "tags": ["llama", "instruct"],
            "description": "Alpha recipe",
        },
        "task": "sft",
        "model": {"name": "test-model"},
        "training": {"epochs": 3},
        "output": {"dir": "./outputs/alpha"},
    }
    recipe_b = {
        "meta": {
            "name": "Beta",
            "task_type": "dpo",
            "difficulty": "intermediate",
            "hardware": "1x 24GB",
            "tags": ["dpo", "instruct"],
            "description": "Beta recipe",
        },
        "task": "dpo",
        "model": {"name": "test-model-dpo"},
        "training": {"epochs": 2},
        "output": {"dir": "./outputs/beta"},
    }
    with open(tmp_path / "alpha.yaml", "w") as f:
        yaml.dump(recipe_a, f)
    with open(tmp_path / "beta.yaml", "w") as f:
        yaml.dump(recipe_b, f)
    return tmp_path


def test_scan_finds_recipes(recipes_dir: Path) -> None:
    """Registry discovers all valid recipes in directory."""
    registry = RecipeRegistry(recipes_dir=recipes_dir)
    items = registry.list_recipes()
    assert len(items) == 2
    names = {r["name"] for r in items}
    assert names == {"Alpha", "Beta"}


def test_filter_by_task(recipes_dir: Path) -> None:
    """Filtering by task_type returns only matching recipes."""
    registry = RecipeRegistry(recipes_dir=recipes_dir)
    sft_only = registry.list_recipes(task_type="sft")
    assert len(sft_only) == 1
    assert sft_only[0]["name"] == "Alpha"


def test_filter_by_tag(recipes_dir: Path) -> None:
    """Filtering by tag returns recipes containing that tag."""
    registry = RecipeRegistry(recipes_dir=recipes_dir)
    dpo_tag = registry.list_recipes(tag="dpo")
    assert len(dpo_tag) == 1
    assert dpo_tag[0]["name"] == "Beta"

    instruct_tag = registry.list_recipes(tag="instruct")
    assert len(instruct_tag) == 2


def test_load_recipe(recipes_dir: Path) -> None:
    """load_recipe returns config with meta stripped."""
    registry = RecipeRegistry(recipes_dir=recipes_dir)
    config = registry.load_recipe("alpha")
    assert "meta" not in config
    assert config["task"] == "sft"
    assert config["model"]["name"] == "test-model"


def test_load_nonexistent_raises(recipes_dir: Path) -> None:
    """Loading a missing recipe raises FileNotFoundError."""
    registry = RecipeRegistry(recipes_dir=recipes_dir)
    with pytest.raises(FileNotFoundError):
        registry.load_recipe("nonexistent")


def test_empty_dir(tmp_path: Path) -> None:
    """Empty directory returns empty list, no crash."""
    registry = RecipeRegistry(recipes_dir=tmp_path)
    assert registry.list_recipes() == []


def test_malformed_yaml_skipped(tmp_path: Path) -> None:
    """Malformed YAML files are skipped without raising."""
    (tmp_path / "broken.yaml").write_text(": : : [invalid")
    good = {
        "meta": {
            "name": "Good",
            "task_type": "sft",
            "difficulty": "beginner",
            "hardware": "1x 24GB",
            "tags": [],
            "description": "Good recipe",
        },
        "task": "sft",
    }
    with open(tmp_path / "good.yaml", "w") as f:
        yaml.dump(good, f)

    registry = RecipeRegistry(recipes_dir=tmp_path)
    items = registry.list_recipes()
    assert len(items) == 1
    assert items[0]["name"] == "Good"


def test_missing_dir_returns_empty() -> None:
    """Non-existent directory returns empty list."""
    registry = RecipeRegistry(
        recipes_dir=Path("/nonexistent/path/recipes")
    )
    assert registry.list_recipes() == []


def test_filter_by_difficulty(recipes_dir: Path) -> None:
    """Filtering by difficulty returns only matching recipes."""
    registry = RecipeRegistry(recipes_dir=recipes_dir)
    beginners = registry.list_recipes(difficulty="beginner")
    assert len(beginners) == 1
    assert beginners[0]["name"] == "Alpha"


@pytest.mark.skipif(
    not _SHIPPED_RECIPES_DIR.is_dir(),
    reason="Shipped recipes dir not found",
)
def test_all_shipped_recipes_valid() -> None:
    """All YAML files in configs/recipes/ have a valid meta block."""
    registry = RecipeRegistry(recipes_dir=_SHIPPED_RECIPES_DIR)
    items = registry.list_recipes()
    assert len(items) >= 15, (
        f"Expected at least 15 shipped recipes, found {len(items)}"
    )
    required_keys = {"name", "task_type", "difficulty", "hardware"}
    for recipe in items:
        missing = required_keys - set(recipe.keys())
        assert not missing, (
            f"Recipe '{recipe.get('name', '?')}' "
            f"missing meta keys: {missing}"
        )
