"""Recipe registry: scan, filter, and load YAML recipe templates."""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_RECIPES_DIR = _PROJECT_ROOT / "configs" / "recipes"


class RecipeRegistry:
    """Scan and serve recipe YAML templates from a directory.

    Args:
        recipes_dir: Directory containing recipe YAML files.
            Defaults to ``configs/recipes/`` relative to project root.
    """

    def __init__(
        self, recipes_dir: Path | None = None
    ) -> None:
        self._dir = Path(recipes_dir) if recipes_dir else _DEFAULT_RECIPES_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_recipes(
        self,
        task_type: Optional[str] = None,
        tag: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List available recipes, optionally filtered.

        Args:
            task_type: Filter by ``meta.task_type`` (e.g. ``sft``).
            tag: Filter by tag presence in ``meta.tags``.
            difficulty: Filter by ``meta.difficulty``.

        Returns:
            List of recipe metadata dicts (including ``file``).
        """
        if not self._dir.is_dir():
            logger.warning("Recipes dir not found: %s", self._dir)
            return []

        results: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("*.yaml")):
            meta = self._read_meta(path)
            if meta is None:
                continue
            if task_type and meta.get("task_type") != task_type:
                continue
            if difficulty and meta.get("difficulty") != difficulty:
                continue
            if tag and tag not in meta.get("tags", []):
                continue
            meta["file"] = path.stem
            results.append(meta)
        return results

    def load_recipe(self, name: str) -> dict[str, Any]:
        """Load a recipe config with the ``meta`` block stripped.

        Args:
            name: Recipe file stem (without ``.yaml``).

        Returns:
            Config dict ready for training dispatch.

        Raises:
            FileNotFoundError: If the recipe file does not exist.
        """
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Recipe '{name}' not found in {self._dir}"
            )
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"Recipe '{name}' is not a valid YAML mapping"
            )
        data.pop("meta", None)
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_meta(self, path: Path) -> dict[str, Any] | None:
        """Extract the ``meta`` block from a recipe file.

        Args:
            path: Path to a recipe YAML file.

        Returns:
            Meta dict or *None* if the file is malformed / has no meta.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError:
            logger.warning("Skipping malformed YAML: %s", path)
            return None

        if not isinstance(data, dict):
            return None
        meta = data.get("meta")
        if not isinstance(meta, dict):
            return None
        return dict(meta)
