"""JSON-based versioned prompt storage."""

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import unified_diff
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("./data/prompts.json")


@dataclass
class PromptVersion:
    """A single version of a prompt."""

    version: int
    system_prompt: str
    variables: list[str]
    model: str
    parameters: dict[str, Any]
    created_at: str
    metrics: dict[str, Any] | None = None


@dataclass
class Prompt:
    """A prompt with version history."""

    id: str
    name: str
    description: str
    current_version: int
    versions: list[dict[str, Any]]
    tags: list[str]
    created_at: str
    updated_at: str


class PromptStore:
    """CRUD + versioning for prompts stored as JSON.

    Args:
        store_path: Path to the JSON file.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = store_path or DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._save([])

    def create(
        self,
        name: str,
        system_prompt: str,
        *,
        description: str = "",
        model: str = "",
        parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Create a new prompt with version 1.

        Args:
            name: Prompt display name.
            system_prompt: The prompt text.
            description: Optional description.
            model: Target model ID.
            parameters: Generation parameters (temperature, etc.).
            tags: Optional tags for categorization.

        Returns:
            Created prompt dict.
        """
        prompts = self._load()
        variables = self._extract_variables(system_prompt)

        version = {
            "version": 1,
            "system_prompt": system_prompt,
            "variables": variables,
            "model": model,
            "parameters": parameters or {},
            "created_at": datetime.now().isoformat(),
            "metrics": None,
        }

        prompt = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "description": description,
            "current_version": 1,
            "versions": [version],
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        prompts.append(prompt)
        self._save(prompts)
        logger.info("Created prompt %s: %s", prompt["id"], name)
        return prompt

    def get(self, prompt_id: str) -> dict | None:
        """Get a prompt by ID with all versions.

        Args:
            prompt_id: Prompt ID.

        Returns:
            Prompt dict or None.
        """
        for p in self._load():
            if p["id"] == prompt_id:
                return p
        return None

    def list_all(self, tag: str | None = None) -> list[dict]:
        """List all prompts, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by.

        Returns:
            List of prompt dicts (newest first).
        """
        prompts = self._load()
        if tag:
            prompts = [p for p in prompts if tag in p.get("tags", [])]
        return sorted(
            prompts, key=lambda p: p.get("updated_at", ""), reverse=True
        )

    def update(
        self,
        prompt_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> dict | None:
        """Update prompt metadata (not content — use add_version for that).

        Args:
            prompt_id: Prompt ID.
            name: New name.
            description: New description.
            tags: New tags.

        Returns:
            Updated prompt dict or None.
        """
        prompts = self._load()
        for p in prompts:
            if p["id"] == prompt_id:
                if name is not None:
                    p["name"] = name
                if description is not None:
                    p["description"] = description
                if tags is not None:
                    p["tags"] = tags
                p["updated_at"] = datetime.now().isoformat()
                self._save(prompts)
                return p
        return None

    def add_version(
        self,
        prompt_id: str,
        system_prompt: str,
        *,
        model: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict | None:
        """Add a new version to a prompt.

        Args:
            prompt_id: Prompt ID.
            system_prompt: New prompt text.
            model: Target model (inherits from previous if None).
            parameters: Generation parameters (inherits if None).

        Returns:
            New version dict or None if prompt not found.
        """
        prompts = self._load()
        for p in prompts:
            if p["id"] == prompt_id:
                prev = p["versions"][-1]
                new_version = len(p["versions"]) + 1
                variables = self._extract_variables(system_prompt)

                version = {
                    "version": new_version,
                    "system_prompt": system_prompt,
                    "variables": variables,
                    "model": model if model is not None else prev.get("model", ""),
                    "parameters": parameters if parameters is not None else prev.get("parameters", {}),
                    "created_at": datetime.now().isoformat(),
                    "metrics": None,
                }

                p["versions"].append(version)
                p["current_version"] = new_version
                p["updated_at"] = datetime.now().isoformat()
                self._save(prompts)
                logger.info("Added version %d to prompt %s", new_version, prompt_id)
                return version
        return None

    def get_version(self, prompt_id: str, version: int) -> dict | None:
        """Get a specific version of a prompt.

        Args:
            prompt_id: Prompt ID.
            version: Version number (1-based).

        Returns:
            Version dict or None.
        """
        prompt = self.get(prompt_id)
        if not prompt:
            return None
        for v in prompt["versions"]:
            if v["version"] == version:
                return v
        return None

    def diff_versions(self, prompt_id: str, v1: int, v2: int) -> dict | None:
        """Generate a unified diff between two versions.

        Args:
            prompt_id: Prompt ID.
            v1: First version number.
            v2: Second version number.

        Returns:
            Dict with diff lines and metadata, or None.
        """
        ver1 = self.get_version(prompt_id, v1)
        ver2 = self.get_version(prompt_id, v2)
        if not ver1 or not ver2:
            return None

        diff_lines = list(unified_diff(
            ver1["system_prompt"].splitlines(keepends=True),
            ver2["system_prompt"].splitlines(keepends=True),
            fromfile=f"v{v1}",
            tofile=f"v{v2}",
        ))

        return {
            "prompt_id": prompt_id,
            "v1": v1,
            "v2": v2,
            "diff": "".join(diff_lines),
            "v1_variables": ver1["variables"],
            "v2_variables": ver2["variables"],
            "variables_added": [v for v in ver2["variables"] if v not in ver1["variables"]],
            "variables_removed": [v for v in ver1["variables"] if v not in ver2["variables"]],
        }

    def delete(self, prompt_id: str) -> bool:
        """Delete a prompt.

        Args:
            prompt_id: Prompt ID.

        Returns:
            True if deleted, False if not found.
        """
        prompts = self._load()
        original = len(prompts)
        prompts = [p for p in prompts if p["id"] != prompt_id]
        if len(prompts) < original:
            self._save(prompts)
            return True
        return False

    @staticmethod
    def _extract_variables(text: str) -> list[str]:
        """Extract {{variable}} placeholders from prompt text."""
        return sorted(set(re.findall(r"\{\{(\w+)\}\}", text)))

    def _load(self) -> list[dict]:
        with open(self.store_path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, prompts: list[dict]) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
