"""SQLite-based persistence layer for llm-forge.

Replaces the load-mutate-save JSON stores with a transactional backend.
"""

from llm_forge.storage.database import Database, get_database

__all__ = ["Database", "get_database"]
