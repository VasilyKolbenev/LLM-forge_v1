"""Image loading utilities for multimodal training.

Supports loading images from local paths and URLs with caching.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_CACHE_DIR = Path("data/image_cache")
IMAGE_COLUMN_NAMES = {"image", "image_path", "image_url", "img", "photo"}


def load_image(source: str) -> Any:
    """Load an image from a local path or URL.

    URLs are cached locally to avoid repeated downloads.

    Args:
        source: Local file path or HTTP(S) URL.

    Returns:
        PIL.Image.Image object.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If source is empty.
    """
    from PIL import Image

    source = source.strip()
    if not source:
        raise ValueError("Empty image source")

    # URL handling
    if source.startswith("http://") or source.startswith("https://"):
        return _load_from_url(source)

    # Local path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    return Image.open(path).convert("RGB")


def _load_from_url(url: str) -> Any:
    """Download and cache an image from URL.

    Args:
        url: HTTP(S) URL.

    Returns:
        PIL.Image.Image object.
    """
    from PIL import Image

    # Check cache first
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = Path(url.split("?")[0]).suffix or ".jpg"
    cache_path = IMAGE_CACHE_DIR / f"{url_hash}{ext}"

    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    # Download
    import urllib.request
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading image: %s", url[:80])
    urllib.request.urlretrieve(url, str(cache_path))
    return Image.open(cache_path).convert("RGB")


def resolve_image_column(df: pd.DataFrame) -> str | None:
    """Detect the image column in a DataFrame.

    Args:
        df: DataFrame to inspect.

    Returns:
        Column name if found, None otherwise.
    """
    for col in df.columns:
        if col.lower() in IMAGE_COLUMN_NAMES:
            return col
    return None


def validate_images(
    df: pd.DataFrame,
    image_column: str,
    base_dir: str = "",
) -> pd.DataFrame:
    """Filter out rows with missing or broken images.

    Args:
        df: Source DataFrame.
        image_column: Column containing image paths/URLs.
        base_dir: Base directory for relative paths.

    Returns:
        Filtered DataFrame with only valid images.
    """
    valid_indices = []

    for idx, row in df.iterrows():
        source = str(row.get(image_column, "")).strip()
        if not source:
            continue

        # Resolve relative paths
        if base_dir and not source.startswith("http") and not Path(source).is_absolute():
            source = str(Path(base_dir) / source)

        # Check existence (skip URL validation for speed)
        if source.startswith("http"):
            valid_indices.append(idx)
        elif Path(source).exists():
            valid_indices.append(idx)
        else:
            logger.warning("Image not found, skipping row %s: %s", idx, source)

    filtered = df.loc[valid_indices].reset_index(drop=True)
    dropped = len(df) - len(filtered)
    if dropped > 0:
        logger.warning("Dropped %d rows with invalid images", dropped)

    return filtered
