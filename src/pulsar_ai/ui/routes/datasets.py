"""Dataset management routes: upload, list, preview, delete."""

import logging
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from pulsar_ai.ui.auth import get_current_user, get_scoped_user_id, get_user_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["datasets"])

DATA_DIR = Path("./data/uploads")


def _get_datasets_dir(user_id: str | None = None) -> Path:
    """Ensure and return the datasets upload directory.

    Args:
        user_id: If provided, returns a user-scoped subdirectory.
            If None (admin), returns the base DATA_DIR.

    Returns:
        Path to the datasets directory.
    """
    if user_id is not None:
        user_dir = DATA_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _read_dataframe(path: Path) -> pd.DataFrame:
    """Read a dataset file into a DataFrame.

    Args:
        path: Path to the dataset file.

    Returns:
        Pandas DataFrame.

    Raises:
        ValueError: If format is unsupported.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


@router.post("/datasets/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...)) -> dict:
    """Upload a dataset file (CSV, JSONL, Parquet, Excel).

    Returns dataset metadata including ID, name, format, and row count.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".csv", ".jsonl", ".parquet", ".xlsx", ".xls"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Use CSV, JSONL, Parquet, or Excel.",
        )

    user_id = get_user_id(request)
    dataset_id = str(uuid.uuid4())[:8]
    dest_dir = _get_datasets_dir(user_id)
    dest_path = dest_dir / f"{dataset_id}{suffix}"

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    try:
        df = _read_dataframe(dest_path)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    return {
        "id": dataset_id,
        "name": file.filename,
        "format": suffix.lstrip("."),
        "num_rows": len(df),
        "columns": list(df.columns),
        "size_bytes": len(content),
        "path": str(dest_path),
    }


@router.get("/datasets")
async def list_datasets(request: Request) -> list[dict]:
    """List all uploaded datasets."""
    user_id = get_scoped_user_id(request)
    results: list[dict] = []
    if user_id is None:
        # Admin: list datasets from all user subdirectories
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dirs = [d for d in sorted(DATA_DIR.iterdir()) if d.is_dir()]
    else:
        dirs = [_get_datasets_dir(user_id)]
    for dest_dir in dirs:
        for path in sorted(dest_dir.iterdir()):
            if path.is_file() and path.suffix in (
                ".csv", ".jsonl", ".parquet", ".xlsx", ".xls",
            ):
                try:
                    df = _read_dataframe(path)
                    results.append(
                        {
                            "id": path.stem,
                            "name": path.name,
                            "format": path.suffix.lstrip("."),
                            "num_rows": len(df),
                            "size_bytes": path.stat().st_size,
                            "path": str(path),
                        }
                    )
                except Exception:
                    logger.warning("Failed to read dataset: %s", path)
    return results


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, request: Request, rows: int = 20) -> dict:
    """Preview first N rows of a dataset.

    Args:
        dataset_id: Dataset identifier (filename stem).
        rows: Number of rows to return (default 20, max 100).
    """
    rows = min(rows, 100)
    user_id = get_scoped_user_id(request)
    dest_dir = _get_datasets_dir(user_id)

    for path in dest_dir.iterdir():
        if path.stem == dataset_id:
            try:
                df = _read_dataframe(path)
                return {
                    "columns": list(df.columns),
                    "rows": df.head(rows).to_dict(orient="records"),
                    "total_rows": len(df),
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, request: Request) -> dict:
    """Delete an uploaded dataset."""
    user_id = get_scoped_user_id(request)
    dest_dir = _get_datasets_dir(user_id)
    for path in dest_dir.iterdir():
        if path.stem == dataset_id:
            path.unlink()
            return {"id": dataset_id, "deleted": True}
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
