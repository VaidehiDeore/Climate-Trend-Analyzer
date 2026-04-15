import pandas as pd
from pathlib import Path


def load_climate_data(file_path: str) -> pd.DataFrame:
    """
    Load climate data from a CSV or Excel file.
    Uses a safer CSV parser setup for low-memory systems.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".csv":
        try:
            # First try normal read
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            # Fallback: Python engine is slower but more memory-safe on some systems
            df = pd.read_csv(path, engine="python")
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

    return df