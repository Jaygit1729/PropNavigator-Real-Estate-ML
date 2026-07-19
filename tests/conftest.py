"""Shared pytest fixtures.

Data and model files are loaded once per session (module scope) — reading a 39k-row
CSV or a 4 MB model for every test would make the suite needlessly slow.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fs_df() -> pd.DataFrame:
    """The modelling dataset: 24 features + target, already de-duplicated."""
    path = ROOT / "data" / "fs" / "feature_selected_properties.csv"
    if not path.exists():
        pytest.skip(f"{path} missing — run `python -m src.main` first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def raw_pp_df() -> pd.DataFrame:
    """Preprocessed data (35 columns), before feature selection."""
    path = ROOT / "data" / "pp" / "preprocessed_properties.csv"
    if not path.exists():
        pytest.skip(f"{path} missing — run `python -m src.main` first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def service():
    """The API inference module (loads model + reference tables once)."""
    model = ROOT / "artifacts" / "best_model.joblib"
    if not model.exists():
        pytest.skip("artifacts/best_model.joblib missing — train a model first")
    from api import inference
    return inference


@pytest.fixture
def valid_request() -> dict:
    """A known-good /predict payload."""
    return {
        "property_type": "Flat",
        "sector": "sector 49",
        "area": 1500,
        "bedRoom": 3,
        "bathroom": 2,
        "furnishing": "semi-furnished",
        "age_possession_category": "New Property",
        "covered_parking": 1,
        "total_floor": 15,
        "has_ac": 1,
        "has_power_backup": 1,
        "has_pool": 0,
        "is_corner": 0,
    }
