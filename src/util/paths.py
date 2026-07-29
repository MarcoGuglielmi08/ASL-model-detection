from pathlib import Path


# Shared project paths used by pipeline modules.
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_DIR = ROOT_DIR / "results"
NESTED_DIR = RESULTS_DIR / "nested_cv"
STATISTICS_DIR = RESULTS_DIR/ "statistics"
PREPROCESSING_DIR = RESULTS_DIR / "preprocessing"
