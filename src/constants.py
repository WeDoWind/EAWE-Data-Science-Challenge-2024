from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Time period that covers the full RPM range (used for Sub-challenges 1-2)
FULL_RPM_RANGE_PERIOD = (pd.Timestamp("2023-06-09 06:33:36.659"), pd.Timestamp("2023-06-09 08:37:02.460"))