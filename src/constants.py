"""Defines project constants"""

from pathlib import Path

this = Path(__file__)

ROOT = this.parents[1]

LOGS = ROOT / "logs"

SRC = ROOT / "src"

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
DATA_INTERIM = DATA / "interim"

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

# Create necessary directories
LOGS.mkdir(exist_ok=True, parents=True)
DATA_RAW.mkdir(exist_ok=True, parents=True)
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
DATA_INTERIM.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)
RESULTS.mkdir(exist_ok=True, parents=True)
FIGURES.mkdir(exist_ok=True, parents=True)
TABLES.mkdir(exist_ok=True, parents=True)
