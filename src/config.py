"""Project configuration: paths, sampling rates, filter and feature parameters, seed, and log settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Global random seed for reproducible runs; imported by scripts and set_seed().
SEED: int = 42

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
WESAD_DIR = RAW_DATA_DIR / "WESAD"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"

# S1, S12 excluded
VALID_SUBJECTS: List[str] = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
]


@dataclass(frozen=True)
class SamplingRates:
    # Chest (RespiBAN), all channels at 700 Hz
    CHEST: float = 700.0

    # Wrist (Empatica E4)
    WRIST_ACC: float = 32.0
    WRIST_BVP: float = 64.0
    WRIST_EDA: float = 4.0
    WRIST_TEMP: float = 4.0


FS = SamplingRates()


@dataclass(frozen=True)
class FilterParams:
    ECG_BANDPASS_LOW: float = 0.5
    ECG_BANDPASS_HIGH: float = 40.0
    ECG_FILTER_ORDER: int = 4

    EDA_LOWPASS: float = 5.0
    EDA_MEDIAN_SIZE: int = 5
    EDA_FILTER_ORDER: int = 4

    RESP_BANDPASS_LOW: float = 0.1
    RESP_BANDPASS_HIGH: float = 0.5
    RESP_FILTER_ORDER: int = 4

    TEMP_LOWPASS: float = 0.1
    TEMP_FILTER_ORDER: int = 2


FILTER_PARAMS = FilterParams()


@dataclass(frozen=True)
class FeatureParams:
    WINDOW_SIZE_SEC: float = 60.0
    WINDOW_OVERLAP: float = 0.5
    EPSILON: float = 1e-10


FEATURE_PARAMS = FeatureParams()

LOG_LEVEL: str = "INFO"
LOG_FILE: Path = LOGS_DIR / "calmsense.log"
