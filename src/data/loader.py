import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import (
    VALID_SUBJECTS,
    LABEL_NAMES,
    FS,
    WESAD_DIR,
)
from ..logging_config import LoggerMixin


class WESADLoader(LoggerMixin):
    VALID_SUBJECTS = VALID_SUBJECTS
    LABELS = LABEL_NAMES

    CHEST_FS = FS.CHEST
    WRIST_ACC_FS = FS.WRIST_ACC
    WRIST_BVP_FS = FS.WRIST_BVP
    WRIST_EDA_FS = FS.WRIST_EDA
    WRIST_TEMP_FS = FS.WRIST_TEMP

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else WESAD_DIR
        self._validate_path()
        self.subjects = self._discover_subjects()

        self.logger.info(
            f"WESADLoader initialized: {len(self.subjects)} subjects available"
        )

    def _validate_path(self) -> None:
        if not self.data_path.exists():
            self.logger.error(f"WESAD data path not found: {self.data_path}")
            raise FileNotFoundError(
                f"WESAD data path not found: {self.data_path}\n"
                "Please download the dataset from: "
                "https://archive.ics.uci.edu/ml/datasets/WESAD\n"
                "See data/raw/README.md for instructions."
            )

    def _discover_subjects(self) -> List[str]:
        subjects = []
        for subj_id in self.VALID_SUBJECTS:
            pkl_path = self.data_path / subj_id / f"{subj_id}.pkl"
            if pkl_path.exists():
                subjects.append(subj_id)

        if not subjects:
            self.logger.warning("No subjects found in dataset directory")

        return subjects

    def load_subject(
        self, subject_id: str, signals: Optional[List[str]] = None
    ) -> Dict:
        if subject_id not in self.VALID_SUBJECTS:
            raise ValueError(
                f"Invalid subject ID: {subject_id}. "
                f"Valid subjects: {self.VALID_SUBJECTS}"
            )

        pkl_path = self.data_path / subject_id / f"{subject_id}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Subject data not found: {pkl_path}")

        self.logger.info(f"Loading subject {subject_id} from {pkl_path}")

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        except Exception as e:
            self.logger.error(f"Failed to load {subject_id}: {e}")
            raise

        chest_signals = data["signal"]["chest"]
        wrist_signals = data["signal"]["wrist"]

        if signals is not None:
            signals_upper = {s.upper() for s in signals}
            chest_signals = {
                k: v for k, v in chest_signals.items() if k.upper() in signals_upper
            }
            wrist_signals = {
                k: v for k, v in wrist_signals.items() if k.upper() in signals_upper
            }
            self.logger.debug(f"Filtered to signals: {signals}")

        result = {
            "subject": subject_id,
            "chest": chest_signals,
            "wrist": wrist_signals,
            "label": data["label"],
        }

        self.logger.info(
            f"Loaded {subject_id}: {len(result['label'])} samples "
            f"({len(result['label']) / self.CHEST_FS:.1f}s)"
        )

        return result

    def __repr__(self) -> str:
        return f"WESADLoader(path='{self.data_path}', subjects={len(self.subjects)})"

    def __len__(self) -> int:
        return len(self.subjects)
