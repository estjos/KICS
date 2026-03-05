from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class IGHM:
    """Information Gain Heat Map (IGHM) instance.

    This is a thin wrapper around the JSON format used in the project.
    The underlying representation is a pandas DataFrame where:
      - each column is a cell id (often int-like strings)
      - rows include x, y, height, width, gain, and suitability

    The code in this repo assumes the format seen in data/ighm_1.json.
    """

    df: pd.DataFrame

    @property
    def cell_ids(self) -> List[str]:
        return list(self.df.columns)

    def coords(self, cell_id: str) -> Tuple[float, float]:
        return float(self.df[cell_id]["x"]), float(self.df[cell_id]["y"])

    def gain(self, cell_id: str) -> float:
        return float(self.df[cell_id]["gain"])

    def suitability(self, cell_id: str) -> Dict[str, float]:
        s = self.df[cell_id]["suitability"]
        if isinstance(s, dict):
            return s
        # pandas sometimes loads nested objects as strings in edge cases
        raise TypeError(f"Unexpected suitability type for cell {cell_id}: {type(s)}")

    def best_suitability_key(self, cell_id: str) -> str:
        s = self.suitability(cell_id)
        return max(s, key=s.get)

    def zoom_levels(self, cell_id: str) -> List[int]:
        """Return all zoom/resolution integers encoded in suitability keys.

        Expected key format examples: 'SA_Z1', 'SB_Z3', 'SC_Z9'.
        """
        z = []
        for k in self.suitability(cell_id).keys():
            if "_Z" not in k:
                continue
            try:
                z.append(int(k.split("_Z")[-1]))
            except ValueError:
                continue
        return z


def load_ighm_json(path: str | Path) -> IGHM:
    """Load an IGHM JSON file into an `IGHM` object."""
    p = Path(path)
    df = pd.read_json(p)
    return IGHM(df=df)
