"""PI-CAI prostate MRI.

Expected layout (``--data-root``)::

    prostate/
      t2/<patient_id>_<study_id>.png     T2-weighted slice
      dwi/<patient_id>_<study_id>.png    high b-value DWI slice
      adc/<patient_id>_<study_id>.png    ADC map slice
      marksheet.csv                      PI-CAI marksheet (patient_id, study_id, psa,
                                         prostate_volume, case_csPCa, ...)

A case is included in a modality combination when a file of the same name exists
in every directory of the combination.
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import pandas as pd

from ..prompts import PROSTATE_INPUT_TYPES, PROSTATE_MODALITIES
from . import VALID_EXT, Unit

COMBOS = [list(c) for r in (1, 2, 3) for c in combinations(PROSTATE_MODALITIES, r)]
INPUT_TYPES = tuple(PROSTATE_INPUT_TYPES)


def _clean(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


class ProstateData:
    def __init__(self, root):
        self.root = Path(root) / "prostate"
        sheet = pd.read_csv(self.root / "marksheet.csv")
        sheet["case_key"] = sheet["patient_id"].astype(str) + "_" + sheet["study_id"].astype(str)
        self.clinical = sheet.set_index("case_key").to_dict("index")

    def files(self, modalities):
        sets = []
        for m in modalities:
            d = self.root / m
            if not d.is_dir():
                raise FileNotFoundError(f"modality directory not found: {d}")
            sets.append({p.name for p in d.iterdir() if p.suffix.lower() in VALID_EXT})
        return sorted(set.intersection(*sets))

    def truth(self, case_key):
        value = str(self.clinical.get(case_key, {}).get("case_csPCa", "")).strip().upper()
        return {"YES": 1, "NO": 0}.get(value)

    def units(self, input_type, combos=None):
        """Main-grid units: every case x modality combination for one input type."""
        for modalities in (combos or COMBOS):
            combo = "+".join(modalities)
            for fname in self.files(modalities):
                case_key = Path(fname).stem
                clin = self.clinical.get(case_key, {})
                psa, vol = _clean(clin.get("psa")), _clean(clin.get("prostate_volume"))
                yield Unit(
                    key=(input_type, combo, case_key),
                    images=[self.root / m / fname for m in modalities],
                    meta={"input_type": input_type, "modality_combo": combo,
                          "num_modalities": len(modalities), "case_key": case_key,
                          "psa": psa, "prostate_volume": vol,
                          "true_binary": self.truth(case_key)},
                    prompt_args={"input_type": input_type, "modalities": modalities,
                                 "psa": psa, "prostate_volume": vol},
                )
