"""Derm7pt skin lesions.

Expected layout (``--data-root``)::

    skin/
      images/<as referenced by meta.csv>    clinical and dermoscopic images
      meta/meta.csv                         official Derm7pt metadata

A lesion is positive when ``management == "excision"``.  ``case_key`` is the
Derm7pt ``case_num`` (unique on all 1,011 rows); ``case_id`` is populated on only
27 rows and is not an identifier.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..prompts import SKIN_INPUT_TYPES
from . import Unit, resolve_case_insensitive

COMBOS = [["clinic"], ["derm"], ["clinic", "derm"]]
INPUT_TYPES = tuple(SKIN_INPUT_TYPES)
_TEXT_COLS = ("clinic", "derm", "location", "elevation", "case_num", "case_id",
              "sex", "diagnosis", "management")


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    return None if s.lower() in {"", "nan", "none", "<na>"} else s


class SkinData:
    def __init__(self, root):
        self.root = Path(root) / "skin"
        self.images = self.root / "images"
        frame = pd.read_csv(self.root / "meta" / "meta.csv")
        for col in _TEXT_COLS:
            if col in frame.columns:
                frame[col] = frame[col].map(_text)
        self.frame = frame

    def image_path(self, rel):
        return resolve_case_insensitive(self.images / rel)

    @staticmethod
    def case_key(row):
        return _text(row.get("case_num")) or _text(row.get("case_id"))

    @staticmethod
    def legacy_key(row):
        """Key that seeds the placebo noise images (case_id first, as in the original runs)."""
        return _text(row.get("case_id")) or _text(row.get("case_num")) or _text(row.get("clinic"))

    @staticmethod
    def truth(row):
        m = _text(row.get("management"))
        return None if m is None else int(m.lower() == "excision")

    def units(self, input_type, combos=None):
        extras = SKIN_INPUT_TYPES[input_type]
        for modalities in (combos or COMBOS):
            combo = "+".join(modalities)
            need = list(modalities) + list(extras)
            for _, row in self.frame.dropna(subset=need).iterrows():
                key = self.case_key(row)
                values = {f: row.get(f) for f in extras}
                yield Unit(
                    key=(input_type, combo, key),
                    images=[self.image_path(row[m]) for m in modalities],
                    meta={"input_type": input_type, "modality_combo": combo,
                          "num_modalities": len(modalities), "case_key": key,
                          "location": row.get("location") if "location" in extras else None,
                          "elevation": row.get("elevation") if "elevation" in extras else None,
                          "true_binary": self.truth(row)},
                    prompt_args={"input_type": input_type, "modalities": modalities,
                                 "extras": values},
                )
