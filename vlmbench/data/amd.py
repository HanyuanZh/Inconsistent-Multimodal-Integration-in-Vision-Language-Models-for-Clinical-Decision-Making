"""MMC-AMD, eye-paired subset ("EyePaired_AMD_448").

Expected layout (``--data-root``)::

    amd/EyePaired_AMD_448/
      manifest.csv     eye_id, subject_id, original_label, cfp_files, oct_files
                       (file lists are "|"-separated, relative to this directory)
      AMD/<eye_id>/{cfp,oct}/*.jpg
      Normal/<eye_id>/{cfp,oct}/*.jpg

Class ids: 0 Normal (original label "healthy"), 1 dry AMD, 2 wet AMD, 3 PCV.

Units: per eye one ``cfp`` request (its CFP image), and per OCT B-scan one ``oct``
request and one ``cfp_oct`` request (CFP followed by that B-scan) -- 768 + 1,217 +
1,217 requests.  One eye (ICG-4449-r) has two CFP images; both are sent.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..prompts import AMD_CLASSES
from . import Unit

ORIG_TO_CLASS = {"healthy": 0, "dry_amd": 1, "wet_amd": 2, "pcv": 3}
MODALITIES = ("cfp", "oct", "cfp_oct")


def _split(value):
    return [p for p in (value or "").split("|") if p]


class AmdData:
    def __init__(self, root):
        self.root = Path(root) / "amd" / "EyePaired_AMD_448"
        with (self.root / "manifest.csv").open(newline="") as f:
            self.rows = list(csv.DictReader(f))

    @staticmethod
    def true_class(row):
        return ORIG_TO_CLASS.get((row.get("original_label") or "").strip(), -1)

    def _meta(self, row, modality, slice_idx, slice_file, n_images):
        tc = self.true_class(row)
        return {"modality": modality, "eye_id": row["eye_id"], "subject_id": row["subject_id"],
                "oct_slice_idx": slice_idx, "oct_slice_file": slice_file,
                "num_input_images": n_images, "true_class": tc,
                "true_class_label": AMD_CLASSES[tc] if 0 <= tc < 4 else ""}

    def units(self, modalities=MODALITIES, row_filter=None):
        """``row_filter(index) -> bool`` selects manifest rows."""
        for idx, row in enumerate(self.rows):
            if row_filter is not None and not row_filter(idx):
                continue
            cfp, octs = _split(row["cfp_files"]), _split(row["oct_files"])
            for modality in modalities:
                if modality == "cfp":
                    items = [(-1, "", cfp)]
                else:
                    base = cfp if modality == "cfp_oct" else []
                    items = [(i, octs[i], base + [octs[i]]) for i in range(len(octs))]
                for sidx, sfile, rels in items:
                    yield Unit(
                        key=(row["eye_id"], modality, str(sidx)),
                        images=[self.root / r for r in rels],
                        meta=self._meta(row, modality, sidx, sfile, len(rels)),
                    )
