"""Placebo images and the placebo condition grid.

Each placebo request holds the real image(s) of an image-only configuration plus
uninformative images, and reuses that configuration's prompt verbatim: the text
never mentions the extra images.  Three arms differ only in the added pixels:

``dup``    the real image repeated
``blank``  uniform mid-grey, RGB (128, 128, 128)
``noise``  independent uniform RGB noise, seeded from the MD5 of a per-case string,
           so every model receives the same noise image for a given case

Generated images have the pixel size of the real image they accompany and are
sent losslessly (PNG).  ``tests/test_placebo.py`` checks the PNG bytes against the
images used in the study.

Grid (counts per model):
  prostate  each single sequence + 2 placebo images (dup, blank, noise) and each
            sequence pair + 1 placebo image (blank, noise): 15 conditions x 1,500 cases
  skin      clinical or dermoscopic image + 1 placebo image (dup, blank, noise):
            6 conditions x 1,011 lesions
  AMD       the eye's first CFP image, and each OCT B-scan separately, + 1 placebo
            image (dup, blank, noise): 3 conditions x 1,985 units
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .data import Unit

BLANK_RGB = (128, 128, 128)


def blank_image(size):
    return Image.new("RGB", size, BLANK_RGB)


def noise_image(size, seed_text: str):
    seed = int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    width, height = size
    return Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8"), "RGB")


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass(frozen=True)
class Placeholder:
    """A placebo image, rendered only when a request is actually sent."""

    kind: str                   # "blank" or "noise"
    size: tuple                 # (width, height)
    seed: Optional[str] = None  # noise only

    def render(self):
        return blank_image(self.size) if self.kind == "blank" else noise_image(self.size, self.seed)


def _size(path):
    with Image.open(path) as im:
        return im.size


def _counts(n_real, n_dup, n_blank, n_noise):
    return {"n_real_modalities": n_real, "n_dup": n_dup, "n_blank": n_blank, "n_noise": n_noise}


def prostate_units(data, case_filter=None):
    """``data`` is a :class:`vlmbench.data.prostate.ProstateData`."""
    for fname in data.files(["t2", "dwi", "adc"]):
        case_key = fname.rsplit(".", 1)[0]
        if case_filter is not None and not case_filter(case_key):
            continue
        paths = {m: data.root / m / fname for m in ("t2", "dwi", "adc")}
        truth = data.truth(case_key)
        for base in ("t2", "dwi", "adc"):
            real, size = paths[base], _size(paths[base])
            blank, noise = Placeholder("blank", size), Placeholder("noise", size, f"{case_key}:{base}")
            for cond, arm, images, counts in (
                (f"{base}_dup_3", "dup", [real, real, real], _counts(1, 2, 0, 0)),
                (f"{base}_blank_3", "blank", [real, blank, blank], _counts(1, 0, 2, 0)),
                (f"{base}_noise_3", "noise", [real, noise, noise], _counts(1, 0, 0, 2)),
            ):
                yield Unit(key=(cond, case_key), images=images,
                           meta={"condition": cond, "arm": arm, "base_modality": base,
                                 "modality_combo": base, "num_images": 3, **counts,
                                 "case_key": case_key, "true_binary": truth},
                           prompt_args={"input_type": "image_only", "modalities": [base]})
        for first, second in (("t2", "dwi"), ("t2", "adc"), ("dwi", "adc")):
            size = _size(paths[first])
            blank = Placeholder("blank", size)
            noise = Placeholder("noise", size, f"{case_key}:{first}_{second}")
            real = [paths[first], paths[second]]
            for cond, arm, images, counts in (
                (f"{first}_{second}_blank_1", "blank", real + [blank], _counts(2, 0, 1, 0)),
                (f"{first}_{second}_noise_1", "noise", real + [noise], _counts(2, 0, 0, 1)),
            ):
                yield Unit(key=(cond, case_key), images=images,
                           meta={"condition": cond, "arm": arm, "base_modality": first,
                                 "modality_combo": f"{first}+{second}", "num_images": 3,
                                 **counts, "case_key": case_key, "true_binary": truth},
                           prompt_args={"input_type": "image_only",
                                        "modalities": [first, second]})


def skin_units(data, case_filter=None):
    """``data`` is a :class:`vlmbench.data.skin.SkinData`; lesions need both images."""
    for _, row in data.frame.iterrows():
        if row.get("clinic") is None or row.get("derm") is None:
            continue
        paths = {"clinic": data.image_path(row["clinic"]), "derm": data.image_path(row["derm"])}
        if not all(p.exists() for p in paths.values()):
            continue
        case_key, seed_key = data.case_key(row), data.noise_seed_key(row)
        if case_filter is not None and not case_filter(case_key):
            continue
        for base in ("clinic", "derm"):
            real, size = paths[base], _size(paths[base])
            blank, noise = Placeholder("blank", size), Placeholder("noise", size, f"{seed_key}:{base}")
            for cond, arm, images, counts in (
                (f"{base}_dup_2", "dup", [real, real], _counts(1, 1, 0, 0)),
                (f"{base}_blank_2", "blank", [real, blank], _counts(1, 0, 1, 0)),
                (f"{base}_noise_2", "noise", [real, noise], _counts(1, 0, 0, 1)),
            ):
                yield Unit(key=(cond, case_key), images=images,
                           meta={"condition": cond, "arm": arm, "base_modality": base,
                                 "modality_combo": base, "num_images": 2, **counts,
                                 "case_key": case_key, "true_binary": data.truth(row)},
                           prompt_args={"input_type": "image_only", "modalities": [base]})


def amd_units(data, row_filter=None):
    """``data`` is a :class:`vlmbench.data.amd.AmdData`."""
    for idx, row in enumerate(data.rows):
        if row_filter is not None and not row_filter(idx):
            continue
        cfp = [p for p in (row["cfp_files"] or "").split("|") if p]
        octs = [p for p in (row["oct_files"] or "").split("|") if p]
        if not cfp or not octs or not (data.root / cfp[0]).exists():
            continue
        bases = [("cfp", -1, cfp[0])] + [("oct", i, rel) for i, rel in enumerate(octs)
                                         if (data.root / rel).exists()]
        tc = data.true_class(row)
        for base, sidx, rel in bases:
            path = data.root / rel
            sample_key = f"{row['eye_id']}|{base}|{sidx}"
            size = _size(path)
            blank, noise = Placeholder("blank", size), Placeholder("noise", size, sample_key)
            for cond, arm, images, counts in (
                (f"{base}_dup_2", "dup", [path, path], _counts(1, 1, 0, 0)),
                (f"{base}_blank_2", "blank", [path, blank], _counts(1, 0, 1, 0)),
                (f"{base}_noise_2", "noise", [path, noise], _counts(1, 0, 0, 1)),
            ):
                yield Unit(key=(cond, sample_key), images=images,
                           meta={"condition": cond, "arm": arm, "base_modality": base,
                                 "modality_combo": base, "num_images": 2, **counts,
                                 "sample_key": sample_key, "eye_id": row["eye_id"],
                                 "subject_id": row["subject_id"],
                                 "cfp_file": rel if base == "cfp" else "",
                                 "oct_slice_idx": sidx,
                                 "oct_slice_file": rel if base == "oct" else "",
                                 "true_class": tc})
