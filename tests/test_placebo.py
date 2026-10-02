"""Generated placebo images must be byte-identical to those used in the study."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vlmbench.placebo import blank_image, noise_image, png_bytes

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text())["placebo_images"]


def md5(b):
    return hashlib.md5(b).hexdigest()


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_placebo_image_bytes(key):
    kind, *rest = key.split("|")
    w, h = map(int, rest[-1].split("x"))
    if kind == "blank":
        assert md5(png_bytes(blank_image((w, h)))) == GOLDEN[key]["png_md5"]
    else:
        img = noise_image((w, h), "|".join(rest[:-1]))  # AMD seeds contain "|"
        assert md5(png_bytes(img)) == GOLDEN[key]["png_md5"]
        assert md5(np.asarray(img).tobytes()) == GOLDEN[key]["pixels_md5"]


def test_noise_differs_between_cases_and_is_stable():
    a1, a2 = noise_image((32, 32), "c1:t2"), noise_image((32, 32), "c1:t2")
    b = noise_image((32, 32), "c2:t2")
    assert png_bytes(a1) == png_bytes(a2) != png_bytes(b)
