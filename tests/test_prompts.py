"""The prompts must match, byte for byte, what the original study runners sent.

``fixtures/golden.json`` holds the MD5 of every prompt rendered from the original
scripts (all four local model families and both API providers were rendered
separately and found identical before the fixture was written).
"""
import hashlib
import json
from pathlib import Path

import pytest

from vlmbench.prompts import amd_prompt, prostate_prompt, skin_prompt

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text())
PROMPT_KEYS = sorted(k for k in GOLDEN if k.count("|") >= 1 and k.split("|")[0] in
                     {"prostate", "skin", "amd"})


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def render(key):
    parts = key.split("|")
    dataset, fmt = parts[0], parts[-1]
    if dataset == "amd":
        return amd_prompt(fmt)
    input_type, combo = parts[1], parts[2].split("+")
    if dataset == "prostate":
        psa, vol = float(parts[3]), float(parts[4])
        return prostate_prompt(input_type, combo, psa=psa, prostate_volume=vol, fmt=fmt)
    loc, elev = parts[3], parts[4]
    return skin_prompt(input_type, combo, {"location": loc, "elevation": elev}, fmt=fmt)


def test_fixture_is_complete():
    # 4 prostate input types x 7 combos x 4 clinical points x 4 formats
    # + 4 skin input types x 3 combos x 3 value pairs x 3 formats + 4 AMD formats
    assert len(PROMPT_KEYS) == 448 + 108 + 4


@pytest.mark.parametrize("key", PROMPT_KEYS)
def test_prompt_matches_original(key):
    assert md5(render(key)) == GOLDEN[key]


def test_placebo_prompts_match_submitted_requests():
    """The placebo arms reuse the image-only JSON prompt verbatim."""
    rendered = {
        "prostate": {md5(prostate_prompt("image_only", c.split("+"), fmt="json"))
                     for c in ("t2", "dwi", "adc", "t2+dwi", "t2+adc", "dwi+adc")},
        "skin": {md5(skin_prompt("image_only", [b], fmt="json")) for b in ("clinic", "derm")},
        "amd": {md5(amd_prompt("json"))},
    }
    for key, submitted in GOLDEN["submitted_placebo_prompts"].items():
        dataset = key.split("|")[1]
        assert set(submitted) <= rendered[dataset], key
