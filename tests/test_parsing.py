"""The parsers must reproduce the probabilities stored in the study's final tables.

``fixtures/parsing_cases.json`` is a stratified sample of (raw reply, stored value)
pairs.  Run against the full tables, the same parsers reproduced all ~673k stored
probabilities; the only two deviations were values the table had rounded to four
decimals, hence the tolerance below.
"""
import json
from pathlib import Path

import pytest

from vlmbench.parsing import parse_amd_json, parse_amd_text, parse_binary_json, parse_binary_prob

CASES = json.loads((Path(__file__).parent / "fixtures" / "parsing_cases.json").read_text())
TOL = 1e-4


def _parse(case):
    kind = case["kind"]
    if kind == "binary_local":
        return parse_binary_prob(case["raw"])
    if kind == "binary_closed":
        return parse_binary_json(case["raw"], case["label_key"])[2]
    if kind == "amd_text":
        return parse_amd_text(case["raw"])
    return parse_amd_json(case["raw"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["kind"])
def test_parser_reproduces_stored_value(case):
    got, expected = _parse(case), case["expected"]
    if expected is None:
        assert got is None
    elif isinstance(expected, list):
        assert got is not None and all(abs(a - b) < TOL for a, b in zip(got, expected))
    else:
        assert got is not None and abs(got - expected) < TOL


def test_spaced_decimal_is_not_read_as_zero():
    # MedGemma-4B occasionally writes "0.  85"; it must parse as 0.85, not 0.
    assert parse_binary_prob("0.  85") == pytest.approx(0.85)


def test_all_zero_amd_vector_is_unusable():
    assert parse_amd_text("Normal=0.00 dry=0.00 wet=0.00 PCV=0.00") is None


def test_last_amd_block_wins():
    raw = "Normal=0.70 dry=0.10 wet=0.10 PCV=0.10\nOn reflection:\nNormal=0.10 dry=0.10 wet=0.70 PCV=0.10"
    assert parse_amd_text(raw)[2] == pytest.approx(0.7)
