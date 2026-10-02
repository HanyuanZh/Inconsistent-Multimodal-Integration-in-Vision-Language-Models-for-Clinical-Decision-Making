"""Turn raw model replies into probabilities.

These parsers reproduce every probability in the study's final tables from the
stored raw replies (``tests/test_parsing.py`` checks a sample of them).

* ``parse_binary_prob``  verbalised number from a local model
* ``parse_binary_json``  {"cspca"/"biopsy": .., "prob_0": .., "prob_1": ..} from an API model
* ``parse_amd_text``     "Normal=.. dry=.. wet=.. PCV=.." line
* ``parse_amd_json``     {"amd_class": .., "prob_normal": .., ...} (falls back to the text form)
"""

from __future__ import annotations

import json
import math
import re
from typing import Optional

_ANCHOR = re.compile(r"[Pp]\s*=\s*(\d*\.?\d+)")
# A decimal; whitespace after the point is tolerated because MedGemma-4B sometimes
# writes "0.  85".  Without this, "0.  85" fell through to the bare-0/1 rule below
# and was read as 0.0.
_DECIMAL = re.compile(r"\d+\.\s*\d+")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_BARE_01 = re.compile(r"\b[01]\b")
_NUMBER = re.compile(r"\d+\.?\d*")


def _clamp(p: float) -> float:
    return max(0.0, min(1.0, p))


def parse_binary_prob(raw) -> Optional[float]:
    """Probability of label 1 stated in free text, or None.

    Priority: ``P=<x>`` anchor, then the last decimal in [0, 1], the last
    percentage, the last bare 0/1, and finally the last number (divided by 100
    when above 1).  Taking the last match matters for verbose replies, which
    restate intermediate numbers before the answer.
    """
    text = "" if raw is None else str(raw)
    if text.strip() in ("", "nan", "None"):
        return None
    m = _ANCHOR.search(text)
    if m:
        try:
            p = float(m.group(1))
            return _clamp(p / 100.0 if p > 1.0 else p)
        except ValueError:
            pass
    in_range = []
    for s in _DECIMAL.findall(text):
        try:
            v = float(re.sub(r"\s+", "", s))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            in_range.append(v)
    if in_range:
        return _clamp(in_range[-1])
    pcts, ones = _PERCENT.findall(text), _BARE_01.findall(text)
    if pcts:
        return _clamp(float(pcts[-1]) / 100.0)
    if ones:
        return _clamp(float(ones[-1]))
    nums = _NUMBER.findall(text)
    if not nums:
        return None
    p = float(nums[-1])
    return _clamp(p / 100.0 if p > 1.0 else p)


def _json_object(raw):
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _first_binary_digit(raw) -> int:
    m = re.search(r"[01]", raw or "")
    return int(m.group()) if m else -1


def parse_binary_json(raw, label_key: str):
    """(label, prob_0, prob_1) from an API model's JSON reply.

    The two probabilities are clamped to [0, 1] and renormalised to sum to 1.
    """
    data = _json_object(raw)
    if data is None:
        return _first_binary_digit(raw), None, None
    try:
        p0, p1 = float(data.get("prob_0")), float(data.get("prob_1"))
    except (TypeError, ValueError):
        p0 = p1 = None
    if p0 is not None and (not math.isfinite(p0) or not math.isfinite(p1)):
        p0 = p1 = None
    if p0 is not None:
        p0, p1 = _clamp(p0), _clamp(p1)
        total = p0 + p1
        p0, p1 = (p0 / total, p1 / total) if total > 0 else (None, None)
    try:
        label = int(data.get(label_key))
    except (TypeError, ValueError):
        label = -1
    if label not in (0, 1) and p0 is not None:
        label = int(p1 >= p0)
    if label not in (0, 1):
        label = _first_binary_digit(raw)
    return label, p0, p1


# A complete answer block: all four labels contiguous.  "<..>" is tolerated for
# models that copy the prompt's "Normal=<p0>" template literally.
_AMD_BLOCK = re.compile(
    r"normal\s*[:=]?\s*<?\s*(\d*\.?\d+)>?[\s,;]*"
    r"dry\s*[:=]?\s*<?\s*(\d*\.?\d+)>?[\s,;]*"
    r"wet\s*[:=]?\s*<?\s*(\d*\.?\d+)>?[\s,;]*"
    r"pcv\s*[:=]?\s*<?\s*(\d*\.?\d+)",
    re.IGNORECASE,
)
# Longer replies are reasoning, not an answer; guessing four numbers out of them
# would pick up intermediate musings.
_AMD_SHORT_REPLY = 200


def parse_amd_text(raw):
    """Four class probabilities (Normal, dry, wet, PCV) summing to 1, or None.

    The LAST complete answer block wins, because models that reason first state
    their conclusion at the end.  Without a block, a reply of at most 200
    characters is read as its first four numbers.  All-zero vectors are
    rejected: they cannot be normalised.
    """
    text = (raw or "").strip()
    if not text:
        return None
    matches = list(_AMD_BLOCK.finditer(text))
    if matches:
        try:
            vals = [float(g) for g in matches[-1].groups()]
        except ValueError:
            return None
    else:
        if len(text) > _AMD_SHORT_REPLY:
            return None
        vals = []
        for n in re.findall(r"\d*\.?\d+", text):
            try:
                vals.append(float(n))
            except ValueError:
                continue
            if len(vals) == 4:
                break
        if len(vals) < 4:
            return None
    vals = [_clamp(v / 100.0 if v > 1.0 else v) for v in vals]
    total = sum(vals)
    return [v / total for v in vals] if total > 0 else None


def parse_amd_json(raw):
    """Four class probabilities from the JSON contract; text form as fallback."""
    data = _json_object(raw)
    if data is not None:
        try:
            vals = [float(data[k]) for k in
                    ("prob_normal", "prob_dry_amd", "prob_wet_amd", "prob_pcv")]
        except (KeyError, TypeError, ValueError):
            vals = None
        if vals is not None and all(math.isfinite(v) for v in vals):
            vals = [_clamp(v) for v in vals]
            total = sum(vals)
            if total > 0:
                return [v / total for v in vals]
    return parse_amd_text(raw)


def argmax(probs) -> int:
    return max(range(len(probs)), key=lambda i: probs[i])
