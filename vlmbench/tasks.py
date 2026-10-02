"""What each (experiment, dataset) runs: its units, prompts, output columns and how a reply is read.

Shared by the local runner, the synchronous API runner and the Batch API path,
so all three send the same requests and write the same columns.
"""

from __future__ import annotations

from .data.amd import AmdData
from .data.prostate import INPUT_TYPES as PROSTATE_TYPES
from .data.prostate import ProstateData
from .data.skin import INPUT_TYPES as SKIN_TYPES
from .data.skin import SkinData
from .parsing import argmax, parse_amd_json, parse_amd_text, parse_binary_json, parse_binary_prob
from .placebo import amd_units, prostate_units, skin_units
from .prompts import AMD_PROB_COLUMNS, amd_prompt, prostate_prompt, skin_prompt

LABEL_KEY = {"prostate": "cspca", "skin": "biopsy"}

_BINARY_RESULT = ["true_binary", "prob_verbalized", "label_pred"]
_AMD_RESULT = ["true_class", *AMD_PROB_COLUMNS, "pred_class"]
_PLACEBO_DESIGN = ["condition", "arm", "base_modality", "modality_combo", "num_images",
                   "n_real_modalities", "n_dup", "n_blank", "n_noise"]

COLUMNS = {
    ("main", "prostate"): ["input_type", "modality_combo", "num_modalities", "case_key", "psa",
                           "prostate_volume", *_BINARY_RESULT, "prob_verbalized_raw",
                           "label_logit", "logit_prob_1", "label_logit_raw"],
    ("main", "skin"): ["input_type", "modality_combo", "num_modalities", "case_key", "location",
                       "elevation", *_BINARY_RESULT, "prob_verbalized_raw",
                       "label_logit", "logit_prob_1", "label_logit_raw"],
    ("main", "amd"): ["modality", "eye_id", "subject_id", "oct_slice_idx", "oct_slice_file",
                      "num_input_images", "true_class_label", *_AMD_RESULT, "prob_raw"],
    ("placebo", "prostate"): [*_PLACEBO_DESIGN, "case_key", *_BINARY_RESULT, "prob_raw"],
    ("placebo", "skin"): [*_PLACEBO_DESIGN, "case_key", *_BINARY_RESULT, "prob_raw"],
    ("placebo", "amd"): [*_PLACEBO_DESIGN, "sample_key", "eye_id", "subject_id", "cfp_file",
                         "oct_slice_idx", "oct_slice_file", *_AMD_RESULT, "prob_raw"],
}
# columns that identify a unit within one output file (used to resume)
KEY_COLUMNS = {
    ("main", "prostate"): ("input_type", "modality_combo", "case_key"),
    ("main", "skin"): ("input_type", "modality_combo", "case_key"),
    ("main", "amd"): ("eye_id", "modality", "oct_slice_idx"),
    ("placebo", "prostate"): ("condition", "case_key"),
    ("placebo", "skin"): ("condition", "case_key"),
    ("placebo", "amd"): ("condition", "sample_key"),
}


def columns(experiment, dataset):
    return ["experiment", "dataset", "model", *COLUMNS[(experiment, dataset)], "error"]


def input_types(dataset):
    return {"prostate": PROSTATE_TYPES, "skin": SKIN_TYPES}.get(dataset, ())


def units(experiment, dataset, data_root, types=None):
    """All units of one experiment, in a fixed order."""
    if dataset == "prostate":
        data = ProstateData(data_root)
        if experiment == "placebo":
            yield from prostate_units(data)
        else:
            for t in (types or PROSTATE_TYPES):
                yield from data.units(t)
    elif dataset == "skin":
        data = SkinData(data_root)
        if experiment == "placebo":
            yield from skin_units(data)
        else:
            for t in (types or SKIN_TYPES):
                yield from data.units(t)
    else:
        data = AmdData(data_root)
        if experiment == "placebo":
            yield from amd_units(data)
        else:
            yield from data.units()


def prompt(unit, dataset, recipe, label=False):
    """Prompt for ``unit``; ``label=True`` gives the constrained-label variant.

    Raises :class:`vlmbench.prompts.MissingClinicalValue` when the unit lacks a
    clinical measure its input type needs.
    """
    fmt = "label" if label else recipe.prompt_format
    if dataset == "amd":
        return amd_prompt(fmt)
    builder = prostate_prompt if dataset == "prostate" else skin_prompt
    return builder(**unit.prompt_args, fmt=fmt)


def interpret(dataset, recipe, raw):
    """Result columns for the verbalised reply ``raw``."""
    if dataset == "amd":
        probs = parse_amd_json(raw) if recipe.prompt_format == "json" else parse_amd_text(raw)
        out = {"prob_raw": raw, "pred_class": argmax(probs) if probs else None}
        out.update(zip(AMD_PROB_COLUMNS, probs or [None] * 4))
        if probs is None:
            out["error"] = "unparseable_reply"
        return out
    if recipe.prompt_format == "json":
        p1 = parse_binary_json(raw, LABEL_KEY[dataset])[2]
    else:
        p1 = parse_binary_prob(raw)
    out = {"prob_verbalized": p1, "label_pred": None if p1 is None else int(p1 >= 0.5)}
    if p1 is None:
        out["error"] = "unparseable_reply"
    return out


def raw_column(experiment, dataset):
    return "prob_verbalized_raw" if (experiment == "main" and dataset != "amd") else "prob_raw"


def base_row(unit, experiment, dataset, model_key):
    row = {"experiment": experiment, "dataset": dataset, "model": model_key}
    row.update(unit.meta)
    if dataset == "amd" and "true_class_label" not in row and row.get("true_class", -1) in range(4):
        from .prompts import AMD_CLASSES
        row["true_class_label"] = AMD_CLASSES[row["true_class"]]
    return row
