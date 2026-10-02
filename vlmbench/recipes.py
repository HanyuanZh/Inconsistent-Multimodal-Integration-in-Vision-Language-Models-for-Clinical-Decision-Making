"""Models and the per-run settings used in the study.

A *recipe* fixes everything that changes a model's answer: checkpoint, prompt
format, decoding and generation length.  ``recipe(experiment, dataset, model)``
returns the setting the paper's numbers were produced with; every field can be
overridden from the command line.

Experiments
-----------
``main``     the imaging x clinical-measure grid (prostate 28, skin 12, AMD 3 settings)
``placebo``  image-only inputs with silently appended duplicate / blank / noise images
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

DATASETS = ("prostate", "skin", "amd")
EXPERIMENTS = ("main", "placebo")


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    family: str          # medgemma | qwen | lingshu | deepseek | gemini | openai
    checkpoint: str      # Hugging Face id, or API model id
    revision: Optional[str] = None   # Hugging Face commit the study ran

    @property
    def is_api(self) -> bool:
        return self.family in ("gemini", "openai")


MODELS = {m.key: m for m in (
    Model("medgemma4", "MedGemma-4B", "medgemma", "google/medgemma-4b-it",
          "290cda5eeccbee130f987c4ad74a59ae6f196408"),
    Model("medgemma27", "MedGemma-27B", "medgemma", "google/medgemma-27b-it",
          "2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f"),
    Model("lingshu7", "Lingshu-7B", "lingshu", "lingshu-medical-mllm/Lingshu-7B",
          "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"),
    Model("lingshu32", "Lingshu-32B", "lingshu", "lingshu-medical-mllm/Lingshu-32B",
          "36b98277cacb60db86f34b75ce0540b1ea35183c"),
    Model("deepseektiny", "DeepSeek-VL2-tiny", "deepseek", "deepseek-ai/deepseek-vl2-tiny",
          "66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa"),
    Model("deepseeksmall", "DeepSeek-VL2-small", "deepseek", "deepseek-ai/deepseek-vl2-small",
          "6033e16432a1d771cf9fe4a6f894ff5e5e1459af"),
    Model("qwen36", "Qwen3.6-35B-A3B", "qwen", "Qwen/Qwen3.6-35B-A3B",
          "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    Model("gemini", "Gemini-3.5-Flash", "gemini", "gemini-3.5-flash"),
    Model("gpt", "GPT-5.6-Sol", "openai", "gpt-5.6-sol"),
)}

# The four models of the placebo experiment.
PLACEBO_MODELS = ("qwen36", "lingshu32", "gemini", "gpt")


@dataclass(frozen=True)
class Recipe:
    prompt_format: str                    # prob | prob_strict | json  (see prompts.FORMATS)
    max_new_tokens: int                   # verbalised reply budget (API: output-token cap)
    label_channel: bool = False           # local binary tasks: also record the constrained 0/1 token
    temperature: Optional[float] = None   # API models; local models decode greedily
    thinking_level: Optional[str] = None  # Gemini; None = API default
    reasoning_effort: Optional[str] = None  # OpenAI; None = API default
    image_detail: Optional[str] = None    # OpenAI image_url detail

    def override(self, **changes) -> "Recipe":
        return replace(self, **{k: v for k, v in changes.items() if v is not None})


# Local models decode greedily (do_sample=False).  A verbalised probability is a
# few tokens, so 16 never binds on the binary tasks; the AMD reply is a
# four-number line, and some models reason before it, hence 2048.
_LOCAL_BINARY = Recipe("prob", 16, label_channel=True)
_LOCAL_AMD = Recipe("prob", 2048)
_AMD_STRICT = ("medgemma4", "medgemma27", "qwen36")

# Gemini: temperature 1.0, default thinking level.  The cap was 2048 in the
# original runs; 8192 is the default here because six main-grid requests spent
# all 2048 tokens on thinking and had to be re-issued at 8192 (temperature 1
# makes the runs non-deterministic anyway, and a larger cap only matters when
# 2048 would have truncated).
_GEMINI = Recipe("json", 8192, temperature=1.0)

# GPT: temperature 1.0 (the model rejects other values), default reasoning effort.
_GPT = Recipe("json", 4096, temperature=1.0)


def recipe(experiment: str, dataset: str, model: str) -> Recipe:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}")
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {', '.join(MODELS)}")
    if experiment == "placebo" and model not in PLACEBO_MODELS:
        raise ValueError(f"the placebo experiment ran {PLACEBO_MODELS}, not {model!r}")

    family = MODELS[model].family
    if family == "gemini":
        if experiment == "placebo":
            # the placebo batches named the thinking level explicitly
            return replace(_GEMINI, thinking_level="medium")
        return _GEMINI
    if family == "openai":
        detail = "low" if dataset == "prostate" else "high"
        tokens = 2048 if (dataset == "amd" and experiment == "main") else 4096
        return replace(_GPT, image_detail=detail, max_new_tokens=tokens)

    if dataset == "amd":
        # Under the standard wording these three models wrote paragraphs of
        # reasoning before (or instead of) the four-number line; they were run
        # with the wording that asks for that line only.  The placebo arms use
        # the same wording as the main grid, so the prompts stay identical.
        if model in _AMD_STRICT:
            return replace(_LOCAL_AMD, prompt_format="prob_strict")
        return _LOCAL_AMD
    if experiment == "placebo":
        # placebo arms record the verbalised probability only
        return replace(_LOCAL_BINARY, label_channel=False)
    if dataset == "prostate" and model == "medgemma27":
        # Under the standard wording MedGemma-27B answered with radiology prose
        # on most prostate requests; it was run with the wording that forbids
        # explanation, with room for the reply.
        return replace(_LOCAL_BINARY, prompt_format="prob_strict", max_new_tokens=2048)
    return _LOCAL_BINARY
