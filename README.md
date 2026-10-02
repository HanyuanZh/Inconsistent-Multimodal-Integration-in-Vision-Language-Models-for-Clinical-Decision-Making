# Inconsistent Multimodal Integration in Vision-Language Models for Clinical Decision-Making

Inference code for the paper of the same title.

Nine vision-language models are asked for a **verbalised probability** of disease
on three tasks, for every combination of imaging modalities and clinical measures:

| Task | Dataset | Imaging | Clinical measures | Classes |
|---|---|---|---|---|
| prostate | PI-CAI, 1,500 bpMRI exams | T2W, high-b DWI, ADC | PSA, prostate volume | csPCa yes / no |
| skin | Derm7pt, 1,011 lesions | clinical photo, dermoscopy | location, elevation | excision yes / no |
| amd | MMC-AMD eye-paired subset, 768 eyes | CFP, OCT | — | Normal / dry AMD / wet AMD / PCV |

Local checkpoints are pinned to the Hugging Face commits the study ran
(`python -m vlmbench recipe` shows them); `--checkpoint` loads another model or a
local copy instead.

| Model | Checkpoint | Runs on |
|---|---|---|
| MedGemma-4B | `google/medgemma-4b-it` | local GPU |
| MedGemma-27B | `google/medgemma-27b-it` | local GPU |
| Lingshu-7B / -32B | `lingshu-medical-mllm/Lingshu-7B`, `-32B` | local GPU |
| DeepSeek-VL2-tiny / -small | `deepseek-ai/deepseek-vl2-tiny`, `-small` | local GPU (separate env) |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | local GPU |
| Gemini-3.5-Flash | `gemini-3.5-flash` | Google Gemini API |
| GPT-5.6-Sol | `gpt-5.6-sol` | OpenAI API |

## Install

```bash
pip install -r requirements.txt            # MedGemma, Lingshu, Qwen, Gemini, GPT
# DeepSeek-VL2 needs transformers 4.38, so give it its own environment:
pip install -r requirements-deepseek.txt
```

API keys are read from the environment only — `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
and `OPENAI_API_KEY`. Never put keys in files inside this repository.

## Data

The datasets are not redistributed. See [data/README.md](data/README.md) for where to
obtain them and the directory layout the code expects under `--data-root`.

## Usage

```bash
# the setting of every run in the paper (checkpoint, prompt, token budget, decoding)
python -m vlmbench recipe

# print the first prompts of a run without loading a model
python -m vlmbench run -e main -d prostate -m qwen36 --data-root /path/to/data --dry-run

# local model, one GPU (re-running the same command resumes)
python -m vlmbench run -e main -d prostate -m qwen36 --data-root /path/to/data

# local model, 4 GPUs as 4 shards, then merged
scripts/run_local_shards.sh main prostate qwen36 /path/to/data 4

# API model through the Batch API (about half price)
python -m vlmbench batch build  -e main -d skin -m gpt --data-root /path/to/data
python -m vlmbench batch submit -e main -d skin -m gpt --yes      # incurs charges
python -m vlmbench batch status -e main -d skin -m gpt
python -m vlmbench batch fetch  -e main -d skin -m gpt
```

`run` also works for the API models (one synchronous call per request), which is
handy for a quick `--limit 20` check.

Results go to `results/<experiment>/<dataset>/<model>.csv`.

### Experiments

| `-e` | What | Requests per model |
|---|---|---|
| `main` | every imaging combination × clinical-measure setting | prostate 42,000 (7 × 4 × 1,500), skin 12,132 (3 × 4 × 1,011), AMD 3,202 (768 CFP + 1,217 OCT + 1,217 CFP + OCT) |
| `placebo` | image-only inputs with silently appended duplicate, blank or noise images; prompt unchanged | prostate 22,500, skin 6,066, AMD 5,955 |

On AMD every OCT B-scan is its own request, alone (`oct`) and paired with the eye's
CFP (`cfp_oct`); each eye's CFP is also sent alone (`cfp`). The placebo experiment was
run for Qwen3.6, Lingshu-32B, Gemini and GPT.

Prostate cases lacking PSA or prostate volume (896 of the 42,000 prostate requests)
are written with `error = missing_psa` / `missing_prostate_volume` and not sent.

### Output columns

| Column | Meaning |
|---|---|
| `prob_verbalized` | binary tasks: stated probability of the positive class |
| `prob_normal` … `prob_pcv`, `pred_class` | AMD: stated class probabilities (normalised) and their argmax |
| `label_pred` | `prob_verbalized >= 0.5` |
| `prob_verbalized_raw` / `prob_raw` | the model's reply, verbatim |
| `label_logit`, `logit_prob_1`, `label_logit_raw` | local models, binary `main` only: one constrained `0`/`1` token and its softmax probability |
| `true_binary` / `true_class` | reference label |
| `error` | why a row has no probability (`missing_*`, `unparseable_reply`, API errors) |

## Default settings

`python -m vlmbench recipe` prints the setting of every run.

* **Local models**: greedy decoding in bfloat16; up to 16 new tokens on the binary
  tasks and 2,048 on AMD. MedGemma-27B on prostate, and MedGemma-4B, MedGemma-27B and
  Qwen3.6 on AMD, use the wording that asks for the probability only.
* **Gemini-3.5-Flash**: temperature 1.0, default thinking level, JSON reply.
* **GPT-5.6-Sol**: temperature 1.0, default reasoning effort, JSON reply;
  `image_detail` low for prostate, high for skin and AMD.

## Tests

```bash
pytest          # no GPU, no API calls, no dataset needed
```

## Citation

To be added on publication.
