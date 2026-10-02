"""Command line.

  python -m vlmbench recipe                       settings used for every run in the paper
  python -m vlmbench run   -e main -d prostate -m qwen36 --data-root data
  python -m vlmbench batch build -e placebo -d skin -m gpt --data-root data
  python -m vlmbench merge -e main -d prostate -m qwen36
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import sys
from pathlib import Path

from .recipes import DATASETS, EXPERIMENTS, MODELS, PLACEBO_MODELS, recipe


def _common(p, need_data=True, data_required=True):
    p.add_argument("-e", "--experiment", choices=EXPERIMENTS, required=True)
    p.add_argument("-d", "--dataset", choices=DATASETS, required=True)
    p.add_argument("-m", "--model", choices=sorted(MODELS), required=True)
    p.add_argument("--out", default="results", help="output root (default: results/)")
    if need_data:
        p.add_argument("--data-root", required=data_required,
                       help="directory holding prostate/, skin/ and amd/ (see data/README.md)")
        p.add_argument("--input-types", nargs="+",
                       help="main grid only: restrict to these clinical-input settings")
        p.add_argument("--limit", type=int, default=0, help="only the first N units (smoke test)")
    g = p.add_argument_group("overrides of the paper setting (see `recipe`)")
    g.add_argument("--checkpoint", help="local model path or alternative model id")
    g.add_argument("--prompt-format", choices=["prob", "prob_strict", "json"])
    g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--temperature", type=float)
    g.add_argument("--thinking-level", help="Gemini, e.g. minimal/low/medium/high")
    g.add_argument("--reasoning-effort", help="OpenAI, e.g. low/medium/high")
    g.add_argument("--image-detail", choices=["low", "high", "auto"])


def _resolve(args):
    model = MODELS[args.model]
    if args.checkpoint:
        model = dataclasses.replace(model, checkpoint=args.checkpoint, revision=None)
    rec = recipe(args.experiment, args.dataset, args.model).override(
        prompt_format=args.prompt_format, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, thinking_level=args.thinking_level,
        reasoning_effort=args.reasoning_effort, image_detail=args.image_detail)
    return model, rec


def _print_recipes():
    rows = []
    for exp in EXPERIMENTS:
        for ds in DATASETS:
            for key, model in MODELS.items():
                try:
                    r = recipe(exp, ds, key)
                except ValueError:
                    continue
                extra = []
                if model.is_api:
                    extra.append(f"temperature={r.temperature}")
                    if model.family == "openai":
                        extra.append(f"detail={r.image_detail}")
                else:
                    extra.append("greedy")
                    if r.label_channel:
                        extra.append("+ constrained 0/1 label")
                ckpt = model.checkpoint + (f"@{model.revision[:7]}" if model.revision else "")
                rows.append((exp, ds, model.name, ckpt, r.prompt_format,
                             str(r.max_new_tokens), ", ".join(extra)))
    widths = [max(len(r[i]) for r in rows) for i in range(7)]
    head = ("experiment", "dataset", "model", "checkpoint", "prompt", "max tokens", "decoding")
    widths = [max(w, len(h)) for w, h in zip(widths, head)]
    for r in [head] + rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vlmbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("recipe", help="print the paper's setting for every run")

    p = sub.add_parser("run", help="run a model one request at a time (local GPU or sync API)")
    _common(p)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="print the first prompts, call nothing")

    b = sub.add_parser("batch", help="Batch API workflow for gemini / gpt")
    b.add_argument("action", choices=["build", "submit", "status", "fetch"])
    _common(b, data_required=False)
    b.add_argument("--yes", action="store_true", help="submit: confirm the API charges")
    b.add_argument("--force", action="store_true", help="build/fetch: overwrite existing files")

    m = sub.add_parser("merge", help="concatenate the shard files of a run")
    _common(m, need_data=False)

    args = ap.parse_args(argv)
    if args.cmd == "recipe":
        _print_recipes()
        return
    model, rec = _resolve(args)
    if args.experiment == "placebo" and args.model not in PLACEBO_MODELS:
        ap.error(f"the placebo experiment ran {', '.join(PLACEBO_MODELS)}")

    if args.cmd == "run":
        from .run import run
        run(args, model, rec)
    elif args.cmd == "batch":
        if not model.is_api:
            ap.error("batch is for the API models (gemini, gpt)")
        if args.action == "build" and not args.data_root:
            ap.error("batch build needs --data-root")
        from . import batch
        getattr(batch, args.action)(args, model, rec)
    else:
        import pandas as pd
        d = Path(args.out) / args.experiment / args.dataset
        parts = sorted(glob.glob(str(d / f"{model.key}.part*of*.csv")))
        if not parts:
            sys.exit(f"no shard files for {model.key} in {d}")
        frame = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in parts])
        frame.to_csv(d / f"{model.key}.csv", index=False)
        print(f"merged {len(parts)} files, {len(frame)} rows -> {d / (model.key + '.csv')}")


if __name__ == "__main__":
    main()
