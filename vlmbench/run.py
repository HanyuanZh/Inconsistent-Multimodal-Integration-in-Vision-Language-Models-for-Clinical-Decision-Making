"""Run one model on one experiment, one request at a time.

Local models run on the GPU(s) of this process (``device_map="auto"``).  API
models are called synchronously -- convenient for small runs; use
``python -m vlmbench batch`` for full runs (the Batch APIs are cheaper).
Rows are appended as they finish; re-running the same command resumes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from . import tasks
from .images import load_rgb
from .prompts import MissingClinicalValue
from .table import RowWriter, done_keys, unit_key


def output_path(out_dir, experiment, dataset, model_key, shard, num_shards):
    name = model_key if num_shards == 1 else f"{model_key}.part{shard:02d}of{num_shards:02d}"
    return Path(out_dir) / experiment / dataset / f"{name}.csv"


def selected_units(args):
    for i, unit in enumerate(tasks.units(args.experiment, args.dataset, args.data_root,
                                         types=args.input_types)):
        if args.limit and i >= args.limit:
            break
        if i % args.num_shards == args.shard:
            yield unit


def load_backend(model):
    """Callable API client, or a loaded local model (see :mod:`vlmbench.backends`)."""
    if model.is_api:
        from .backends.api import client_for
        return client_for(model)
    from .backends import local
    return local.load(model)


def run(args, model, recipe):
    key_cols = tasks.KEY_COLUMNS[(args.experiment, args.dataset)]
    out = output_path(args.out, args.experiment, args.dataset, model.key, args.shard, args.num_shards)
    done = done_keys(out, key_cols)
    raw_col = tasks.raw_column(args.experiment, args.dataset)
    binary = args.dataset != "amd"

    backend = None if args.dry_run else load_backend(model)

    n_new = n_err = 0
    started = time.time()
    with RowWriter(out, tasks.columns(args.experiment, args.dataset)) as writer:
        for unit in selected_units(args):
            row = tasks.base_row(unit, args.experiment, args.dataset, model.key)
            if unit_key(row, key_cols) in done:
                continue
            try:
                text = tasks.prompt(unit, args.dataset, recipe)
                if args.dry_run:
                    print(f"--- {unit_key(row, key_cols)}  images={[str(i) for i in unit.images]}")
                    print(text)
                    if n_new >= 2:
                        return
                    n_new += 1
                    continue
                if model.is_api:
                    raw = backend(unit.images, text, recipe)
                else:
                    images = [load_rgb(i) for i in unit.images]
                    if binary and recipe.label_channel:
                        raw_l, probs = backend.label_probs(
                            images, tasks.prompt(unit, args.dataset, recipe, label=True))
                        row.update(label_logit_raw=raw_l,
                                   logit_prob_1=probs["1"] if probs else None,
                                   label_logit=(int(probs["1"] >= probs["0"]) if probs else None))
                    raw = backend.generate(images, text, recipe.max_new_tokens)
                row[raw_col] = raw
                row.update(tasks.interpret(args.dataset, recipe, raw))
            except MissingClinicalValue as exc:
                row["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - record and continue
                if type(exc).__name__ == "OutOfMemoryError":
                    import torch
                    torch.cuda.empty_cache()
                    row["error"] = "OOM"
                else:
                    row["error"] = repr(exc)[:500]
            n_new += 1
            n_err += bool(row.get("error"))
            writer.write(row)
            if n_new % 50 == 0:
                rate = n_new / (time.time() - started)
                print(f"[{model.key} {args.experiment}/{args.dataset} shard {args.shard}] "
                      f"{n_new} done ({rate:.2f}/s), {n_err} with error", file=sys.stderr, flush=True)
    print(f"wrote {n_new} rows ({n_err} with error) -> {out}")
