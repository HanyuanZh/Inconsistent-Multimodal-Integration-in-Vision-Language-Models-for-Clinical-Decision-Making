"""Batch API workflow for the API models (about half the price of synchronous calls).

  build   write request JSONL files (offline; no key, no cost)
  submit  upload them and create batch jobs           -- costs money, needs --yes
  status  show job states
  fetch   download finished results and write the output CSV

Files live in ``<out>/<experiment>/<dataset>/<model>_batch/``.  Each request file
has a sidecar manifest with the output-row metadata of every request; units that
cannot be sent (missing clinical value) are kept in the manifest as error rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import tasks
from .backends.api import gemini_request, gemini_response_text, openai_body, openai_response_text
from .prompts import MissingClinicalValue
from .table import RowWriter

ENDPOINT = "/v1/chat/completions"
# Provider limits: Gemini input files <= 2 GB; OpenAI <= 200 MB and 50,000 requests.
LIMITS = {"gemini": (int(1.75 * 1024 ** 3), 10 ** 9), "openai": (190 * 1024 ** 2, 50_000)}


def batch_dir(args, model):
    return Path(args.out) / args.experiment / args.dataset / f"{model.key}_batch"


def _state_path(d):
    return d / "state.json"


def _load_state(d):
    p = _state_path(d)
    return json.loads(p.read_text()) if p.exists() else {"chunks": []}


def _save_state(d, state):
    tmp = _state_path(d).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(_state_path(d))


def build(args, model, recipe):
    d = batch_dir(args, model)
    d.mkdir(parents=True, exist_ok=True)
    if _load_state(d)["chunks"] and not args.force:
        raise SystemExit(f"{d} already has a build; pass --force to rebuild")
    max_bytes, max_requests = LIMITS[model.family]
    chunks, idx = [], 0
    req_fh = man_fh = None
    size = count = 0

    def open_chunk():
        nonlocal req_fh, man_fh, size, count
        if req_fh:
            req_fh.close(); man_fh.close()
        n = len(chunks) + 1
        chunks.append({"index": n, "requests": f"chunk{n:03d}_requests.jsonl",
                       "manifest": f"chunk{n:03d}_manifest.jsonl", "n_requests": 0})
        req_fh = (d / chunks[-1]["requests"]).open("w")
        man_fh = (d / chunks[-1]["manifest"]).open("w")
        size = count = 0

    open_chunk()
    for i, unit in enumerate(tasks.units(args.experiment, args.dataset, args.data_root,
                                         types=args.input_types)):
        if args.limit and i >= args.limit:
            break
        row = tasks.base_row(unit, args.experiment, args.dataset, model.key)
        rid = f"{args.dataset}-{idx:08d}"
        idx += 1
        try:
            text = tasks.prompt(unit, args.dataset, recipe)
        except MissingClinicalValue as exc:
            man_fh.write(json.dumps({"id": None, "row": {**row, "error": str(exc)}}, default=str) + "\n")
            continue
        if model.family == "gemini":
            line = {"key": rid, "request": gemini_request(unit.images, text, recipe)}
        else:
            line = {"custom_id": rid, "method": "POST", "url": ENDPOINT,
                    "body": openai_body(unit.images, text, recipe, model.checkpoint)}
        data = json.dumps(line) + "\n"
        if count and (size + len(data) > max_bytes or count >= max_requests):
            open_chunk()
        req_fh.write(data)
        man_fh.write(json.dumps({"id": rid, "row": row}, default=str) + "\n")
        size += len(data); count += 1
        chunks[-1]["n_requests"] = count
    req_fh.close(); man_fh.close()
    state = {"model": model.checkpoint, "family": model.family, "recipe": recipe.__dict__,
             "chunks": chunks}
    _save_state(d, state)
    total = sum(c["n_requests"] for c in chunks)
    print(f"built {total} requests in {len(chunks)} file(s) -> {d}")


def _client(model):
    import os
    if model.family == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
    from openai import OpenAI
    return OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)


def submit(args, model, recipe):
    d = batch_dir(args, model)
    state = _load_state(d)
    pending = [c for c in state["chunks"] if not c.get("job") and c["n_requests"]]
    n = sum(c["n_requests"] for c in pending)
    if not args.yes:
        raise SystemExit(f"would submit {n} requests in {len(pending)} job(s) to {model.checkpoint}; "
                         "this incurs API charges -- re-run with --yes to proceed")
    client = _client(model)
    for c in pending:
        path = d / c["requests"]
        if model.family == "gemini":
            from google.genai import types
            up = client.files.upload(file=str(path), config=types.UploadFileConfig(
                display_name=f"{d.name}-{c['index']:03d}", mime_type="application/jsonl"))
            job = client.batches.create(model=model.checkpoint, src=up.name,
                                        config={"display_name": f"{d.name}-{c['index']:03d}"})
            c["file"], c["job"] = up.name, job.name
        else:
            with path.open("rb") as fh:
                up = client.files.create(file=fh, purpose="batch")
            job = client.batches.create(input_file_id=up.id, endpoint=ENDPOINT,
                                        completion_window="24h")
            c["file"], c["job"] = up.id, job.id
        _save_state(d, state)   # after every job, so a crash never loses a job id
        print(f"chunk {c['index']:03d}: {c['n_requests']} requests -> {c['job']}")


def _job_state(client, model, job_id):
    if model.family == "gemini":
        job = client.batches.get(name=job_id)
        st = job.state
        return getattr(st, "name", str(st)), job
    job = client.batches.retrieve(job_id)
    return job.status, job


def status(args, model, recipe):
    d = batch_dir(args, model)
    client = _client(model)
    for c in _load_state(d)["chunks"]:
        st = _job_state(client, model, c["job"])[0] if c.get("job") else "not submitted"
        print(f"chunk {c['index']:03d}  {c['n_requests']:6d} requests  {st}")


def fetch(args, model, recipe):
    d = batch_dir(args, model)
    state = _load_state(d)
    client = _client(model)
    done = {"JOB_STATE_SUCCEEDED", "completed"}
    replies = {}
    for c in state["chunks"]:
        if not c.get("job"):
            continue
        st, job = _job_state(client, model, c["job"])
        if st not in done:
            raise SystemExit(f"chunk {c['index']:03d} is {st}; fetch once every job has finished")
        res = d / f"chunk{c['index']:03d}_results.jsonl"
        if not res.exists():
            if model.family == "gemini":
                content = client.files.download(file=job.dest.file_name)
                res.write_bytes(content if isinstance(content, bytes) else content.encode())
            else:
                text = client.files.content(job.output_file_id).text
                if job.error_file_id:
                    text += client.files.content(job.error_file_id).text
                res.write_text(text)
        for line in res.read_text().splitlines():
            o = json.loads(line)
            if model.family == "gemini":
                rid = o.get("key")
                replies[rid] = (gemini_response_text(o["response"]) if "response" in o
                                else f"<batch error: {json.dumps(o.get('error'))[:300]}>")
            else:
                rid, resp = o.get("custom_id"), o.get("response") or {}
                replies[rid] = (openai_response_text(resp.get("body") or {})
                                if resp.get("status_code") == 200
                                else f"<batch error: {json.dumps(o.get('error') or resp)[:300]}>")
    out = Path(args.out) / args.experiment / args.dataset / f"{model.key}.csv"
    raw_col = tasks.raw_column(args.experiment, args.dataset)
    from .recipes import Recipe
    rec = Recipe(**state["recipe"])
    n = missing = 0
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    out.unlink(missing_ok=True)
    with RowWriter(out, tasks.columns(args.experiment, args.dataset)) as writer:
        for c in state["chunks"]:
            for line in (d / c["manifest"]).read_text().splitlines():
                m = json.loads(line)
                row = m["row"]
                if m["id"] is not None:
                    raw = replies.get(m["id"])
                    if raw is None:
                        row["error"] = "no_result"
                        missing += 1
                    else:
                        row[raw_col] = raw
                        row.update(tasks.interpret(args.dataset, rec, raw))
                writer.write(row)
                n += 1
    print(f"wrote {n} rows ({missing} without a result) -> {out}")
