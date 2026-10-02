"""End-to-end runs on a tiny synthetic dataset with stub models (no GPU, no API calls)."""
import csv
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vlmbench import batch, run as runmod
from vlmbench.recipes import MODELS, recipe


def _png(path, size=(8, 8), value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, value).save(path)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    pro = root / "prostate"
    for case in ("1_11", "2_22"):
        for m in ("t2", "dwi", "adc"):
            _png(pro / m / f"{case}.png")
    (pro / "marksheet.csv").write_text(
        "patient_id,study_id,psa,prostate_volume,case_csPCa\n1,11,7.7,55,YES\n2,22,,40,NO\n")
    skin = root / "skin"
    for rel in ("A/a1.jpg", "A/a2.jpg", "B/b1.jpg", "B/b2.jpg"):
        _png(skin / "images" / rel, size=(12, 8))
    (skin / "meta").mkdir(parents=True)
    (skin / "meta" / "meta.csv").write_text(
        "case_num,case_id,clinic,derm,location,elevation,management\n"
        "1,,A/a1.jpg,A/a2.jpg,back,flat,excision\n"
        "2,3a1,B/b1.jpg,b/B2.JPG,chest,palpable,clinical follow up\n")  # wrong case on purpose
    amd = root / "amd" / "EyePaired_AMD_448"
    for rel in ("AMD/e1/cfp/f.jpg", "AMD/e1/oct/o1.jpg", "AMD/e1/oct/o2.jpg"):
        _png(amd / rel, size=(10, 10))
    (amd / "manifest.csv").write_text(
        "eye_id,subject_id,original_label,cfp_files,oct_files\n"
        "e1,s1,wet_amd,AMD/e1/cfp/f.jpg,AMD/e1/oct/o1.jpg|AMD/e1/oct/o2.jpg\n")
    return root


class StubLocal:
    def __init__(self):
        self.calls = []

    def generate(self, images, prompt, max_new_tokens):
        self.calls.append((len(images), prompt, max_new_tokens))
        return "Normal=0.1 dry=0.2 wet=0.6 PCV=0.1" if "retinal" in prompt else "0.8"

    def label_probs(self, images, prompt, labels=("0", "1")):
        assert prompt.endswith('Reply with exactly one character: "1" or "0".')
        return "1", {"0": 0.3, "1": 0.7}


def _args(tmp_path, data_root, **kw):
    base = dict(out=str(tmp_path / "out"), data_root=str(data_root), input_types=None, limit=0,
                shard=0, num_shards=1, dry_run=False, force=False, yes=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_local_main_prostate_with_resume(tmp_path, data_root, monkeypatch):
    stub = StubLocal()
    monkeypatch.setattr(runmod, "load_backend", lambda model: stub)
    args = _args(tmp_path, data_root, experiment="main", dataset="prostate", model="qwen36")
    runmod.run(args, MODELS["qwen36"], recipe("main", "prostate", "qwen36"))
    rows = _rows(tmp_path / "out/main/prostate/qwen36.csv")
    assert len(rows) == 2 * 7 * 4                      # cases x combos x input types
    missing = [r for r in rows if r["error"] == "missing_psa"]
    assert len(missing) == 7 * 2                       # case 2 lacks PSA: image_psa, image_psa_volume
    ok = [r for r in rows if not r["error"]]
    assert {r["prob_verbalized"] for r in ok} == {"0.8"} and {r["label_logit"] for r in ok} == {"1"}
    assert {r["true_binary"] for r in rows} == {"1", "0"}
    assert all(t == 16 for _, _, t in stub.calls)       # verbalised budget of the recipe
    n_calls = len(stub.calls)
    runmod.run(args, MODELS["qwen36"], recipe("main", "prostate", "qwen36"))   # resume
    assert len(stub.calls) == n_calls and len(_rows(tmp_path / "out/main/prostate/qwen36.csv")) == 56


def test_local_amd_one_unit_per_oct_slice(tmp_path, data_root, monkeypatch):
    stub = StubLocal()
    monkeypatch.setattr(runmod, "load_backend", lambda model: stub)
    args = _args(tmp_path, data_root, experiment="main", dataset="amd", model="qwen36")
    runmod.run(args, MODELS["qwen36"], recipe("main", "amd", "qwen36"))
    rows = _rows(tmp_path / "out/main/amd/qwen36.csv")
    # cfp once per eye; oct and cfp_oct once per B-scan
    assert [(r["modality"], r["oct_slice_idx"]) for r in rows] == [
        ("cfp", "-1"), ("oct", "0"), ("oct", "1"), ("cfp_oct", "0"), ("cfp_oct", "1")]
    assert [c[0] for c in stub.calls] == [1, 1, 1, 2, 2]
    assert all(c[1].endswith("no analysis, and no explanation.") for c in stub.calls)  # strict
    assert {r["pred_class"] for r in rows} == {"2"} and {r["true_class"] for r in rows} == {"2"}


def test_skin_case_insensitive_image_path(tmp_path, data_root, monkeypatch):
    stub = StubLocal()
    monkeypatch.setattr(runmod, "load_backend", lambda model: stub)
    args = _args(tmp_path, data_root, experiment="main", dataset="skin", model="medgemma4",
                 input_types=["image_only"])
    runmod.run(args, MODELS["medgemma4"], recipe("main", "skin", "medgemma4"))
    rows = _rows(tmp_path / "out/main/skin/medgemma4.csv")
    assert len(rows) == 2 * 3 and not any(r["error"] for r in rows)


def test_api_sync_placebo_amd(tmp_path, data_root, monkeypatch):
    seen = []

    def client(images, prompt, rec):
        seen.append((len(images), rec.prompt_format, rec.thinking_level))
        return '{"amd_class": 3, "prob_normal": 0, "prob_dry_amd": 0, "prob_wet_amd": 0.2, "prob_pcv": 0.8}'

    monkeypatch.setattr(runmod, "load_backend", lambda model: client)
    args = _args(tmp_path, data_root, experiment="placebo", dataset="amd", model="gemini")
    runmod.run(args, MODELS["gemini"], recipe("placebo", "amd", "gemini"))
    rows = _rows(tmp_path / "out/placebo/amd/gemini.csv")
    assert len(rows) == 3 * 3                          # (CFP + 2 B-scans) x dup/blank/noise
    assert {r["pred_class"] for r in rows} == {"3"}
    assert set(seen) == {(2, "json", "medium")}


def test_batch_build_and_fetch_openai(tmp_path, data_root, monkeypatch):
    args = _args(tmp_path, data_root, experiment="placebo", dataset="skin", model="gpt")
    rec = recipe("placebo", "skin", "gpt")
    batch.build(args, MODELS["gpt"], rec)
    d = tmp_path / "out/placebo/skin/gpt_batch"
    lines = [json.loads(x) for x in (d / "chunk001_requests.jsonl").read_text().splitlines()]
    assert len(lines) == 2 * 6
    body = lines[0]["body"]
    assert body["model"] == "gpt-5.6-sol" and body["temperature"] == 1.0
    assert body["max_completion_tokens"] == 4096
    assert [c["type"] for c in body["messages"][0]["content"]] == ["image_url", "image_url", "text"]
    assert body["messages"][0]["content"][0]["image_url"]["detail"] == "high"

    with pytest.raises(SystemExit):                    # submit refuses without --yes
        batch.submit(args, MODELS["gpt"], rec)

    state = json.loads((d / "state.json").read_text())
    state["chunks"][0]["job"] = "batch_test"
    (d / "state.json").write_text(json.dumps(state))
    results = [{"custom_id": x["custom_id"], "response": {"status_code": 200, "body": {"choices": [
        {"message": {"content": '```json\n{"biopsy": 1, "prob_0": 0.25, "prob_1": 0.75}\n```'},
         "finish_reason": "stop"}]}}} for x in lines[:-1]]
    (d / "chunk001_results.jsonl").write_text("\n".join(json.dumps(r) for r in results))
    monkeypatch.setattr(batch, "_client", lambda model: None)
    monkeypatch.setattr(batch, "_job_state", lambda client, model, job: ("completed", None))
    batch.fetch(args, MODELS["gpt"], rec)
    rows = _rows(tmp_path / "out/placebo/skin/gpt.csv")
    assert len(rows) == 12
    assert sum(r["prob_verbalized"] == "0.75" for r in rows) == 11
    assert sum(r["error"] == "no_result" for r in rows) == 1
