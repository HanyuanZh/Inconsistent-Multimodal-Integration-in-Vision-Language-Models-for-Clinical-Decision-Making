"""Local vision-language models (Hugging Face), decoded greedily.

Each family reproduces the input construction of the original study runners:
chat template, image placement (images before the text), dtype and the extra
``generate`` arguments.  Two calls are exposed:

``generate(images, prompt, max_new_tokens)``  free text (verbalised probability)
``label_probs(images, prompt, labels)``        one constrained token; softmax over
                                               the tokens that decode to each label
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList

DTYPE = torch.bfloat16


class _RestrictTo(LogitsProcessor):
    def __init__(self, token_ids):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        allowed = self.token_ids.to(scores.device)
        mask[:, allowed] = scores[:, allowed]
        return mask


def _label_token_ids(tokenizer, labels):
    """label -> single-token ids that decode to it, with or without a leading space."""
    ids = {}
    for label in labels:
        found = []
        for s in (label, " " + label):
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1 and tokenizer.decode(enc).strip() == label:
                found.append(enc[0])
        if not found:
            raise ValueError(f"no single-token encoding for label {label!r}")
        ids[label] = sorted(set(found))
    return ids


class LocalVLM:
    """Shared generation logic; subclasses build the model inputs."""

    pass_pad_token = False   # MedGemma passes pad_token_id=eos to generate()

    def __init__(self, checkpoint: str, revision=None):
        self.checkpoint = checkpoint
        self.hub = {"revision": revision} if revision else {}
        self._label_cache = {}

    # subclasses: return (inputs, input_length) ready for model.generate(**inputs)
    def _inputs(self, images, prompt):
        raise NotImplementedError

    def _decode(self, new_ids) -> str:
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _generate(self, inputs, **kw):
        if self.pass_pad_token:
            kw["pad_token_id"] = self.tokenizer.eos_token_id
        return self.model.generate(**inputs, do_sample=False, **kw)

    @torch.inference_mode()
    def generate(self, images, prompt, max_new_tokens: int) -> str:
        inputs, n = self._inputs(images, prompt)
        out = self._generate(inputs, max_new_tokens=max_new_tokens)
        text = self._decode(out[0][n:])
        del inputs, out
        torch.cuda.empty_cache()
        return text

    @torch.inference_mode()
    def label_probs(self, images, prompt, labels=("0", "1")):
        """(raw token, {label: probability}) from one constrained decoding step."""
        if labels not in self._label_cache:
            ids = _label_token_ids(self.tokenizer, labels)
            allowed = sorted({i for v in ids.values() for i in v})
            self._label_cache[labels] = (ids, LogitsProcessorList([_RestrictTo(allowed)]), allowed)
        ids, processor, allowed = self._label_cache[labels]
        inputs, n = self._inputs(images, prompt)
        out = self._generate(inputs, max_new_tokens=1, logits_processor=processor,
                             return_dict_in_generate=True, output_scores=True)
        scores = out.scores[0][0]
        probs_t = torch.softmax(scores[torch.tensor(allowed, device=scores.device)].float(), dim=-1)
        by_label = {label: 0.0 for label in labels}
        for tid, p in zip(allowed, probs_t.cpu().tolist()):
            by_label[self.tokenizer.decode([tid]).strip()] += float(p)
        total = sum(by_label.values())
        probs = {k: v / total for k, v in by_label.items()} if total > 0 else None
        raw = self._decode(out.sequences[0][n:])
        del inputs, out
        torch.cuda.empty_cache()
        return raw, probs


def _messages(images, prompt):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class MedGemma(LocalVLM):
    pass_pad_token = True

    def __init__(self, checkpoint, revision=None):
        super().__init__(checkpoint, revision)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, torch_dtype=DTYPE, device_map="auto", **self.hub).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint, **self.hub)
        self.tokenizer = self.processor.tokenizer

    def _inputs(self, images, prompt):
        inputs = self.processor.apply_chat_template(
            _messages(images, prompt), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs, inputs["input_ids"].shape[-1]


class _QwenStyle(LocalVLM):
    """Qwen-VL chat template + qwen_vl_utils.process_vision_info."""

    def _template(self, messages):
        return self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)

    def _inputs(self, images, prompt):
        from qwen_vl_utils import process_vision_info
        messages = _messages(images, prompt)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[self._template(messages)], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.device)
        return inputs, inputs.input_ids.shape[-1]

    def _decode(self, new_ids) -> str:
        return self.processor.batch_decode(
            [new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


class Qwen36(_QwenStyle):
    def __init__(self, checkpoint, revision=None):
        super().__init__(checkpoint, revision)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True,
                                                       **self.hub)
        # SDPA attention, as in the study (its environment had no flash-attn);
        # a different kernel changes the numerics and can change greedy answers.
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, torch_dtype=DTYPE, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="sdpa", trust_remote_code=True, **self.hub).eval()
        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device

    def _template(self, messages):
        try:   # answer directly, without the thinking block
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return super()._template(messages)


class Lingshu(_QwenStyle):
    def __init__(self, checkpoint, revision=None):
        super().__init__(checkpoint, revision)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=DTYPE, device_map="auto", **self.hub).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint, **self.hub)
        self.tokenizer = self.processor.tokenizer
        self.device = self.model.device


class DeepSeekVL2(LocalVLM):
    """DeepSeek-VL2 needs the ``deepseek_vl2`` package and transformers 4.38."""

    def __init__(self, checkpoint, revision=None):
        super().__init__(checkpoint, revision)
        from deepseek_vl2.models import DeepseekVLV2Processor
        from transformers import AutoModelForCausalLM
        self.processor = DeepseekVLV2Processor.from_pretrained(checkpoint, **self.hub)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True, torch_dtype=DTYPE, **self.hub).cuda().eval()

    def _inputs(self, images, prompt):
        conversation = [
            {"role": "<|User|>", "content": "<image>\n" * len(images) + prompt,
             "images": [f"image_{i}" for i in range(len(images))]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prep = self.processor(conversations=conversation, images=images, force_batchify=True,
                              system_prompt="").to(self.model.device, dtype=DTYPE)
        inputs = dict(inputs_embeds=self.model.prepare_inputs_embeds(**prep),
                      input_ids=prep.input_ids, images=prep.images,
                      images_seq_mask=prep.images_seq_mask,
                      images_spatial_crop=prep.images_spatial_crop,
                      attention_mask=prep.attention_mask, past_key_values=None)
        return inputs, len(prep.input_ids[0])

    def _generate(self, inputs, **kw):
        tok = self.tokenizer
        out = self.model.generate(**inputs, pad_token_id=tok.eos_token_id,
                                  bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                                  do_sample=False, use_cache=True, return_dict_in_generate=True,
                                  **{k: v for k, v in kw.items() if k != "return_dict_in_generate"})
        return out if kw.get("output_scores") else out.sequences

    def _decode(self, new_ids) -> str:
        return self.tokenizer.decode(new_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate(self, images, prompt, max_new_tokens: int) -> str:
        inputs, n = self._inputs(images, prompt)
        seq = self._generate(inputs, max_new_tokens=max_new_tokens)[0]
        # with inputs_embeds, generate() may return only the new tokens
        text = self._decode(seq[n:] if seq.shape[-1] > n else seq)
        del inputs, seq
        torch.cuda.empty_cache()
        return text

    @torch.inference_mode()
    def label_probs(self, images, prompt, labels=("0", "1")):
        if labels not in self._label_cache:
            ids = _label_token_ids(self.tokenizer, labels)
            allowed = sorted({i for v in ids.values() for i in v})
            self._label_cache[labels] = (ids, LogitsProcessorList([_RestrictTo(allowed)]), allowed)
        _, processor, allowed = self._label_cache[labels]
        inputs, n = self._inputs(images, prompt)
        out = self._generate(inputs, max_new_tokens=1, logits_processor=processor,
                             output_scores=True)
        scores = out.scores[0][0]
        probs_t = torch.softmax(scores[torch.tensor(allowed, device=scores.device)].float(), dim=-1)
        by_label = {label: 0.0 for label in labels}
        for tid, p in zip(allowed, probs_t.cpu().tolist()):
            by_label[self.tokenizer.decode([tid]).strip()] += float(p)
        total = sum(by_label.values())
        seq = out.sequences[0]
        raw = self._decode(seq[n:] if seq.shape[-1] > n else seq)
        del inputs, out
        torch.cuda.empty_cache()
        return raw, ({k: v / total for k, v in by_label.items()} if total > 0 else None)


FAMILIES = {"medgemma": MedGemma, "qwen": Qwen36, "lingshu": Lingshu, "deepseek": DeepSeekVL2}


def load(model) -> LocalVLM:
    """``model`` is a :class:`vlmbench.recipes.Model`; ``checkpoint`` may be a local path."""
    return FAMILIES[model.family](model.checkpoint, model.revision)
