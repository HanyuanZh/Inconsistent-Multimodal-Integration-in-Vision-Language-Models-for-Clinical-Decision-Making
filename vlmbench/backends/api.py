"""Gemini and OpenAI requests.

Request bodies are built in one place and used both for synchronous calls and
for Batch API files, so the two paths send identical content.  Images precede
the text, as in the study.  API keys are read from the environment only:
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import base64
import os
import time

from ..images import api_bytes

SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ----------------------------------------------------------------------- Gemini
def gemini_request(images, prompt, recipe) -> dict:
    """Body of one Gemini generateContent request (Batch API ``request`` field).

    Safety filters are switched off: medical images otherwise trip them and the
    model refuses instead of answering.
    """
    parts = []
    for item in images:
        data, mime = api_bytes(item)
        parts.append({"inline_data": {"mime_type": mime, "data": _b64(data)}})
    parts.append({"text": prompt})
    gen = {"max_output_tokens": recipe.max_new_tokens}
    if recipe.thinking_level:
        gen["thinking_config"] = {"thinking_level": recipe.thinking_level}
    if recipe.temperature is not None:
        gen["temperature"] = recipe.temperature
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generation_config": gen,
        "safety_settings": [{"category": c, "threshold": "BLOCK_NONE"} for c in SAFETY_CATEGORIES],
    }


def gemini_response_text(response: dict) -> str:
    """Answer text of a generateContent response (REST/JSON form).

    Thinking parts are skipped.  When there is no text -- e.g. the whole output
    budget went to thinking -- a diagnostic string with the finish reason is
    returned so the row records why it is empty.
    """
    try:
        cand = response["candidates"][0]
    except (KeyError, IndexError, TypeError):
        return f"<empty response; prompt_feedback={response.get('promptFeedback') if isinstance(response, dict) else None}>"
    texts = [p.get("text", "") for p in cand.get("content", {}).get("parts", [])
             if not p.get("thought")]
    text = "".join(texts).strip()
    if text:
        return text
    reason = cand.get("finishReason") or cand.get("finish_reason")
    return f"<empty response; finish_reason={reason}>"


class GeminiClient:
    def __init__(self, model_id, max_retries=5, timeout_s=120):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY (or GOOGLE_API_KEY)")
        self.types = types
        self.client = genai.Client(api_key=key,
                                   http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.model_id, self.max_retries = model_id, max_retries

    def __call__(self, images, prompt, recipe) -> str:
        t = self.types
        req = gemini_request(images, prompt, recipe)
        contents = [t.Part.from_bytes(data=base64.b64decode(p["inline_data"]["data"]),
                                      mime_type=p["inline_data"]["mime_type"])
                    for p in req["contents"][0]["parts"] if "inline_data" in p]
        contents.append(t.Part.from_text(text=prompt))
        gen = req["generation_config"]
        config = t.GenerateContentConfig(
            max_output_tokens=gen["max_output_tokens"],
            temperature=gen.get("temperature"),
            thinking_config=(t.ThinkingConfig(thinking_level=gen["thinking_config"]["thinking_level"])
                             if "thinking_config" in gen else None),
            safety_settings=[t.SafetySetting(category=s["category"], threshold=s["threshold"])
                             for s in req["safety_settings"]],
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(model=self.model_id, contents=contents,
                                                           config=config)
                if resp.text:   # the SDK skips thought parts here
                    return resp.text.strip()
                try:
                    reason = resp.candidates[0].finish_reason
                except (AttributeError, IndexError, TypeError):
                    return f"<empty response; prompt_feedback={resp.prompt_feedback}>"
                return f"<empty response; finish_reason={getattr(reason, 'name', reason)}>"
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code in (400, 401, 403, 404) or attempt == self.max_retries:
                    raise
                time.sleep(min(60, 2 ** attempt))


# ----------------------------------------------------------------------- OpenAI
def openai_body(images, prompt, recipe, model_id) -> dict:
    """Body of one /v1/chat/completions request (Batch API ``body`` field)."""
    content = []
    for item in images:
        data, mime = api_bytes(item)
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{_b64(data)}",
                                      "detail": recipe.image_detail}})
    content.append({"type": "text", "text": prompt})
    body = {"model": model_id, "messages": [{"role": "user", "content": content}],
            "temperature": recipe.temperature, "max_completion_tokens": recipe.max_new_tokens}
    if recipe.reasoning_effort:
        body["reasoning_effort"] = recipe.reasoning_effort
    return body


def openai_response_text(body: dict) -> str:
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError):
        return f"<empty response; body={str(body)[:200]}>"
    text = (choice.get("message", {}).get("content") or "").strip()
    return text or f"<empty response; finish_reason={choice.get('finish_reason')}>"


class OpenAIClient:
    def __init__(self, model_id, max_retries=5, timeout_s=120):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("set OPENAI_API_KEY")
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        self.model_id, self.max_retries, self.timeout_s = model_id, max_retries, timeout_s

    def __call__(self, images, prompt, recipe) -> str:
        body = openai_body(images, prompt, recipe, self.model_id)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**body, timeout=self.timeout_s)
                return openai_response_text(resp.model_dump())
            except Exception as exc:  # noqa: BLE001
                if getattr(exc, "status_code", None) in (400, 401, 402, 403, 404) \
                        or attempt == self.max_retries:
                    raise
                time.sleep(min(60, 2 ** attempt))


def client_for(model, **kw):
    """``model`` is a :class:`vlmbench.recipes.Model`."""
    return (GeminiClient if model.family == "gemini" else OpenAIClient)(model.checkpoint, **kw)
