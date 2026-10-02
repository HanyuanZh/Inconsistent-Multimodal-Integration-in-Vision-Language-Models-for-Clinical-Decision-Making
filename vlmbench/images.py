"""Turn the items of ``Unit.images`` (file paths or placebo placeholders) into model input."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from PIL import Image

from .placebo import Placeholder, png_bytes


def load_rgb(item):
    """PIL RGB image for a local model."""
    if isinstance(item, Placeholder):
        return item.render()
    with Image.open(item) as im:
        return im.convert("RGB").copy()


def api_bytes(item):
    """(bytes, mime type) for an API request: real images as their file bytes,
    generated placebo images as PNG."""
    if isinstance(item, Placeholder):
        return png_bytes(item.render()), "image/png"
    path = Path(item)
    mime, _ = mimetypes.guess_type(str(path))
    return path.read_bytes(), mime or "image/png"
