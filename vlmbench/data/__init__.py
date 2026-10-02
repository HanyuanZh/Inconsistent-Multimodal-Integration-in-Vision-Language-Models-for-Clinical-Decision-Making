"""Dataset readers.  Each yields :class:`Unit` objects: the images and metadata of one request."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Unit:
    """One request: the images in order, plus metadata copied to the output row."""

    key: tuple                  # resume key, unique within one output file
    images: list                # Path objects or PIL images (generated placebo images)
    meta: dict = field(default_factory=dict)
    prompt_args: dict = field(default_factory=dict)  # passed to the dataset's prompt builder
    error: Optional[str] = None  # set when the unit cannot be run (e.g. missing clinical value)


def resolve_case_insensitive(path: Path) -> Path:
    """Return ``path``, or the one existing path that differs from it only in letter case.

    The official Derm7pt ``meta.csv`` refers to ``FCl/Fcl068.jpg`` (case 816) while the
    directory on disk is ``FCL``; on a case-sensitive file system that one reference
    would otherwise fail.
    """
    if path.exists():
        return path
    parts = path.parts
    probe = Path(parts[0])
    for part in parts[1:]:
        candidate = probe / part
        if not candidate.exists():
            if not probe.is_dir():
                return path
            matches = [p for p in probe.iterdir() if p.name.lower() == part.lower()]
            if len(matches) != 1:
                return path
            candidate = matches[0]
        probe = candidate
    return probe
