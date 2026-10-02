"""Append-only CSV output with resume."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def done_keys(path, key_columns):
    """Keys of the rows already in ``path`` (as tuples of strings)."""
    path = Path(path)
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {tuple(row[c] for c in key_columns) for row in csv.DictReader(f)}


def unit_key(row, key_columns):
    return tuple("" if row.get(c) is None else str(row.get(c)) for c in key_columns)


class RowWriter:
    def __init__(self, path, columns, flush_every=20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=columns, extrasaction="ignore")
        if new:
            self._w.writeheader()
        self._n, self._every = 0, flush_every

    def write(self, row):
        self._w.writerow(row)
        self._n += 1
        if self._n % self._every == 0:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self):
        self._fh.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
