# pii/service.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Literal, Optional
from . import yj_pii_marker as yj


@dataclass(frozen=True)
class PiiSpan:
    start: int
    end: int
    label: str
    text: str
    priority: int = 0


@lru_cache(maxsize=1)
def _get_ner_pipe():
    # Load and cache the HF NER pipeline once per process.
    # This can be heavy; caching avoids re-loading on every file.
    device_hf, device_st = yj.detect_device()
    ner_pipe, _embedder_unused = yj.load_models(device_hf, device_st)
    return ner_pipe


def _to_pii_spans(text: str, spans: Iterable[yj.Span]) -> List[PiiSpan]:
    out: List[PiiSpan] = []
    for s in spans:
        # Guard against out-of-range spans (shouldn't happen, but keep it robust)
        start = max(0, min(len(text), s.start))
        end = max(0, min(len(text), s.end))
        if end <= start:
            continue
