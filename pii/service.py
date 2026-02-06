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
    """
    Load and cache the HF NER pipeline once per process.
    This can be heavy; caching avoids re-loading on every file.
    """
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
        out.append(
            PiiSpan(
                start=start,
                end=end,
                label=s.tag,
                text=text[start:end],
                priority=getattr(s, "priority", 0),
            )
        )
    return out


def detect_pii(
    text: str,
    *,
    use_ner: bool = True,
    use_regex: bool = True,
    resolve_overlaps: bool = True,
) -> List[PiiSpan]:
    """
    Detect PII spans in `text`.

    Returns spans with offsets into the original `text`.
    Spans are non-overlapping if resolve_overlaps=True (recommended).
    """
    if not text:
        return []

    spans: List[yj.Span] = []

    if use_regex:
        spans.extend(yj.regex_spans(text))

    if use_ner:
        ner_pipe = _get_ner_pipe()
        spans.extend(yj.ner_spans(ner_pipe, text))

    if resolve_overlaps:
        spans = yj.resolve(spans)

    # Ensure consistent ordering (start ascending)
    spans = sorted(spans, key=lambda s: (s.start, s.end))

    return _to_pii_spans(text, spans)


PlaceholderStyle = Literal["tag", "stars", "fixed", "keep_length"]


def redact(
    text: str,
    spans: List[PiiSpan],
    *,
    style: PlaceholderStyle = "tag",
    fixed_placeholder: str = "[REDACTED]",
) -> str:
    """
    Apply redactions to text using PiiSpan offsets.

    styles:
      - "tag":        replaces with "[<label>]" e.g. "[email]"
      - "stars":      replaces with "***" (keeps no info)
      - "fixed":      replaces with fixed_placeholder (default "[REDACTED]")
      - "keep_length": replaces with '█' repeated to original span length
    """
    if not text or not spans:
        return text

    # Sort right-to-left so replacements don't shift earlier offsets
    spans_sorted = sorted(spans, key=lambda s: (s.start, s.end), reverse=True)

    out = text
    n = len(out)

    for s in spans_sorted:
        start = max(0, min(n, s.start))
        end = max(0, min(n, s.end))
        if end <= start:
            continue

        if style == "tag":
            repl = f"[{s.label}]"
        elif style == "stars":
            repl = "***"
        elif style == "fixed":
            repl = fixed_placeholder
        elif style == "keep_length":
            repl = "█" * (end - start)
        else:
            # Fallback
            repl = f"[{s.label}]"

        out = out[:start] + repl + out[end:]
        n = len(out)  # update length after replacement

    return out
