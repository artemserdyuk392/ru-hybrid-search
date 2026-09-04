"""Split documents into overlapping chunks on paragraph/sentence boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 1500
OVERLAP = 200

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_id: int
    chunk_index: int
    text: str
    title: str
    source: str | None = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "title": self.title,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(
            doc_id=d["doc_id"],
            chunk_index=d["chunk_index"],
            text=d["text"],
            title=d["title"],
            source=d.get("source"),
        )


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _pieces(text: str, max_chars: int) -> list[str]:
    """Break text into units no longer than max_chars, coarsest first."""
    pieces = []
    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sent in _SENTENCE_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_chars:
                pieces.append(sent)
            else:
                pieces.extend(_hard_split(sent, max_chars))
    return pieces


def _joined_len(pieces: list[str]) -> int:
    return sum(len(p) for p in pieces) + max(0, len(pieces) - 1)


def _overlap_tail(pieces: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for piece in reversed(pieces):
        if total + len(piece) > overlap:
            # a single trailing piece longer than the overlap: keep only its end,
            # otherwise the next chunk would inherit almost a whole chunk of text
            if not tail:
                tail.append(piece[-overlap:])
            break
        tail.append(piece)
        total += len(piece)
    tail.reverse()
    return tail


def chunk_document(
    doc_id: int,
    text: str,
    title: str = "",
    source: str | None = None,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    pieces = _pieces(text, max_chars)
    if not pieces:
        return []

    texts: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in pieces:
        extra = len(piece) + (1 if cur else 0)
        if cur and cur_len + extra > max_chars:
            texts.append(" ".join(cur))
            cur = _overlap_tail(cur, overlap)
            cur_len = _joined_len(cur)
            extra = len(piece) + (1 if cur else 0)
        cur.append(piece)
        cur_len += extra
    if cur:
        texts.append(" ".join(cur))

    return [
        Chunk(doc_id=doc_id, chunk_index=i, text=t, title=title, source=source)
        for i, t in enumerate(texts)
    ]
