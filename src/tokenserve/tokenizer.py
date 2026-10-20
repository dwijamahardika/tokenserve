"""Tokenizers, incremental detokenization and stop-string handling."""

from __future__ import annotations

from collections.abc import Sequence as Seq
from typing import Protocol

ChatMessage = dict[str, str]


def default_chat_template(messages: Seq[ChatMessage]) -> str:
    """Minimal role-tagged template for models without one of their own."""
    parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    parts.append("<|assistant|>\n")
    return "".join(parts)


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Seq[int]) -> str: ...

    def is_special(self, token_id: int) -> bool: ...

    def encode_chat(self, messages: Seq[ChatMessage]) -> list[int]:
        """Render messages with the model's chat template and tokenize."""
        ...


class ByteTokenizer:
    """UTF-8 byte-level tokenizer: token ``i < 256`` is byte ``i``; 256 is EOS.

    Used with the simulated backend. One token per byte makes prompt lengths
    exact and controllable in benchmarks (an ASCII prompt of N characters is
    N tokens), and multi-byte characters exercise incremental detokenization.
    """

    vocab_size = 257
    eos_token_id: int | None = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Seq[int]) -> str:
        return bytes(t for t in token_ids if t < 256).decode("utf-8", errors="replace")

    def is_special(self, token_id: int) -> bool:
        return token_id >= 256

    def encode_chat(self, messages: Seq[ChatMessage]) -> list[int]:
        return self.encode(default_chat_template(messages))


class IncrementalDetokenizer:
    """Turns a stream of token ids into text deltas.

    Decoding tokens one at a time is wrong for byte-level and BPE vocabularies:
    a character can span several tokens, and some tokenizers change how a token
    renders depending on its left neighbour (leading spaces). The standard fix
    (used by TGI and vLLM) decodes a short window twice — with and without the
    new token — and emits the difference once it no longer ends in a partial
    character (U+FFFD).
    """

    _CONTEXT = 5

    def __init__(self, tokenizer: Tokenizer, prompt_token_ids: Seq[int]) -> None:
        self._tok = tokenizer
        self._ids = list(prompt_token_ids[-self._CONTEXT :])
        self._prefix_offset = 0
        self._read_offset = len(self._ids)

    def add(self, token_id: int) -> str:
        if self._tok.is_special(token_id):
            return ""
        self._ids.append(token_id)
        prefix = self._tok.decode(self._ids[self._prefix_offset : self._read_offset])
        full = self._tok.decode(self._ids[self._prefix_offset :])
        if len(full) <= len(prefix) or full.endswith("�"):
            return ""  # incomplete character: wait for more tokens
        delta = full[len(prefix) :]
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._ids)
        # Keep the window bounded.
        del self._ids[: self._prefix_offset]
        self._read_offset -= self._prefix_offset
        self._prefix_offset = 0
        return delta


class StopStringMatcher:
    """Detects stop strings in streamed text without leaking a partial match.

    A trailing piece of text that is a prefix of some stop string is held
    back, because it might complete in a later token; everything before it is
    safe to stream. On a match the text is truncated before the stop string.
    """

    def __init__(self, stop: Seq[str]) -> None:
        self._stop = tuple(s for s in stop if s)
        self._hold = max((len(s) for s in self._stop), default=1) - 1
        self.text = ""
        self._emitted = 0

    def push(self, delta: str) -> tuple[str, bool]:
        """Add text; return ``(text safe to emit now, stopped)``."""
        search_from = max(0, len(self.text) - self._hold)
        self.text += delta
        if self._stop:
            hits = [i for s in self._stop if (i := self.text.find(s, search_from)) != -1]
            if hits:
                self.text = self.text[: min(hits)]
                return self.flush(), True
        safe = len(self.text) - self._partial_match_len()
        if safe <= self._emitted:
            return "", False
        out = self.text[self._emitted : safe]
        self._emitted = safe
        return out, False

    def _partial_match_len(self) -> int:
        """Length of the longest suffix of the text that is a proper prefix of a stop."""
        best = 0
        for stop in self._stop:
            for k in range(min(len(stop) - 1, len(self.text)), best, -1):
                if self.text.endswith(stop[:k]):
                    best = k
                    break
        return best

    def flush(self) -> str:
        out = self.text[self._emitted :]
        self._emitted = len(self.text)
        return out
