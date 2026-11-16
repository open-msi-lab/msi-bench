"""Append-only LLM response cache / generation transcript.

A single artifact serves two purposes:

* **Cache:** identical generation requests are memoised, so re-running
  ``build`` — the ``all`` pipeline, the schema-aligned *and* degraded brackets,
  a post-tweak rebuild, or any multi-seed replay of the *same* seed — replays
  stored text instead of re-calling (and re-paying for) the API. Because the
  cached text is returned verbatim, a fixed seed reproduces the *exact*
  benchmark, which a live ``temperature``-0.8 generator could never do.
* **Transcript:** every real (cache-miss) generation appends one JSON line
  recording the model, prompt, response, and timing — a provenance log for
  fidelity analysis (ROUGE / length / lexical-diversity studies) and the
  datasheet.

The store is a newline-delimited JSON file written **incrementally** (one line
per generation, flushed immediately). That makes it streamable (``tail -f`` the
file to watch a long build), crash-safe, and resumable: a build that dies
half-way leaves its completed generations on disk, and the next run replays them
for free and continues.

Cache key = SHA-256 over ``provider | model | temperature | max_tokens |
system | user | salt`` (+ an optional ``extra`` field, see below). The optional
``salt`` lets a caller give each *logical* draw its own slot even when the prompt
is identical (see ``generation.benchmark``): without it, the several genuine
reports of one event that happen to share a (hazard, expertise, quality) prompt
would collapse to a single response and erase the very lexical diversity the
overlay is meant to introduce.

The key is a pure function of request content, so it is **independent of the
machine and of the generation concurrency**: a cache built on the networked PC
replays bit-for-bit on an offline laptop, and ``llm.concurrency`` only reorders
the writes, never the keys. A hit therefore implies the request was identical in
every output-affecting respect — provider, model, sampling params, the full
prompt (which carries the seed-derived draws and the raw-corpus context), the
per-draw salt (which carries ``s{seed}`` and the event id), **and** the
``extra`` fingerprint of any remaining generation knobs a client exposes (e.g. a
Responses model's ``reasoning_effort`` / ``min_output_tokens``). ``extra`` is
appended only when non-empty, so providers without such knobs keep the exact
keys of the original 7-field scheme.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def _est_tokens(text: str) -> int:
    """Whitespace-token estimate (cheap; not a billed-token count).

    The full response text is stored, so an exact count can be recomputed
    offline with the model's own tokenizer if a study needs it.
    """
    return len((text or "").split())


class ResponseCache:
    """Disk-backed, append-only memo of LLM completions.

    Parameters
    ----------
    path:
        JSONL file. Created on first write; its parent must exist.
    enabled:
        When false the object is inert (no reads, no writes) so callers need not
        special-case a disabled cache.
    """

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._mem: Dict[str, str] = {}
        self._lock = threading.Lock()  # serialises concurrent puts (parallel build)
        if self.enabled and self.path.is_file():
            self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        """Populate the in-memory map from the JSONL log (last write wins)."""
        for line in self.path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted write
            key, resp = rec.get("key"), rec.get("response")
            if isinstance(key, str) and isinstance(resp, str):
                self._mem[key] = resp

    # ------------------------------------------------------------------- key
    @staticmethod
    def make_key(provider: str, model: str, temperature: float, max_tokens: int,
                 system: str, user: str, salt: Optional[str],
                 extra: str = "") -> str:
        h = hashlib.sha256()
        parts = [provider, model, f"{float(temperature):.6f}", str(int(max_tokens)),
                 system, user, salt or ""]
        if extra:
            # Generation-affecting llm-config knobs not captured by the fields
            # above (e.g. a Responses model's reasoning_effort / effective output
            # budget). Appended only when present, so a hit stays honest about
            # config while providers without such knobs keep their original keys.
            parts.append(extra)
        for part in parts:
            h.update(part.encode("utf-8"))
            h.update(b"\x00")  # unambiguous field separator
        return h.hexdigest()

    # --------------------------------------------------------------- lookup
    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        hit = self._mem.get(key)  # dict reads are atomic under the GIL
        if hit is not None:
            with self._lock:
                self.hits += 1
        return hit

    def put(self, key: str, response: str, meta: Dict) -> None:
        """Record a fresh generation: update memory and append one JSONL line.

        Thread-safe: the in-memory update, counter, and file append are guarded
        by a lock so concurrent (parallel-build) writers never interleave lines.
        """
        line = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "key": key,
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "temperature": meta.get("temperature"),
            "max_tokens": meta.get("max_tokens"),
            "salt": meta.get("salt"),
            "system": meta.get("system"),
            "user": meta.get("user"),
            "response": response,
            "est_tokens": _est_tokens(response),
        }
        with self._lock:
            self._mem[key] = response
            self.misses += 1
            if not self.enabled:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                fh.flush()  # stream to disk so a tail -f / crash sees each generation

    def __len__(self) -> int:
        return len(self._mem)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ResponseCache(path={self.path}, enabled={self.enabled}, "
                f"entries={len(self._mem)}, hits={self.hits}, misses={self.misses})")
