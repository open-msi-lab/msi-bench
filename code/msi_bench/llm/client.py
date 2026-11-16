"""LLM client abstraction.

A uniform ``complete(system, user, temperature, max_tokens) -> str`` interface
over Anthropic, OpenAI / OpenAI-compatible endpoints, Google Gemini,
Anthropic- and OpenAI-compatible gateways (proxy providers), and a
deterministic mock.
Generators check ``client.is_mock`` and, when true, synthesise templated text
locally instead of calling ``complete`` — this keeps the whole pipeline runnable
with no API key while preserving channel separation (code owns all metadata and
coordinates regardless of which client is used).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..config import Config
from .cache import ResponseCache


@dataclass
class LLMClient:
    """Base client. Subclasses implement :meth:`_complete` (the raw API call).

    :meth:`complete` is a concrete wrapper that transparently memoises responses
    through the optional :class:`~msi_bench.llm.cache.ResponseCache` (set on
    ``cache``): a cache hit returns stored text without an API call, a miss calls
    ``_complete`` and records the result. The mock client and a ``None`` cache
    bypass it. ``cache_salt`` gives a caller's *logical* draw its own cache slot
    even when the prompt is identical (see ``generation.benchmark``).
    """

    provider: str
    model: str
    temperature: float = 0.8
    is_mock: bool = False
    cache: Optional[ResponseCache] = None
    # Canonical fingerprint of generation-affecting config knobs not already in
    # the cache key (provider/model/temperature/max_tokens). Set by ``get_client``
    # from the llm spec, so the real and offline clients agree and a config change
    # that alters output also busts the cache. Empty for clients without extras.
    key_extra: str = ""

    def complete(self, system: str, user: str, max_tokens: int = 512,
                 temperature: Optional[float] = None,
                 cache_salt: Optional[str] = None) -> str:
        temp = self.temperature if temperature is None else temperature
        cache = self.cache
        if cache is None or self.is_mock:
            return self._complete(system, user, max_tokens, temp)
        key = cache.make_key(self.provider, self.model, temp, max_tokens,
                             system, user, cache_salt, self.key_extra)
        hit = cache.get(key)
        if hit is not None:
            return hit
        resp = self._complete(system, user, max_tokens, temp)
        cache.put(key, resp, {
            "provider": self.provider, "model": self.model, "temperature": temp,
            "max_tokens": max_tokens, "salt": cache_salt, "system": system, "user": user,
        })
        return resp

    def _complete(self, system: str, user: str, max_tokens: int,
                  temperature: float) -> str:
        """Perform the actual API call. ``temperature`` is already resolved."""
        raise NotImplementedError

    @property
    def tag(self) -> str:
        return f"{self.provider}:{self.model}"


class MockClient(LLMClient):
    """Deterministic offline client.

    Returns a seeded, lightly-varied paraphrase so that pipelines depending on
    the generic interface still run. Already deterministic and free, so it is
    not routed through the response cache. In practice the generators use their
    own templated paths when ``is_mock`` is true.
    """

    def __init__(self, model: str = "mock-1", temperature: float = 0.8) -> None:
        super().__init__(provider="mock", model=model, temperature=temperature, is_mock=True)

    def _complete(self, system: str, user: str, max_tokens: int,
                  temperature: float) -> str:
        seed = int(hashlib.sha256((system + user).encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)
        openers = ["Observed", "Sighted", "Reporting", "Noted", "Spotted"]
        return f"{rng.choice(openers)}: {user[:200]}".strip()


class OfflineCacheMiss(RuntimeError):
    """Raised in offline mode when a request is absent from the replayed cache."""


class OfflineCacheClient(LLMClient):
    """Cache-only client for cross-machine replay (``llm.offline: true``).

    Replays an ``llm_cache.jsonl`` generated elsewhere (e.g. on a networked
    machine) with **no API key, SDK, or network**: a hit returns the stored text; a
    miss raises :class:`OfflineCacheMiss` with an actionable message rather than
    calling out or silently falling back to templated mock text. The
    provider/model/temperature must equal those used to *generate* the cache so
    the keys line up — keep the ``llm`` block identical across machines; only
    ``offline`` differs (it is not part of the cache key).
    """

    def complete(self, system, user, max_tokens=512, temperature=None, cache_salt=None):
        temp = self.temperature if temperature is None else temperature
        cache = self.cache
        if cache is None:
            raise RuntimeError("llm.offline=true requires a response cache to replay.")
        key = cache.make_key(self.provider, self.model, temp, max_tokens,
                             system, user, cache_salt, self.key_extra)
        hit = cache.get(key)
        if hit is not None:
            return hit
        raise OfflineCacheMiss(
            f"offline mode: no cached response for {self.provider}:{self.model} "
            f"salt={cache_salt!r} (key {key[:12]}…). The cache at {cache.path} has "
            f"{len(cache)} entries but not this one, so it is incomplete for this "
            f"build. Regenerate on the networked machine with the SAME seed + llm "
            f"config, re-copy {cache.path.name}, or set llm.offline=false.")

    def _complete(self, system, user, max_tokens, temperature):  # pragma: no cover
        raise NotImplementedError  # complete() fully handles the offline path


class AnthropicClient(LLMClient):  # pragma: no cover - requires network + key
    def __init__(self, model: str, api_key: str, temperature: float = 0.8) -> None:
        super().__init__(provider="anthropic", model=model, temperature=temperature)
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def _complete(self, system, user, max_tokens, temperature):
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        if isinstance(resp, str):
            # Some proxy deployments respond as raw SSE text; parse deltas directly.
            return _anthropic_sse_text(resp)
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class OpenAIClient(LLMClient):  # pragma: no cover - requires network + key
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 temperature: float = 0.8) -> None:
        super().__init__(provider="openai", model=model, temperature=temperature)
        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url or None)

    def _complete(self, system, user, max_tokens, temperature):
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class GeminiClient(LLMClient):  # pragma: no cover - requires network + key
    """Google Gemini client (new ``google-genai`` SDK).

    The system prompt is passed as ``system_instruction``; the user prompt is the
    content. Requires ``pip install google-genai`` and a Gemini API key.
    """

    def __init__(self, model: str, api_key: str, temperature: float = 0.8) -> None:
        super().__init__(provider="gemini", model=model, temperature=temperature)
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _complete(self, system, user, max_tokens, temperature):
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text or ""


# --- Anthropic- and OpenAI-compatible gateway clients ------------------------
# The providers "anthropic_proxy", "openai_responses" and "openai_proxy" use the
# standard Anthropic / OpenAI SDK clients. An optional ``base_url`` (provider spec,
# else the ``LLM_BASE_URL`` environment variable) points them at a compatible
# gateway; without it the official endpoint is used. The full REST paths both
# contain "/v1/" (".../v1/messages" and ".../v1/responses"), but the two SDKs
# split the URL differently: the Anthropic SDK appends the whole "/v1/messages"
# to its base_url, whereas the OpenAI SDK appends only "/responses" to a
# "/v1"-suffixed base_url. So the Anthropic base must NOT carry "/v1" and the
# OpenAI base MUST. The normalisers below accept either form, so a stray (or
# missing) "/v1" in an override can't produce "/v1/v1/messages" or drop the
# version segment.


def _iter_sse_json_events(payload: str):
    """Yield parsed JSON objects from ``text/event-stream`` payload text."""
    for line in payload.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except Exception:  # noqa: BLE001 - best effort parsing
            continue
        if isinstance(event, dict):
            yield event


def _anthropic_sse_text(payload: str) -> str:
    """Extract assistant text from an Anthropic-style SSE payload."""
    parts = []
    for event in _iter_sse_json_events(payload):
        etype = event.get("type")
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str):
                    parts.append(text)
        elif etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _openai_response_text_dict(response_obj: Dict) -> str:
    """Extract text from an OpenAI ``response`` object represented as a dict."""
    parts = []
    for item in response_obj.get("output") or []:
        for block in item.get("content") or []:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _openai_sse_text(payload: str) -> str:
    """Extract assistant text from an OpenAI Responses-style SSE payload."""
    parts = []
    completed_text = ""
    for event in _iter_sse_json_events(payload):
        etype = event.get("type")
        if etype == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                parts.append(delta)
        elif etype == "response.completed":
            completed = event.get("response") or {}
            if isinstance(completed, dict):
                completed_text = _openai_response_text_dict(completed)
    joined = "".join(parts)
    return joined or completed_text


def _anthropic_proxy_base(url: str) -> str:
    """Base URL for the Anthropic SDK, which appends ``/v1/messages`` itself.

    Strips a trailing ``/v1`` if an override carried one, otherwise the SDK
    would request ``/v1/v1/messages``.
    """
    u = url.rstrip("/")
    return u[: -len("/v1")] if u.endswith("/v1") else u


def _responses_proxy_base(url: str) -> str:
    """Base URL for the OpenAI Responses SDK, which appends ``/responses``.

    The OpenAI SDK expects ``/v1`` to live in the base URL, so add it back when
    an override omitted it.
    """
    u = url.rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


class AnthropicProxyClient(AnthropicClient):  # pragma: no cover - requires network + key
    """Anthropic models via an Anthropic-compatible gateway (proxy).

    Same wire protocol as the public Anthropic API — identical ``x-api-key``
    auth and ``/v1/messages`` path, so :meth:`AnthropicClient._complete` is
    reused verbatim — but the SDK's ``base_url`` can point at a gateway.
    Without ``base_url`` the official endpoint is used.
    """

    def __init__(self, model: str, api_key: str,
                 base_url: Optional[str] = None, temperature: float = 0.8) -> None:
        # Initialise the shared LLMClient fields directly (the grandparent) so
        # we can inject ``base_url`` without modifying AnthropicClient.__init__.
        LLMClient.__init__(self, provider="anthropic_proxy", model=model,
                           temperature=temperature)
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=_anthropic_proxy_base(base_url) if base_url else None,
        )


def _responses_output_text(resp) -> str:
    """Extract the assistant text from an OpenAI Responses API object.

    Prefers the ``output_text`` convenience aggregator; falls back to walking
    ``output[*].content[*].text`` for SDK builds that lack it.
    """
    if isinstance(resp, str):
        # Some proxy deployments respond as raw SSE text; parse deltas directly.
        return _openai_sse_text(resp)

    text = getattr(resp, "output_text", None)
    if text:
        return text
    parts = []
    for item in getattr(resp, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            chunk = getattr(block, "text", None)
            if chunk:
                parts.append(chunk)
    return "".join(parts)


class OpenAIResponsesClient(LLMClient):  # pragma: no cover - requires network + key
    """GPT via an OpenAI-compatible gateway (proxy).

    Targets the OpenAI **Responses** endpoint (``/responses``); an optional
    ``base_url`` points it at a compatible gateway. Distinct from
    :class:`OpenAIClient`, which
    speaks Chat Completions. Auth is ``Authorization: Bearer`` (the OpenAI SDK
    default); the system prompt is sent as ``instructions`` and the user prompt
    as ``input``.

    On the Responses API ``max_output_tokens`` budgets *hidden reasoning plus
    visible output*. These models reason before answering, so a small
    per-call cap (e.g. the judge's ``max_tokens=8``) would be consumed entirely
    by reasoning and yield empty text. The budget is therefore floored at
    ``min_output_tokens`` (the caller's larger value still wins). Optionally set
    ``reasoning_effort`` (``minimal``/``low``/``medium``/``high``) to trade
    reasoning depth for cost and latency on these short-output tasks.
    """

    def __init__(self, model: str, api_key: str,
                 base_url: Optional[str] = None, temperature: float = 0.8,
                 reasoning_effort: Optional[str] = None,
                 min_output_tokens: int = 2048) -> None:
        super().__init__(provider="openai_responses", model=model, temperature=temperature)
        import openai

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=_responses_proxy_base(base_url) if base_url else None,
        )
        self._reasoning_effort = reasoning_effort or None
        self._min_output_tokens = int(min_output_tokens)

    def _complete(self, system, user, max_tokens, temperature):
        temp = temperature
        # Floor the budget so hidden reasoning tokens can't starve visible output.
        budget = max(int(max_tokens), self._min_output_tokens)
        kwargs = dict(model=self.model, instructions=system, input=user,
                      max_output_tokens=budget)
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        try:
            resp = self._client.responses.create(temperature=temp, **kwargs)
        except Exception as exc:  # noqa: BLE001 - narrow retry; re-raised otherwise
            # Reasoning models (the GPT-5 reasoning family) reject a non-default
            # temperature; retry once without it rather than failing the run.
            if "temperature" not in str(exc).lower():
                raise
            resp = self._client.responses.create(**kwargs)
        return _responses_output_text(resp)


# Process-level registry: one ResponseCache per file path, so every LLM family
# in a run shares a single in-memory map and appends to one transcript file.
_CACHES: Dict[str, ResponseCache] = {}


def _default_cache(cfg: Config) -> Optional[ResponseCache]:
    """Shared on-disk LLM response cache, unless disabled via ``llm.cache: false``."""
    if not bool(cfg.get("llm", "cache", default=True)):
        return None
    path = cfg.cache_dir / "llm_cache.jsonl"
    key = str(path)
    if key not in _CACHES:
        _CACHES[key] = ResponseCache(path)
    return _CACHES[key]


def _build_remote_client(spec: Dict, provider: str, model: str,
                         temperature: float, key: Optional[str]) -> LLMClient:
    """Construct a network-backed client; raise on missing key / unknown provider."""
    if provider == "anthropic":
        if not key:
            raise RuntimeError("missing ANTHROPIC API key")
        return AnthropicClient(model, key, temperature)
    if provider == "openai":
        if not key:
            raise RuntimeError("missing OPENAI API key")
        base = os.environ.get(spec.get("base_url_env", "") or "", None)
        return OpenAIClient(model, key, base_url=base, temperature=temperature)
    if provider in ("gemini", "google"):
        if not key:
            raise RuntimeError("missing GEMINI API key")
        return GeminiClient(model, key, temperature)
    if provider == "anthropic_proxy":
        if not key:
            raise RuntimeError("missing gateway API token")
        base = spec.get("base_url") or os.environ.get("LLM_BASE_URL") or None
        return AnthropicProxyClient(model, key, base_url=base, temperature=temperature)
    if provider in ("openai_responses", "openai_proxy"):
        if not key:
            raise RuntimeError("missing gateway API token")
        base = spec.get("base_url") or os.environ.get("LLM_BASE_URL") or None
        return OpenAIResponsesClient(
            model, key, base_url=base, temperature=temperature,
            reasoning_effort=spec.get("reasoning_effort"),
            min_output_tokens=int(spec.get("min_output_tokens", 2048)),
        )
    raise RuntimeError(f"unknown provider '{provider}'")


def _generation_fingerprint(spec: Dict, provider: str) -> str:
    """Canonical string of generation-affecting llm-config knobs NOT already in
    the cache key (which covers provider/model/temperature/max_tokens).

    These change the model's actual output, so they must change the cache key
    too — otherwise a config tweak would silently replay stale text. Derived from
    the config ``spec`` (not an instantiated SDK client) so the real client and
    the offline-replay client compute the identical fingerprint. Returns ``""``
    for providers without such knobs, leaving their keys byte-for-byte unchanged.
    """
    if provider in ("openai_responses", "openai_proxy"):
        # The Responses path floors the output budget at ``min_output_tokens`` and
        # can set ``reasoning_effort``; both alter the returned text.
        effort = spec.get("reasoning_effort") or ""
        floor = int(spec.get("min_output_tokens", 2048))
        return f"reasoning_effort={effort};min_output_tokens={floor}"
    return ""


def get_client(cfg: Config, family: str,
               cache: Optional[ResponseCache] = None) -> LLMClient:
    """Build the client for an LLM family (``generator_a``/``b``/``judge_c``).

    Falls back to :class:`MockClient` when LLM use is disabled, the provider is
    ``mock``, the required library is missing, or the API key is absent. Real
    clients are wired to the shared response cache (``cache`` overrides the
    process default) so identical requests are memoised across runs.
    """
    spec = cfg.get("llm", family, default={}) or {}
    enabled = bool(cfg.get("llm", "enabled", default=False))
    provider = spec.get("provider", "mock")
    model = spec.get("model", "mock-1")
    temperature = float(spec.get("temperature", 0.8))

    if not enabled or provider == "mock":
        return MockClient(model=f"mock::{model}", temperature=temperature)

    if bool(cfg.get("llm", "offline", default=False)):
        shared = cache if cache is not None else _default_cache(cfg)
        if shared is None:
            raise RuntimeError("llm.offline=true requires the response cache "
                               "(do not set llm.cache=false).")
        client = OfflineCacheClient(provider=provider, model=model, temperature=temperature)
        client.cache = shared
        client.key_extra = _generation_fingerprint(spec, provider)
        return client

    try:
        client = _build_remote_client(spec, provider, model, temperature, cfg.api_key(family))
    except Exception as exc:
        print(f"[llm] family '{family}' falling back to mock ({exc}).")
        return MockClient(model=f"mock::{model}", temperature=temperature)
    client.cache = cache if cache is not None else _default_cache(cfg)
    client.key_extra = _generation_fingerprint(spec, provider)
    return client


def get_generator_clients(cfg: Config) -> Dict[str, LLMClient]:
    """Return the three configured clients keyed by 'A', 'B', 'judge'.

    All three share one :class:`ResponseCache`, so a build appends every family's
    generations to a single transcript and reuses them on replay.
    """
    cache = _default_cache(cfg)
    return {
        "A": get_client(cfg, "generator_a", cache=cache),
        "B": get_client(cfg, "generator_b", cache=cache),
        "judge": get_client(cfg, "judge_c", cache=cache),
    }
