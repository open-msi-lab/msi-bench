"""Configuration loading for MSI-Bench.

Resolves a layered configuration:

1. Built-in defaults (mirroring ``config.example.yaml``).
2. ``config.yaml`` in the ``code/`` directory, if present (overrides defaults).
3. Environment variables (for secrets such as API keys), loaded from a ``.env``
   file in ``code/`` if present, then from the process environment.

The public entry point is :func:`get_config`, which returns a :class:`Config`.
Nothing here requires network access or API keys; the LLM section is only
consulted by the generation modules.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "PyYAML is required. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc


# Directory containing this package's parent (i.e. the ``code/`` directory).
CODE_DIR = Path(__file__).resolve().parent.parent
# Project root (one level above ``code/``) — holds ``data/``, ``results/`` and the benchmark directories.
PROJECT_ROOT = CODE_DIR.parent


# Built-in defaults. Kept in sync with config.example.yaml. Using a literal
# dict (rather than reading the example file) means the package is importable
# even if the example file is moved.
_DEFAULTS: Dict[str, Any] = {
    "paths": {"data_dir": "data", "output_dir": "results", "cache_dir": ".cache"},
    "corpus": {
        "nav_area": 4,
        "embedding_years": [2012, 2021],
        "benchmark_years": [2022, 2023],
        "holdout_years": [2024, 2024],
        "replay_years": [2020, 2021],
    },
    "seed": 20260613,
    "n_pipeline_seeds": 5,
    "bootstrap_resamples": 1000,
    "benchmark": {
        "dev_test_split": 0.70,
        "n_contributors": 800,
        "crowd_per_event": [1, 5],
        "crowd_multiplier": 2.5,
        "sensor_multiplier": 0.5,
        "quality_tiers": {"good": 0.60, "moderate": 0.25, "poor": 0.15},
        "attack_multipliers": {
            "fabrication": 0.20,
            "replay": 0.12,
            "tamper": 0.12,
            "impersonation": 0.08,
        },
        # 32 = 8 campaigns for each of the four sophistication levels
        # (low/medium/high/paraphrased).
        "sybil_campaigns": 32,
        "rouge_l_max": 0.60,
        # Attack reserve: EXTRA attack records (this multiple of the base
        # attack counts) generated for E6's fixed-genuine pressure sweep.
        # 5.0 → total pool ≈ 6× base ≈ enough to reach 50% prevalence with the
        # genuine corpus held fixed. 0 disables (multi-seed/holdout builds do).
        "reserve_attack_multiplier": 0,
    },
    "embedding": {"model": None, "finetuned_path": None, "device": "cpu"},
    "mcam": {
        "dim_weights": {
            "device": 0.25,
            "content": 0.30,
            "spatiotemporal": 0.20,
            "social": 0.25,
        },
        "delta_ind_nm": 0.5,
        "delta_disp_nm": 2.0,
        "lambda_history_days": 90,
        "n_min_history": 10,
    },
    "clustering": {
        "min_pts": 2,
        "eps_fallback": 0.45,
        "sim_weights": {"spatial": 0.40, "temporal": 0.25, "semantic": 0.35},
    },
    "fusion": {"q_discount": 0.6, "sigmoid_beta": 10, "sigmoid_k0": 0.5},
    "llm": {"enabled": False},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (no external dependency).

    Parses ``KEY=VALUE`` lines, ignoring blanks and ``#`` comments. Existing
    environment variables are *not* overwritten, so the shell environment wins.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _resolve_env(value: Any) -> Any:
    """Expand ``${VAR}`` references in strings using the environment."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


@dataclass
class Config:
    """Resolved configuration with convenient typed accessors."""

    raw: Dict[str, Any] = field(default_factory=dict)
    project_root: Path = PROJECT_ROOT
    code_dir: Path = CODE_DIR

    # -- path helpers (always returned as absolute Paths) --
    def _path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def data_dir(self) -> Path:
        return self._path("data_dir")

    @property
    def output_dir(self) -> Path:
        out = self._path("output_dir")
        out.mkdir(parents=True, exist_ok=True)
        return out

    @property
    def cache_dir(self) -> Path:
        cache = Path(self.raw["paths"]["cache_dir"])
        cache = cache if cache.is_absolute() else (self.code_dir / cache)
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def nav_area(self) -> int:
        return int(self.raw["corpus"]["nav_area"])

    def years(self, partition: str) -> range:
        """Return an inclusive ``range`` of years for a named partition.

        ``partition`` is one of ``embedding``, ``benchmark``, ``holdout``,
        ``replay`` (matching the ``corpus`` keys, minus the ``_years`` suffix).
        """
        lo, hi = self.raw["corpus"][f"{partition}_years"]
        return range(int(lo), int(hi) + 1)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup, e.g. ``cfg.get('mcam', 'delta_disp_nm')``."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def api_key(self, family: str) -> Optional[str]:
        """Return the API key for an LLM family (``generator_a``/``b``/``judge_c``)."""
        spec = self.get("llm", family, default={}) or {}
        env_name = spec.get("api_key_env")
        return os.environ.get(env_name) if env_name else None


def get_config(config_path: Optional[os.PathLike] = None) -> Config:
    """Build the resolved :class:`Config`.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML config. If omitted, ``code/config.yaml``
        is used when present; otherwise built-in defaults apply.
    """
    # Load .env first so ${VAR} expansion can see the secrets.
    load_dotenv(CODE_DIR / ".env")

    raw = copy.deepcopy(_DEFAULTS)
    path = Path(config_path) if config_path else (CODE_DIR / "config.yaml")
    if path.is_file():
        user_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, user_cfg)
    raw = _resolve_env(raw)
    return Config(raw=raw)
