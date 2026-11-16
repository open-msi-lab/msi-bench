"""Synthetic multi-source overlay generation (Section 4.3).

Channel separation is enforced throughout: LLM clients produce
*narrative text only*; this package's code owns every coordinate, timestamp,
device/network identifier, perturbation, and forged credential.

* :mod:`contributors` — the simulated contributor population + histories.
* :mod:`generators`   — crowd / sensor / attack record generation.
* :mod:`benchmark`    — event extraction, assembly, event-level split, release.
"""

from .benchmark import build_benchmark, Benchmark  # noqa: F401
