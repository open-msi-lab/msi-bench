"""Simulated contributor population with longitudinal histories (Section 4.3).

Each contributor has a persistent ``provider_id``, a device class and
authentication level drawn from the device and authentication mixes, a latent
reliability ρ ∈ [0.5, 0.99], and a simulated 12-month submission history
(N ~ Poisson(15) past outcomes, each CORRECT/INCORRECT ~ Bernoulli(ρ), plus
20% UNRESOLVED). The histories are exactly what HistoryScore (Eq. (3))
consumes; without them experiment 1d/5d would be vacuous.

Adversarial identities come in two sophistication tiers (the adaptive-adversary
hardening of Section 4.3):

* **throwaway** — the low-effort signature: anonymous, consumer-tier device,
  empty submission history. Cheap to create, and exactly what the device
  dimension is *supposed* to catch.
* **farmed** — an account-farming adversary (Kerckhoffs: the attacker knows the
  defence): device class, network origin, and an aged submission history drawn
  from the SAME distributions as genuine contributors, authentication up to
  PKI_BASIC (hardware-vetted PKI_FULL stays out of reach). Identity metadata
  then carries no attack label, so detection must come from content, temporal,
  and social evidence — the realistic hard case.

Draw-sequence discipline: :meth:`ContributorPopulation.adversary` executes the
same five shared-rng draws regardless of tier, so the benchmark builder's global
draw sequence — and therefore every downstream record id and LLM-cache salt —
is independent of the tier. All
tier-specific randomness comes from dedicated RNGs derived from the population
seed and the identity key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Device-class mix (provider_type CROWDSOURCE).
_DEVICE_MIX = [
    ("PROFESSIONAL", 0.05),
    ("MARINE_CERTIFIED", 0.15),
    ("CONSUMER_GPS", 0.30),
    ("MOBILE_DEVICE", 0.45),
    ("UNKNOWN", 0.05),
]
# Authentication-level mix.
_AUTH_MIX = [
    ("PKI_FULL", 0.10),
    ("PKI_BASIC", 0.20),
    ("REGISTERED", 0.40),
    ("ANONYMOUS", 0.30),
]
# Farmed-account authentication mix: an account farmer registers (or buys)
# accounts, so a farmed identity is at least REGISTERED; PKI_BASIC is obtainable
# with effort; hardware-vetted PKI_FULL is not (Section 3.2). Proportions are the
# genuine REGISTERED:PKI_BASIC ratio renormalised.
_FARMED_AUTH_MIX = [
    ("PKI_BASIC", 1.0 / 3.0),
    ("REGISTERED", 2.0 / 3.0),
]
# Per-device positional noise (nm) used when substituting the {{COORD}} token.
DEVICE_NOISE_NM = {
    "PROFESSIONAL": 0.5,
    "MARINE_CERTIFIED": 0.8,
    "CONSUMER_GPS": 1.2,
    "MOBILE_DEVICE": 2.0,
    "UNKNOWN": 2.0,
}


@dataclass
class Contributor:
    provider_id: str
    device_class: str
    auth_level: str
    rho: float
    home: Tuple[float, float]
    network_origin: str


def _weighted_choice(rng: random.Random, items):
    r = rng.random()
    acc = 0.0
    for name, w in items:
        acc += w
        if r <= acc:
            return name
    return items[-1][0]


@dataclass
class ContributorPopulation:
    n: int
    seed: int = 0
    now: datetime = datetime(2024, 1, 1)
    contributors: List[Contributor] = field(default_factory=list)
    histories: Dict[str, List[Tuple[datetime, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        for i in range(self.n):
            dev = _weighted_choice(rng, _DEVICE_MIX)
            auth = _weighted_choice(rng, _AUTH_MIX)
            rho = rng.uniform(0.5, 0.99)
            home = (rng.uniform(5.0, 45.0), rng.uniform(-95.0, -40.0))  # NAVAREA IV-ish
            net = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.0/24"
            cid = f"crowd-{i:04d}"
            self.contributors.append(Contributor(cid, dev, auth, rho, home, net))
            self.histories[cid] = self._make_history(rng, rho)

    def _make_history(self, rng: random.Random, rho: float,
                      lam: float = 15) -> List[Tuple[datetime, str]]:
        n = _poisson(rng, lam)
        out = []
        for _ in range(n):
            t = self.now - timedelta(days=rng.uniform(0, 365))
            if rng.random() < 0.20:
                outcome = "UNRESOLVED"
            else:
                outcome = "CORRECT" if rng.random() < rho else "INCORRECT"
            out.append((t, outcome))
        out.sort()
        return out

    def sample(self, rng: random.Random) -> Contributor:
        return rng.choice(self.contributors)

    def farm_identity(self, cid: str, lam: float = 15,
                      auth_mix=None, key: Optional[str] = None) -> Contributor:
        """A 'farmed' adversarial identity (adaptive-adversary tier).

        Device class, network origin, and an aged submission history are drawn
        from the SAME distributions as genuine contributors (auth from the
        farmed mix, capped at PKI_BASIC), so identity metadata carries no attack
        label. ``lam`` scales the history size (a thinly-farmed account has a
        short record); ``key`` scopes the derived RNG when the caller's ``cid``
        namespace could collide. Deterministic per (population seed, key). The
        generated history is registered under ``cid`` so HistoryScore sees it.
        """
        rng = random.Random(f"{self.seed}|farm|{key or cid}")
        dev = _weighted_choice(rng, _DEVICE_MIX)
        auth = _weighted_choice(rng, auth_mix or _FARMED_AUTH_MIX)
        # Farmed history is curated to look good: reliability drawn from the
        # upper genuine range (correct-looking, low-stakes submissions).
        rho = rng.uniform(0.60, 0.95)
        home = (rng.uniform(5.0, 45.0), rng.uniform(-95.0, -40.0))
        net = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.0/24"
        self.histories[cid] = self._make_history(rng, rho, lam=lam)
        return Contributor(cid, dev, auth, rho, home, net)

    def adversary(self, rng: random.Random, idx: int, farmed: bool = False,
                  tag: str = "adv") -> Contributor:
        """An adversarial contributor at one of two sophistication tiers.

        ``farmed=False`` (throwaway) is the low-effort signature: anonymous,
        consumer-tier device, empty history. ``farmed=True`` delegates to
        :meth:`farm_identity`, whose metadata matches the genuine population.

        The five original shared-rng draws below are executed unconditionally so
        the benchmark build's draw sequence (and thus every downstream LLM-cache
        salt) is independent of the tier; the farmed tier
        takes its actual values from a dedicated derived RNG instead.
        """
        cid = f"{tag}-{idx:04d}"
        n1, n2 = rng.randint(0, 255), rng.randint(0, 255)          # original draws 1-2
        legacy_dev = rng.choice(["MOBILE_DEVICE", "UNKNOWN", "CONSUMER_GPS"])  # 3
        home = (rng.uniform(5.0, 45.0), rng.uniform(-95.0, -40.0))  # original draws 4-5
        if farmed:
            return self.farm_identity(cid)
        contributor = Contributor(
            provider_id=cid,
            device_class=legacy_dev,
            auth_level="ANONYMOUS",
            rho=0.1,
            home=home,
            # Same address space/format as genuine contributors: the subnet is a
            # coordination signal (independence check), not a provenance flag.
            network_origin=f"10.{n1}.{n2}.0/24",
        )
        self.histories.setdefault(cid, [])  # throwaway: empty history → cold start
        return contributor


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler (no numpy dependency in this module)."""
    import math

    l = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1
