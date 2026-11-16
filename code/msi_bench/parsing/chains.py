"""Layer-1 lifecycle chains (Section 4.3).

Ground-truth event identities are built **without any similarity model** by
linking messages through two metadata-only edge sources:

1. the ``cancel*`` columns, which record *which message cancelled this one*
   (e.g. message 512 carries cancelMsgNumber 520 → edge 512–520); and
2. in-text supersession references "CANCEL NAVAREA IV n/yy" (e.g. 525 → 496).

Self-references (a message cancelling itself at expiry) are ignored. The
weakly-connected components of the resulting graph are the event chains. Because
no text similarity is involved, these chains are a valid, independent evaluation
target for the clustering experiments (no circularity between target and method).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..schema import ParsedMessage


class _UnionFind:
    """Disjoint-set forest with path compression and union by size."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


@dataclass
class ChainResult:
    """Result of Layer-1 chain construction."""

    chain_id_of: Dict[str, str] = field(default_factory=dict)   # present uid -> root
    chains: Dict[str, List[str]] = field(default_factory=dict)   # root -> present uids
    edges: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    def length_distribution(self) -> Dict[int, int]:
        """Map chain length (number of present messages) -> count of chains."""
        dist: Dict[int, int] = {}
        for members in self.chains.values():
            dist[len(members)] = dist.get(len(members), 0) + 1
        return dist


def build_chains(messages: Sequence[ParsedMessage]) -> ChainResult:
    """Build Layer-1 lifecycle chains over ``messages``.

    References to messages outside ``messages`` still create linking nodes, so
    two present messages that both reference the same external message are
    grouped together. Only present messages are reported in the chains.
    """
    uf = _UnionFind()
    present = {m.uid for m in messages}
    edges: List[Tuple[str, str]] = []

    for m in messages:
        uf.add(m.uid)

        # Edge source 1: the cancel* columns.
        if m.cancel_msg_number is not None and m.cancel_msg_year is not None:
            other = f"{m.cancel_msg_year}-{m.cancel_msg_number}"
            if other != m.uid:
                uf.union(m.uid, other)
                edges.append((m.uid, other))

        # Edge source 2: in-text "CANCEL NAVAREA IV n/yy" supersessions.
        for num, year in m.in_text_cancels:
            other = f"{year}-{num}"
            if other != m.uid:
                uf.union(m.uid, other)
                edges.append((m.uid, other))

    chain_id_of: Dict[str, str] = {}
    chains: Dict[str, List[str]] = {}
    # Iterate in sorted order so the chains dict has a deterministic key order:
    # `present` is a set, whose iteration order is randomised per process
    # (PYTHONHASHSEED), which otherwise leaks into the event order downstream
    # (`_extract_events`) and makes every same-seed build — and its LLM cache
    # salts — non-reproducible across runs/machines.
    for uid in sorted(present):
        root = uf.find(uid)
        chain_id_of[uid] = root
        chains.setdefault(root, []).append(uid)

    # Sort members within each chain for determinism.
    for members in chains.values():
        members.sort()

    return ChainResult(chain_id_of=chain_id_of, chains=chains, edges=edges)
