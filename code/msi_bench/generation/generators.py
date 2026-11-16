"""Narrative-text and structured-field generators (Section 4.3).

Text is produced either by an LLM client (real run) or by deterministic
templates (mock run / no API key). In both cases the LLM is asked to emit a
``{{COORD}}`` placeholder, and *code* substitutes the noised position in correct
mariner format — guaranteeing the claimed spatial-error distribution regardless
of the generator (channel separation). The exact prompts are kept here so
they can be released with the benchmark.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional, Sequence, Tuple

from ..llm.client import LLMClient
from ..similarity import text_duplication
from .contributors import DEVICE_NOISE_NM

COORD_TOKEN = "{{COORD}}"

# Genuine-crowd word-count quantiles (q05, q10, q20, …, q90, q95), measured on
# the DEV split of the shipped benchmark (n = 2,005 GENUINE_CROWD
# reports; median 41 words / 237 chars). Used as the length-target anchor for
# sophisticated fabrications and Sybil base texts: a Kerckhoffs
# adversary matches the length profile of the population it infiltrates, so the
# generator must not hand detectors a "systematically short attack text"
# artifact. A fixed table (rather than
# the live crowd realization) keeps attack prompts identical across the
# aligned/degraded brackets and the multi-seed variance builds, so their texts
# stay shared in the response cache. Naive-tier fabrications keep their terse
# style — low effort IS that tier's signature.
CROWD_WORDCOUNT_Q = (22, 24, 27, 29, 32, 41, 47, 52, 56, 62, 66)


def sample_target_words(rng: random.Random) -> int:
    """Draw a report-length target (words) from the genuine-crowd anchor table."""
    return int(rng.choice(CROWD_WORDCOUNT_Q))

# Output-token ceiling for a single report. The prompts ask for 1-3 sentences
# (~40-80 tokens), so this is a generous headroom that prevents mid-sentence
# truncation rather than a target length: a well-behaved model stops well short
# of it, and billing is per token generated, so the slack is effectively free.
# Override per run via `llm.gen_max_tokens` in config.
GEN_MAX_TOKENS = 512

# --- released prompt templates --------------------------------------------
CROWD_SYSTEM = (
    "You simulate a mariner submitting a brief, informal hazard observation to "
    "a crowdsourced maritime safety app. Write 1-3 sentences in a plausible "
    "seafarer's voice. Do NOT copy official wording verbatim. Where you mention "
    "the position, write the literal token {{COORD}} instead of any coordinates."
)
# A T1 fabrication is, by threat-model definition, an ordinary-looking crowd
# report submitted for an event that never happened: the "fabricated" property is
# carried by the record label (Category.ATTACK_FABRICATION) and the absence of a
# real anchor event, NOT by the prose announcing its own fakeness. We therefore
# generate it with the *same* seafarer voice as a genuine crowd report — keeping
# the two linguistically indistinguishable, which is the whole point of the attack
# and avoids a generator artifact that leaks the label into the text channel. The
# alternative "realistic but ENTIRELY FICTITIOUS / describe a hazard that does not
# exist" wording reliably tripped model refusals (and produced disclaimer-laden,
# trivially separable output); this framing does not ask the model to fake-yet-
# disclaim, so it complies cleanly.
FABRICATION_SYSTEM = (
    "You simulate a mariner submitting a brief, informal hazard observation to "
    "a crowdsourced maritime safety app. Write 1-3 sentences in a plausible "
    "seafarer's voice. Output only the observation text — no headers, labels, "
    "disclaimers, or commentary. Where you mention the position, write the "
    "literal token {{COORD}} instead of any coordinates."
)


def format_coord(lat: float, lon: float) -> str:
    """Format a (lat, lon) as a NAVAREA mariner coordinate string."""
    def fmt(value: float, hemis: Tuple[str, str], deg_width: int) -> str:
        hemi = hemis[0] if value >= 0 else hemis[1]
        v = abs(value)
        deg = int(v)
        minutes = (v - deg) * 60.0
        return f"{deg:0{deg_width}d}-{minutes:05.2f}{hemi}"

    return f"{fmt(lat, ('N', 'S'), 2)} {fmt(lon, ('E', 'W'), 3)}"


def noise_position(
    centroid: Tuple[float, float], sigma_nm: float, rng: random.Random
) -> Tuple[float, float]:
    """Add bivariate-normal positional noise (σ in nm) to a centroid."""
    bearing = rng.uniform(0, 2 * math.pi)
    dist_nm = abs(rng.gauss(0, sigma_nm))
    dlat = (dist_nm * math.cos(bearing)) / 60.0
    dlon = (dist_nm * math.sin(bearing)) / (60.0 * max(0.2, math.cos(math.radians(centroid[0]))))
    return (centroid[0] + dlat, centroid[1] + dlon)


# --- templated (mock) text -------------------------------------------------
_HAZARD_PHRASES = {
    "derelict": ["adrift derelict vessel", "abandoned hull drifting", "partially submerged wreck adrift"],
    "wreck": ["submerged wreck", "uncharted obstruction", "wreckage on the bottom"],
    "rocket": ["rocket launch hazard area", "range operations, keep clear", "launch window restrictions"],
    "survey": ["survey vessel operating", "seismic survey under way", "cable operations in progress"],
    "iceberg": ["iceberg sighted", "growlers reported", "ice in the area"],
    "buoy": ["buoy off station", "navigation light unreliable", "marker missing"],
    "generic": ["hazard to navigation", "obstruction reported", "caution advised"],
}
_EXPERT_OPENERS = {
    "officer": ["Master reports", "Bridge watch confirms", "OOW observed"],
    "recreational": ["Saw something out here —", "Heads up,", "Not sure but"],
    "fishing": ["F/V reporting", "Hauling gear, spotted", "Crew sighted"],
}


def _templated_crowd_text(rng: random.Random, hazard_key: str, expertise: str,
                          quality: str) -> str:
    opener = rng.choice(_EXPERT_OPENERS.get(expertise, _EXPERT_OPENERS["recreational"]))
    phrase = rng.choice(_HAZARD_PHRASES.get(hazard_key, _HAZARD_PHRASES["generic"]))
    body = f"{opener} {phrase} near {COORD_TOKEN}."
    if quality == "good":
        body += " Position confirmed by GPS; recommend wide berth."
    elif quality == "poor":
        body = f"{opener} {phrase} somewhere around {COORD_TOKEN}, not certain."
    return body


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L (LCS-based F1) over whitespace tokens — paraphrase-distance gate."""
    ta, tb = (a or "").lower().split(), (b or "").lower().split()
    if not ta or not tb:
        return 0.0
    # LCS length via DP
    dp = [[0] * (len(tb) + 1) for _ in range(len(ta) + 1)]
    for i in range(1, len(ta) + 1):
        for j in range(1, len(tb) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ta[i - 1] == tb[j - 1] else max(
                dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(ta)][len(tb)]
    prec, rec = lcs / len(ta), lcs / len(tb)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def generate_crowd_text(
    client: LLMClient, hazard_key: str, expertise: str, quality: str,
    official_context: Optional[str], rng: random.Random, max_rouge: float = 0.6,
    max_tokens: int = GEN_MAX_TOKENS, cache_salt: Optional[str] = None,
) -> str:
    """Return crowd narrative text containing the {{COORD}} placeholder.

    For real (non-mock) generation with official context, a ROUGE-L gate rejects
    text too close to the source (Section 4.3): one regeneration with a
    stronger paraphrase instruction is attempted, then a templated fallback.

    ``cache_salt`` (typically the report's record id) gives each logical draw its
    own response-cache slot, with the attempt index appended so the ROUGE retry
    is a distinct entry; this keeps independent reports of one event diverse
    while letting a same-seed rebuild replay them verbatim.
    """
    if client.is_mock:
        return _templated_crowd_text(rng, hazard_key, expertise, quality)
    base_user = (
        f"Hazard type: {hazard_key}. Reporter: {expertise} ({quality} quality). "
        + (f"Official context (do NOT copy): {official_context[:400]}"
           if official_context else "No official text provided; write free-form.")
    )
    for attempt in range(2):
        user = base_user + (" Paraphrase strongly; reuse no phrases." if attempt else "")
        salt = f"{cache_salt}#a{attempt}" if cache_salt else None
        text = client.complete(CROWD_SYSTEM, user, max_tokens=max_tokens, cache_salt=salt)
        if COORD_TOKEN not in text:  # ensure the placeholder is present
            text = text.rstrip(". ") + f" near {COORD_TOKEN}."
        if not official_context or rouge_l(text, official_context) <= max_rouge:
            return text
    return _templated_crowd_text(rng, hazard_key, expertise, quality)  # fallback


def _templated_fabrication_text(rng: random.Random, hazard_key: str,
                                sophisticated: bool) -> str:
    """Deterministic fallback fabrication text (mock runs and refusal fallback)."""
    phrase = rng.choice(_HAZARD_PHRASES.get(hazard_key, _HAZARD_PHRASES["generic"]))
    if sophisticated:
        return (f"Reporting {phrase} at {COORD_TOKEN}. Confirmed visually, "
                f"vessels advised to maintain a safe distance.")
    return f"there is {phrase} at {COORD_TOKEN}"


# A model may decline the request or wrap the report in synthetic-data
# disclaimers ("EXERCISE — NOT FOR NAVIGATION", "fabricated=true", a refusal plus
# an offer of a clearly-labeled sample). Such text must never enter the benchmark:
# it leaks the ground-truth label straight into the prose channel and is trivially
# separable from genuine crowd reports, inflating any detector's score (the
# artifact-screening guard of Section 4.6). Markers are matched as phrases, not bare
# domain words — "exercise"/"operations"/"firing" are legitimate hazard terms.
_REFUSAL_MARKERS = (
    "i can't", "i can’t", "i cannot", "i can not", "i won't", "i won’t",
    "i will not", "i'm sorry", "i’m sorry", "i am sorry", "i'm unable",
    "i’m unable", "i am unable", "as an ai", "i'm not able", "i’m not able",
)
_DISCLAIMER_MARKERS = (
    "not for navigation", "not operational", "operational_use", "fabricated=true",
    "training_purpose", "validation tagging", "synth-train", "training sample",
    "for research", "research use", "research purpose", "for dataset",
    "for testing", "this is a simulated", "this is a synthetic", "exercise only",
    "this is fictional", "this is fictitious", "clearly fictional", "exercise /",
)


def _looks_invalid(text: str) -> bool:
    """True if the model refused or emitted synthetic-data labels/disclaimers."""
    t = (text or "").strip().lower()
    if len(t) < 10:
        return True
    return (any(m in t for m in _REFUSAL_MARKERS)
            or any(m in t for m in _DISCLAIMER_MARKERS))


def generate_fabrication_text(
    client: LLMClient, hazard_key: str, sophisticated: bool, rng: random.Random,
    max_tokens: int = GEN_MAX_TOKENS, cache_salt: Optional[str] = None,
    target_words: Optional[int] = None,
) -> str:
    """T1 fabrication text. ``target_words`` (sophisticated tier)
    appends a length instruction anchored on the genuine-crowd distribution so
    fabrications are not systematically shorter than the population they mimic;
    ``None`` (the naive tier) leaves the prompt — and thus its cache slot —
    unchanged.
    """
    if client.is_mock:
        return _templated_fabrication_text(rng, hazard_key, sophisticated)
    style = "complete, plausible, in a shipping lane, correct register" if sophisticated \
        else "terse and incomplete"
    user = f"Hazard type: {hazard_key}. Style: {style}."
    if target_words is not None:
        user += f" Aim for roughly {int(target_words)} words."
    # Bounded resample on refusal/contamination. The cache salt carries the attempt
    # index, so each try is a distinct cache slot (a same-seed rebuild replays the
    # exact sequence) and a retry never just replays the cached refusal. Resampling
    # rescues only *stochastic* refusals; a categorical refuser is caught by the
    # deterministic template fallback rather than an unbounded, non-terminating loop.
    for attempt in range(3):
        salt = f"{cache_salt}#a{attempt}" if cache_salt else None
        text = client.complete(FABRICATION_SYSTEM, user, max_tokens=max_tokens,
                               cache_salt=salt)
        if not _looks_invalid(text):
            if COORD_TOKEN not in text:
                text += f" at {COORD_TOKEN}"
            return text
    return _templated_fabrication_text(rng, hazard_key, sophisticated)  # fallback


# --- paraphrased-Sybil member text (fourth sophistication tier) -------------
# The residual limit of independence check (d) (Section 5.1): an adversary who authors
# distinct prose per campaign identity defeats a near-verbatim duplication test
# BY CONSTRUCTION. This tier models exactly that adversary — each member's text
# is a generative paraphrase of the campaign's base claim, gated (like the
# adversary themselves would, knowing the defence) to stay below the θ_ndup
# containment band. The tier EXTENDS the sophistication axis; the scripted
# low/medium/high tiers are unchanged, so check (d)'s win on them stays
# measured and the residual limit becomes a number instead of an assertion.
PARAPHRASE_SYSTEM = (
    "You simulate a mariner submitting a brief, informal hazard observation to "
    "a crowdsourced maritime safety app. You are shown another mariner's report "
    "of a hazard. Write YOUR OWN independent report of the same hazard: a "
    "different voice, different wording and sentence structure, the same "
    "essential facts, and a similar overall length (1-3 sentences). Do not "
    "reuse phrases from the original. Where you mention the position, write "
    "the literal token {{COORD}} instead of any coordinates. Output only the "
    "report text — no headers, labels, or commentary."
)

# Deterministic paraphrase frames for the mock/fallback path. Frames share no
# 3-word phrase with each other or with the templated base texts, so mock
# campaign members stay below the θ_ndup containment band like real ones.
_PARA_FRAMES = (
    "Different vessel here, we can confirm {phrase} close to {coord}; keeping well clear of it.",
    "Passed the same spot an hour ago and the {phrase} is right there near {coord}.",
    "Adding our sighting: {phrase} seen by our whole watch around {coord} today.",
    "Our bridge team independently observed {phrase} in the vicinity of {coord}.",
    "Second report from us — {phrase} plainly visible off {coord}, advise a wide berth.",
    "Transiting now; that {phrase} everyone mentioned sits just about at {coord}.",
    "We verified it ourselves: {phrase} still present, position roughly {coord}.",
    "Lookout on our boat also picked up the {phrase} standing near {coord}.",
)


def _templated_paraphrase(rng: random.Random, hazard_key: str,
                          siblings: Sequence[str] = (), gate: float = 0.60) -> str:
    """Deterministic mock paraphrase honouring the same containment gate as the
    real path: candidates (frame × phrase, rng-shuffled) are tried until one
    clears the gate against every accepted sibling, so mock campaigns exhibit
    the tier's defining property (distinct prose per member) for any k ≤ 8."""
    phrases = _HAZARD_PHRASES.get(hazard_key, _HAZARD_PHRASES["generic"])
    combos = [(f, p) for f in _PARA_FRAMES for p in phrases]
    rng.shuffle(combos)
    fallback = None
    for frame, phrase in combos:
        text = frame.format(phrase=phrase, coord=COORD_TOKEN)
        if fallback is None:
            fallback = text
        if all(text_duplication(text, s) < gate for s in siblings):
            return text
    return fallback


def generate_paraphrase(
    client: LLMClient, base_text: str, hazard_key: str, rng: random.Random,
    siblings: Sequence[str] = (), max_tokens: int = GEN_MAX_TOKENS,
    cache_salt: Optional[str] = None, gate: float = 0.60,
) -> str:
    """One campaign member's independently-authored paraphrase of ``base_text``.

    The containment gate models the adaptive adversary: candidates whose
    word-3-gram containment against the base text or any already-accepted
    sibling reaches ``gate`` (margin below θ_ndup = 0.75) are resampled — a
    campaign operator who knows the duplication check would proof their texts
    the same way. Bounded retries; on exhaustion the least-contained candidate
    is kept (realized containment is recorded by the caller, so a partial
    defeat of check (d) is measured, never hidden).
    """
    if client.is_mock:
        return _templated_paraphrase(rng, hazard_key, siblings, gate)
    user = f"Original report: {base_text}\nWrite your own independent report of this hazard."
    best: Optional[str] = None
    best_c = float("inf")
    for attempt in range(3):
        salt = f"{cache_salt}#a{attempt}" if cache_salt else None
        prompt = user if attempt == 0 else (
            user + " Use COMPLETELY different wording; share no phrase of three "
                   "or more words with the original.")
        text = client.complete(PARAPHRASE_SYSTEM, prompt, max_tokens=max_tokens,
                               cache_salt=salt)
        if _looks_invalid(text):
            continue
        if COORD_TOKEN not in text:
            text = text.rstrip(". ") + f" near {COORD_TOKEN}."
        c = max([text_duplication(text, base_text)]
                + [text_duplication(text, s) for s in siblings])
        if c < gate:
            return text
        if c < best_c:
            best, best_c = text, c
    return best if best is not None else _templated_paraphrase(rng, hazard_key,
                                                               siblings, gate)


def substitute_coord(text: str, lat: float, lon: float) -> str:
    """Replace every {{COORD}} token with a formatted coordinate."""
    return text.replace(COORD_TOKEN, format_coord(lat, lon))


def sensor_text(sensor_type: str, lat: float, lon: float, rng: random.Random) -> str:
    """Terse, formulaic sensor-report text (no LLM)."""
    coord = format_coord(lat, lon)
    if sensor_type == "radar":
        return f"RADAR CONTACT, UNCHARTED OBSTRUCTION, POS {coord}, {rng.randint(2,5)} SWEEPS, RANGE {rng.randint(3,12)}NM"
    if sensor_type == "ais":
        return f"AIS ANOMALY, TARGET LOITERING, POS {coord}, SOG {rng.uniform(0,3):.1f}KN"
    return f"MET STATION OBSERVATION, POS {coord}, WIND {rng.randint(5,45)}KN, SEA STATE {rng.randint(1,7)}"
