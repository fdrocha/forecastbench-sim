#!/usr/bin/env python3
"""Per-template resolution-criteria strings for natcond elicitation prompts.

Each rendered string states what fbsim_core.questions.resolver.QuestionResolver
ACTUALLY computes for that template — verified by reading the resolver and by
edge-case cross-checks that run the real resolver on constructed game data
(tests/test_resolution_criteria.py). The criteria follow the resolver, not the
other way around. Resolver facts the wording encodes:

  * comparatives (_resolve_comparative): value_a > value_b, STRICT, both values
    read from the time series at exactly resolution_turn (signal_at does no
    carry-forward). Tie -> NO. Either value missing at that turn -> NO.
  * score_rank_1 (_resolve_rank): rank read at exactly resolution_turn from the
    recorded snapshot rankings (fallback: stable descending sort of the scores
    time series). Exactly one civ holds rank #1 even on a tied top score — the
    stable sort breaks ties by recorded player order — so being tied for first
    is NOT sufficient for YES. No recorded rank -> NO.
  * tech_discovered (_resolve_milestone): cumulative over the WHOLE game,
    events with turn <= resolution_turn (inclusive); pre-snapshot discoveries
    count. Matches tech_name or tech_id, that civ only.
  * wonder_completed (_resolve_event): NOT cumulative from game start — the
    event window is (snapshot_turn, resolution_turn], i.e. strictly after the
    snapshot turn and up to and including the resolution turn. Any civ counts.
  * government_at (_resolve_state_check): the state at resolution_turn is the
    destination of the most recent recorded government_change at or before
    that turn (changed-and-changed-back is judged by the final state). If NO
    change is recorded by then, the answer is NO for every government,
    including the civ's actual starting government.

Derived-horizon questions (q0007_h2..._h5, derive_horizon_bank.py) reuse the
base template ids with a later resolution_turn, so they render through the
same entries here. "h0_"-prefixed comprehension ids resolve through the base
template (mirrored from TemplateRegistry.get).
"""
from __future__ import annotations

from typing import Any

# Every criteria string ends with exactly this sentence-final phrase.
CRITERIA_SUFFIX = "as determined by the game's recorded metrics."

# Generic fallback: the pre-renderer wording, used when a cell carries no
# template_id/parameters (older cell files) or an unknown template.
GENERIC_CRITERIA = (
    "Resolves YES if the answer to the question is affirmative in the "
    "simulation state at turn {rt}, " + CRITERIA_SUFFIX)

# The six comparative templates differ only in the compared metric.
COMPARATIVE_METRICS = {
    "tech_comparative": "count of known technologies",
    "score_comparative": "game score",
    "population_comparative": "total population",
    "city_count_comparative": "number of cities",
    "territory_comparative": "number of map tiles controlled",
    "treasury_comparative": "treasury (gold reserve)",
}

BINARY_TEMPLATE_IDS = tuple(COMPARATIVE_METRICS) + (
    "score_rank_1", "tech_discovered", "wonder_completed", "government_at")


def _get(parameters: dict[str, Any], template_id: str, key: str) -> Any:
    value = (parameters or {}).get(key)
    if value in (None, ""):
        raise ValueError(
            f"template {template_id!r} needs parameter {key!r} to render "
            f"resolution criteria (got {value!r})")
    return value


def _comparative(template_id: str, parameters: dict, rt: int) -> str:
    a = _get(parameters, template_id, "civ_a")
    b = _get(parameters, template_id, "civ_b")
    metric = COMPARATIVE_METRICS[template_id]
    return (
        f"Resolves YES if {a}'s recorded {metric} is strictly greater than "
        f"{b}'s at exactly turn {rt} (both values are read at exactly that "
        f"turn; values at earlier or later turns do not matter). An exact tie "
        f"resolves NO. If either civilization's value is missing from the "
        f"record at turn {rt}, the question resolves NO. Resolution is "
        + CRITERIA_SUFFIX)


def _score_rank_1(template_id: str, parameters: dict, rt: int) -> str:
    civ = _get(parameters, template_id, "civ")
    return (
        f"Resolves YES if {civ} holds rank #1 in the game's recorded score "
        f"rankings at exactly turn {rt} (the ranking is read at exactly that "
        f"turn). Exactly one civilization holds rank #1 even when the top "
        f"score is tied: the recorded ranking breaks ties by the order in "
        f"which players appear in the game record, so {civ} being tied for "
        f"the top score resolves YES only if the record ranks {civ} #1. If "
        f"{civ} has no recorded rank at turn {rt}, the question resolves NO. "
        f"Resolution is " + CRITERIA_SUFFIX)


def _tech_discovered(template_id: str, parameters: dict, rt: int) -> str:
    civ = _get(parameters, template_id, "civ")
    tech = _get(parameters, template_id, "tech_name")
    return (
        f"Resolves YES if the game record shows {civ} discovering {tech} at "
        f"any turn up to and including turn {rt}; a discovery recorded at any "
        f"earlier point in the game counts, including turns before the "
        f"report's snapshot. Resolves NO if no such discovery by {civ} is "
        f"recorded by turn {rt}. Resolution is " + CRITERIA_SUFFIX)


def _wonder_completed(template_id: str, parameters: dict, rt: int) -> str:
    wonder = _get(parameters, template_id, "wonder_name")
    # The resolver windows this event to (snapshot_turn, resolution_turn].
    # Banks always carry snapshot_turn; this pipeline observes at turn 60.
    snap = (parameters or {}).get("snapshot_turn", 60)
    return (
        f"Resolves YES if the game record shows {wonder} being completed by "
        f"any civilization at any turn up to and including turn {rt}, "
        f"counting only completions strictly after turn {snap} (a completion "
        f"recorded at or before turn {snap} does not count). Resolves NO "
        f"otherwise. Resolution is " + CRITERIA_SUFFIX)


def _government_at(template_id: str, parameters: dict, rt: int) -> str:
    civ = _get(parameters, template_id, "civ")
    gov = _get(parameters, template_id, "government_type")
    return (
        f"Resolves YES if {civ}'s government at exactly turn {rt} is {gov}. "
        f"The government at turn {rt} is the destination of the most recent "
        f"government change recorded for {civ} at any turn up to and "
        f"including turn {rt}; changing into {gov} and back out again before "
        f"turn {rt} does not count — only the government in effect at turn "
        f"{rt} matters. If no government change for {civ} is recorded by "
        f"turn {rt}, the question resolves NO regardless of {civ}'s starting "
        f"government. Resolution is " + CRITERIA_SUFFIX)


_RENDERERS = {tid: _comparative for tid in COMPARATIVE_METRICS}
_RENDERERS.update({
    "score_rank_1": _score_rank_1,
    "tech_discovered": _tech_discovered,
    "wonder_completed": _wonder_completed,
    "government_at": _government_at,
})


def render_criteria(template_id: str, parameters: dict[str, Any] | None,
                    resolution_turn: int) -> str:
    """Precise criteria for one binary template at one resolution turn.

    Raises ValueError for unknown template ids or missing parameters —
    callers that must never fail use criteria_or_generic().
    """
    tid = template_id[3:] if template_id.startswith("h0_") else template_id
    if tid not in _RENDERERS:
        raise ValueError(f"no criteria renderer for template_id {template_id!r}")
    return _RENDERERS[tid](tid, parameters or {}, resolution_turn)


def criteria_or_generic(template_id: str | None,
                        parameters: dict[str, Any] | None,
                        resolution_turn: int) -> str:
    """render_criteria, falling back to the generic wording when the cell
    carries no template_id/parameters (older cell files) or an unknown id."""
    if template_id:
        try:
            return render_criteria(template_id, parameters, resolution_turn)
        except ValueError:
            pass
    return GENERIC_CRITERIA.format(rt=resolution_turn)
