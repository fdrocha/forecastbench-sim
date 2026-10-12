"""Core resolver + registry generalize to any world (no FreeCiv import).

Builds a tiny synthetic world: one comparative template, turn-major time_series,
and checks the resolver resolves through a world-provided registry.
"""

from fbsim_core.questions.schema import QuestionInstance, QuestionTemplate
from fbsim_core.questions.registry import TemplateRegistry
from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.timeseries import value_at


TEMPLATE = QuestionTemplate(
    template_id="widgets_comparative",
    signal_name="widgets",
    question_template="Will {entity_a} have more widgets than {entity_b} at t={resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.widgets.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["entity_a", "entity_b", "player_id_a", "player_id_b", "resolution_turn"],
)


def _game_data():
    # turn-major: time_series[metric][turn][entity] = value
    return {
        "time_series": {
            "widgets": {
                "10": {"0": 5, "1": 9},
                "20": {"0": 12, "1": 7},
            }
        }
    }


def test_value_at_turn_major():
    gd = _game_data()
    assert value_at(gd, "widgets", 1, 10) == 9
    assert value_at(gd, "widgets", 0, 20) == 12
    assert value_at(gd, "widgets", 99, 20) is None  # missing entity
    assert value_at(gd, "widgets", 0, 999) is None  # missing turn


def test_registry_lookup_and_h0_prefix():
    reg = TemplateRegistry([TEMPLATE])
    assert reg.get("widgets_comparative") is TEMPLATE
    assert reg.get("h0_widgets_comparative") is TEMPLATE  # prefix tolerated
    assert "widgets_comparative" in reg


def test_resolver_comparative():
    reg = TemplateRegistry([TEMPLATE])
    resolver = QuestionResolver(reg)
    q = QuestionInstance(
        question_id="q0",
        template_id="widgets_comparative",
        resolution_turn=20,
        horizon="H1",
        parameters={"entity_a": "A", "entity_b": "B", "player_id_a": 0, "player_id_b": 1},
        question_text="...",
    )
    res = resolver.resolve(q, _game_data(), snapshot_turn=10)
    assert res.answer is True          # A=12 > B=7 at turn 20
    assert res.value_a == 12 and res.value_b == 7
