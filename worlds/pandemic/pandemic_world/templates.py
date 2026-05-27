"""Pandemic question templates + the world's registry.

Registered into a core TemplateRegistry — no global monkey-patching.
"""

from fbsim_core.questions.schema import QuestionTemplate
from fbsim_core.questions.registry import TemplateRegistry

CASES_COMPARATIVE = QuestionTemplate(
    template_id="cases_comparative",
    signal_name="cumulative_cases",
    question_template="Will {region_a} have more cumulative cases than {region_b} at day {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.cumulative_cases.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["region_a", "region_b", "player_id_a", "player_id_b", "resolution_turn"],
)

DEATHS_COMPARATIVE = QuestionTemplate(
    template_id="deaths_comparative",
    signal_name="cumulative_deaths",
    question_template="Will {region_a} have more cumulative deaths than {region_b} at day {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.cumulative_deaths.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["region_a", "region_b", "player_id_a", "player_id_b", "resolution_turn"],
)

CASES_CONTINUOUS = QuestionTemplate(
    template_id="cases_continuous",
    signal_name="cumulative_cases",
    question_template="How many cumulative cases will {region} have at day {resolution_turn}?",
    resolution_type="continuous",
    data_path="time_series.cumulative_cases.{player_id}.{resolution_turn}",
    comparison_op="value",
    required_params=["region", "player_id", "resolution_turn"],
)

ALL_TEMPLATES = [CASES_COMPARATIVE, DEATHS_COMPARATIVE, CASES_CONTINUOUS]

REGISTRY = TemplateRegistry(ALL_TEMPLATES)
