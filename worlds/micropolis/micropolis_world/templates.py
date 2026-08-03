"""Micropolis question templates + the world's registry.

Registered into a core TemplateRegistry — no global monkey-patching.

TODO: adjust signal_name/question_template/data_path per metric once the
runner's actual metric set is finalized (see runner.METRICS).
"""

from fbsim_core.questions.schema import QuestionTemplate
from fbsim_core.questions.registry import TemplateRegistry

POPULATION_COMPARATIVE = QuestionTemplate(
    template_id="population_comparative",
    signal_name="population",
    question_template="Will {region_a} have more population than {region_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.population.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["region_a", "region_b", "player_id_a", "player_id_b", "resolution_turn"],
)

FUNDS_COMPARATIVE = QuestionTemplate(
    template_id="funds_comparative",
    signal_name="funds",
    question_template="Will {region_a} have more funds than {region_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.funds.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["region_a", "region_b", "player_id_a", "player_id_b", "resolution_turn"],
)

POPULATION_CONTINUOUS = QuestionTemplate(
    template_id="population_continuous",
    signal_name="population",
    question_template="How large will {region}'s population be at turn {resolution_turn}?",
    resolution_type="continuous",
    data_path="time_series.population.{player_id}.{resolution_turn}",
    comparison_op="value",
    required_params=["region", "player_id", "resolution_turn"],
)

ALL_TEMPLATES = [POPULATION_COMPARATIVE, FUNDS_COMPARATIVE, POPULATION_CONTINUOUS]

REGISTRY = TemplateRegistry(ALL_TEMPLATES)
