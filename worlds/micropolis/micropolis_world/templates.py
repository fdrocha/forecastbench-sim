"""Micropolis question templates + the world's registry.

Registered into a core TemplateRegistry — no global monkey-patching.

TODO: adjust signal_name/question_template/data_path per metric once the
runner's actual metric set is finalized (see city_sim.METRICS).
"""

from fbsim_core.questions.registry import TemplateRegistry
from fbsim_core.questions.schema import QuestionTemplate

from . import module_globals as g

def _get_continuous_template(metric: str) -> QuestionTemplate:
    return QuestionTemplate(
    template_id=f"{metric}_continuous",
    signal_name=,
    # Note we assume there is only one region/player in the Micropolis world, so we don't need to specify a region_id in the question.
    question_template="What population be at turn {resolution_turn}?",
    resolution_type="continuous",
    data_path="time_series.population.{player_id}.{resolution_turn}",
    comparison_op="value",
    required_params=["region", "player_id", "resolution_turn"],
)

ALL_TEMPLATES = [POPULATION_COMPARATIVE, FUNDS_COMPARATIVE, POPULATION_CONTINUOUS]

REGISTRY = TemplateRegistry(ALL_TEMPLATES)
