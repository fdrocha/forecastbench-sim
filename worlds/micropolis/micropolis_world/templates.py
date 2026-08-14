"""Micropolis question templates + the world's registry."""

from fbsim_core.questions.registry import TemplateRegistry
from fbsim_core.questions.schema import QuestionTemplate

from . import module_globals as g


def _get_continuous_template(metric: str) -> QuestionTemplate:
    metric_label = g.METRIC_LABELS[metric]
    return QuestionTemplate(
        template_id=f"{metric}_continuous",
        signal_name=metric,
        # Note we assume there is only one region/player in the Micropolis world, so we don't need to specify a region_id in the question.
        question_template=f"What will the {metric_label} be at turn {{resolution_turn}}?",
        resolution_type="continuous",
        data_path=f"time_series.{metric}.{{player_id}}.{{resolution_turn}}",
        comparison_op="value",
        required_params=["player_id", "resolution_turn"],
    )


Q_METRICS = [
    #    "cityScore",
    "cityPop",
    "totalFunds",
    "trafficAverage",
    "pollutionAverage",
    "crimeAverage",
    "landValueAverage",
]

ALL_TEMPLATES: list[QuestionTemplate] = [
    _get_continuous_template(metric) for metric in Q_METRICS
]

REGISTRY = TemplateRegistry(ALL_TEMPLATES)
