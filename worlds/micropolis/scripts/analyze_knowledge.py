#!/usr/bin/env -S uv run python3
"""Score the Micropolis domain-knowledge test and correlate it with ECI.

Reports each model's normalized knowledge score, then correlates that score
against Epoch's ECI capability index — overall and split by statement difficulty
and honeypot status. The question is whether knowing Micropolis tracks general
capability.

Reads only the responses already cached by scripts/run_knowledge_eval.py, so it
prompts no models and needs no API keys or network. The only thing it writes is
one scatter plot per statement subset, which --no-plot suppresses.

Usage:
    scripts/analyze_knowledge.py
    scripts/analyze_knowledge.py --min-n 6
    scripts/analyze_knowledge.py --no-plot
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

import micropolis_world.module_globals as g
from micropolis_world.config import CONFIG_DIR, Config, main_with_config
from micropolis_world.knowledge_eval.runner import (
    OUT_DIR,
    Answer,
    Statement,
    get_cached_answers,
    model_slug,
    statements,
)
from micropolis_world.knowledge_eval.scoring import score, tally
from micropolis_world.plot_labels import place_labels

# The config that defines this dataset, so unlike the other scripts there is no
# config argument to choose a different one.
CONFIG_PATH = CONFIG_DIR / "knowledge_eval.json5"

PLOTS_PATH = OUT_DIR / "plots"


@dataclass(frozen=True)
class Entry:
    """One model's cached answers, with the ECI score to correlate against."""

    slug: str
    name: str
    eci: int
    answers: list[Answer]


def eci_of(model_id: str) -> int | None:
    """ECI score for a provider/name model id, or None if it has none.

    ECI_MAP is keyed on the bare model name, without the provider prefix.
    """
    return g.ECI_MAP.get(model_id.split("/", 1)[1])


@dataclass(frozen=True)
class Unscored:
    """A cached model left out of the correlation, and why."""

    name: str
    reason: str
    answers: list[Answer]


def load_entries() -> tuple[list[Entry], list[Unscored], list[str]]:
    """Cached models joined to ECI_MAP, plus what did not join and why.

    Returns the joined entries sorted by ECI descending, the models left out,
    and the ECI_MAP names with no cached response.

    The join runs through the config's model ids: get_cached_answers() is keyed
    on filename slugs, which have lost the "/" that separates provider from
    name, and applying model_slug() to the configured ids recovers it exactly.
    A response gathered under --models for an id the config no longer lists
    falls back to splitting the slug on its first "_", which is right for every
    provider prefix in use but is a guess, so it says so.
    """
    configured = Config.load(CONFIG_PATH).get_str_list("models")
    slug_to_id = {model_slug(m): m for m in configured}

    entries, skipped = [], []
    for slug, answers in get_cached_answers().items():
        model_id = slug_to_id.get(slug)
        guessed = model_id is None
        if guessed:
            model_id = slug.replace("_", "/", 1)
        eci = eci_of(model_id)
        name = model_id.split("/", 1)[1]
        if eci is None:
            reason = "no ECI score"
            if guessed:
                reason += f" (id guessed from slug as {model_id})"
            skipped.append(Unscored(name, reason, answers))
            continue
        entries.append(Entry(slug, name, eci, answers))

    entries.sort(key=lambda e: -e.eci)
    joined = {e.name for e in entries}
    return entries, skipped, sorted(set(g.ECI_MAP) - joined)


@dataclass(frozen=True)
class Subset:
    """A slice of the statement set to score and correlate over."""

    name: str
    stmts: list[Statement]

    @property
    def label(self) -> str:
        """The name with its size, for tables and plot titles."""
        return f"{self.name} (n={len(self.stmts)})"

    @property
    def slug(self) -> str:
        """The name as a filename component, without the volatile count.

        Derived from the name rather than stored, so a renamed subset cannot
        keep writing to a filename describing the old one. The count is left
        out so adding statements doesn't orphan the previous run's plots.
        """
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")


def subsets() -> list[Subset]:
    """The statement slices to correlate over.

    Difficulty tiers and honeypots are this world's analogue of the horizons
    freeciv's score_bin_condition.py breaks its correlations down by.
    """
    slices = [Subset("All", statements)]
    for d in sorted({s.difficulty for s in statements}):
        slices.append(
            Subset(f"Difficulty {d}", [s for s in statements if s.difficulty == d])
        )
    slices.append(Subset("Honeypot", [s for s in statements if s.is_honeypot]))
    slices.append(Subset("Non-honeypot", [s for s in statements if not s.is_honeypot]))
    return slices


def significance(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def direction(rho: float) -> str:
    """Whether a correlation runs with or against general capability.

    The knowledge score is higher-is-better, so a positive correlation with ECI
    means the more capable models know Micropolis better: pro-g. Note this is
    the opposite of the equivalent label in freeciv's score_bin_condition.py,
    which correlates ECI against CRPS, where lower is better.
    """
    return "pro-g" if rho > 0 else "anti-g"


def correlate(xs: list[int], ys: list[float], min_n: int) -> dict | None:
    """Spearman and Pearson correlation of ECI against score, or None.

    None when there are too few models to say anything, or when either variable
    is constant and a correlation is undefined. Both sides are guarded: a set of
    models that happens to share one ECI score leaves no spread in x, which
    yields nan rather than a coefficient, and nan would print as a real number
    with an "anti-g" direction label.
    """
    if len(xs) < min_n or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rho, p_rho = stats.spearmanr(xs, ys)
    r, p_r = stats.pearsonr(xs, ys)
    return {"rho": rho, "p_rho": p_rho, "r": r, "p_r": p_r, "n": len(xs)}


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a table with the first column left-aligned and the rest right."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
            for i, c in enumerate(cells)
        )

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def print_join_report(
    entries: list[Entry], skipped: list[Unscored], no_response: list[str]
) -> None:
    """Say which models made it into the correlation, and which did not."""
    print(f"\n{'=' * 70}")
    print("MODELS")
    print("=" * 70)
    total = len(entries) + len(skipped)
    print(
        f"{total} cached response(s), {len(g.ECI_MAP)} ECI score(s), "
        f"{len(entries)} matched"
    )
    for u in sorted(skipped, key=lambda u: u.name):
        print(f"  skipped {u.name}: {u.reason}")
    if no_response:
        print(f"  ECI score but no cached response: {', '.join(no_response)}")

    counts = sorted({s.difficulty for s in statements})
    census = "/".join(
        str(sum(1 for s in statements if s.difficulty == d)) for d in counts
    )
    honeypots = sum(1 for s in statements if s.is_honeypot)
    print(
        f"\n{len(statements)} statements "
        f"(difficulty {'/'.join(map(str, counts))} = {census}; "
        f"{honeypots} honeypot)"
    )


def print_scores(entries: list[Entry], skipped: list[Unscored]) -> None:
    """Per-model score table, best first, including models without an ECI."""
    print(f"\n{'=' * 70}")
    print("SCORES (1.0 = every statement correct, 0.0 = all unknown, -2.0 = all wrong)")
    print("=" * 70)

    scored = [(e.name, str(e.eci), e.answers) for e in entries]
    scored += [(u.name, "—", u.answers) for u in skipped]

    rows = []
    for name, eci, answers in scored:
        counts = tally(answers)
        rows.append(
            [
                name,
                eci,
                str(counts["correct"]),
                str(counts["wrong"]),
                str(counts["unknown"]),
                str(counts["unparseable"]),
                f"{score(counts):.3f}",
            ]
        )
    rows.sort(key=lambda r: -float(r[-1]))
    print_table(
        ["Model", "ECI", "Correct", "Wrong", "Unknown", "Unparseable", "Score"], rows
    )


def correlations_by_subset(
    entries: list[Entry], min_n: int
) -> list[tuple[Subset, dict | None]]:
    """Correlate ECI against the score on each statement subset.

    One pass, since the Spearman and Pearson tables and the scatter plots all
    report on the same fits. A subset whose correlation is undefined carries
    None.
    """
    ecis = [e.eci for e in entries]
    return [
        (
            sub,
            correlate(
                ecis, [score(tally(e.answers, sub.stmts)) for e in entries], min_n
            ),
        )
        for sub in subsets()
    ]


def print_correlations(results: list[tuple[Subset, dict | None]]) -> None:
    """ECI against normalized score, as a Spearman table then a Pearson one.

    Spearman leads because it asks the question the ECI ranking supports: do
    the more capable models know more? Pearson additionally assumes the
    relationship is linear in ECI points, which is a stronger claim about an
    index like this, so it is reported alongside rather than instead.
    """
    for title, coef, key, p_key in (
        ("Spearman (rank)", "ρ", "rho", "p_rho"),
        ("Pearson (linear)", "r", "r", "p_r"),
    ):
        print(f"\n{'=' * 70}")
        print(f"ECI × knowledge score by statement subset — {title}")
        print("=" * 70)
        for sub, result in results:
            if result is None:
                print(
                    f"  {sub.label:<22} —  (too few models, or no variation in score)"
                )
                continue
            print(
                f"  {sub.label:<22} {coef}={result[key]:+.3f}  p={result[p_key]:.4f} "
                f"{significance(result[p_key]):<4} "
                f"({direction(result[key])}, n={result['n']})"
            )


def plot_scatter(
    entries: list[Entry],
    sub: Subset,
    result: dict | None,
    outdir: Path = PLOTS_PATH,
) -> Path:
    """Scatter each model's ECI against its score on one statement subset.

    The point of the figure over the correlation coefficient is that it shows
    the shape: whether the trend is carried by the whole range or by a couple
    of models at the ends, and which models sit off the line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ecis = [e.eci for e in entries]
    scores = [score(tally(e.answers, sub.stmts)) for e in entries]

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.scatter(ecis, scores, s=70, color="#3266a8", zorder=3)

    # Least-squares fit, drawn only when a correlation was reportable at all.
    if result is not None and len(set(ecis)) > 1:
        fit = stats.linregress(ecis, scores)
        xs = [min(ecis), max(ecis)]
        ax.plot(
            xs,
            [fit.intercept + fit.slope * x for x in xs],
            color="#c2432d",
            lw=1.5,
            zorder=2,
            label=(
                f"fit: ρ={result['rho']:+.3f} (p={result['p_rho']:.4f}), "
                f"r={result['r']:+.3f} (p={result['p_r']:.4f})"
            ),
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("ECI (Epoch capability index)")
    ax.set_ylabel("Normalized knowledge score (1.0 = all correct)")
    ax.set_title(
        f"Micropolis domain knowledge vs. ECI — {sub.name}\n"
        f"{len(entries)} models, {len(sub.stmts)} statements"
    )
    # 0.0 is where a model that abstained on everything lands, so it separates
    # knowing something from guessing badly. Only drawn when a model is close
    # enough for it to be a useful reference; otherwise it would stretch the
    # y-axis over empty space.
    if min(scores) < 0.15:
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.12, y=0.08)
    fig.tight_layout()

    # After tight_layout, so the labels are measured against the axes the figure
    # actually ends up with rather than the provisional ones.
    fig.canvas.draw()
    place_labels(fig, ax, [e.name for e in entries], ecis, scores)

    out = outdir / f"eci_vs_knowledge_score-{sub.slug}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_caveats(entries: list[Entry]) -> None:
    print(f"\n{'=' * 70}")
    print("NOTES")
    print("=" * 70)
    honeypots = sum(1 for s in statements if s.is_honeypot)
    print(
        "The score is higher-is-better, so ρ>0 means the more capable models\n"
        "score better (pro-g). This is the opposite sign convention to\n"
        "worlds/freeciv/scripts/score_bin_condition.py, which correlates ECI\n"
        "against CRPS, where lower is better.\n"
        f"\nWith n={len(entries)} models the per-difficulty ρ values are not\n"
        "statistically distinguishable from each other; treat their ordering as\n"
        f"suggestive. The honeypot row rests on {honeypots} statements per model and is\n"
        "underpowered.\n"
        "\nECI_MAP notes that grok-4.20's single published score was assigned to\n"
        "the reasoning variant, a judgment call carrying one of the points here."
    )


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-n",
        type=int,
        default=4,
        help="Minimum models with an ECI score before a correlation is reported "
        "(default: 4)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Print the tables without writing the scatter plots",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("MICROPOLIS WORLD — domain knowledge analysis")
    print("=" * 70)

    entries, skipped, no_response = load_entries()
    if not entries:
        print("\nNo cached responses with an ECI score. Run run_knowledge_eval.py.")
        return

    print_join_report(entries, skipped, no_response)
    print_scores(entries, skipped)

    results = correlations_by_subset(entries, args.min_n)
    print_correlations(results)
    print_caveats(entries)

    if not args.no_plot:
        print()
        for sub, result in results:
            print(f"Wrote {plot_scatter(entries, sub, result)}")


if __name__ == "__main__":
    main()
