"""Plot population, funds, crime, and pollution over time for a Micropolis run.

Works off an already-completed simulation's log/events files, loading them from
disk when the CitySimulation doesn't have them in memory yet. Also overlays a
vertical line for every disaster event found in the events log, color-coded by
disaster type, with a legend listing only the disaster types that actually
occurred in this run.
"""

import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from . import module_globals as g
from .city_sim import CitySimulation

# Line color per disaster type; the names come from g.DISASTER_MESSAGES.
DISASTER_COLORS = {
    "Fire": "tab:red",
    "Monster": "tab:purple",
    "Tornado": "tab:gray",
    "Earthquake": "tab:brown",
    "Plane crash": "tab:orange",
    "Shipwreck": "tab:cyan",
    "Train crash": "tab:olive",
    "Helicopter crash": "gold",
    "Firebombing": "darkred",
    "Explosion": "magenta",
    "Flooding": "tab:blue",
    "Nuclear meltdown": "lime",
    "Riots": "black",
}

# cityTime advances 4 times per in-game month (see update.cpp: cityMonth =
# (cityTime % 48) >> 2), so cityYear/cityMonth alone can't distinguish the 4
# sub-month ticks. Spread them across 4 fixed, roughly-evenly-spaced days
# instead of collapsing them onto a single date per month.
SUB_MONTH_DAYS = [1, 8, 15, 22]


def date_for(city_time, city_year, city_month):
    # cityMonth is 0-indexed (Jan=0..Dec=11) per the engine's update.cpp.
    day = SUB_MONTH_DAYS[city_time % 4]
    return datetime.date(city_year, city_month + 1, day)


def plot_run(sim: CitySimulation, output: str | None = None) -> None:
    """Build the population/funds/crime/pollution figure for a CitySimulation.

    Writes the figure to `output` if given, otherwise shows it interactively.
    """
    if sim.log_data is None or sim.events_data is None:
        sim.load_from_disk()
    rows = sim.log_data or []
    events_data = sim.events_data or []

    dates = [date_for(r["cityTime"], r["cityYear"], r["cityMonth"]) for r in rows]
    pop = [r["cityPop"] for r in rows]
    funds = [r["totalFunds"] for r in rows]
    crime = [r["crimeAverage"] for r in rows]
    pollution = [r["pollutionAverage"] for r in rows]

    disaster_events = []
    for event in events_data:
        if event.get("event") != "sendMessage":
            continue
        label = g.DISASTER_MESSAGES.get(event.get("messageNum"))
        if label is None:
            continue
        date = date_for(event["cityTime"], event["cityYear"], event["cityMonth"])
        disaster_events.append((date, (label, DISASTER_COLORS.get(label, "black"))))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(rows[0].get("cityName", sim.city_name))

    plots = [
        (axes[0][0], pop, "Population", "tab:blue"),
        (axes[0][1], funds, "Funds ($)", "tab:green"),
        (axes[1][0], crime, "Crime Average", "tab:red"),
        (axes[1][1], pollution, "Pollution Average", "tab:orange"),
    ]
    disaster_lines = []  # (axes, Line2D, label, date) for hover lookups
    for ax, values, title, color in plots:
        ax.plot(dates, values, color=color)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, alpha=0.3)
        for date, (label, disaster_color) in disaster_events:
            line = ax.axvline(
                date, color=disaster_color, linestyle="--", linewidth=1, alpha=0.7
            )
            disaster_lines.append((ax, line, label, date))

    # Only legend the disaster types that actually occurred in this run.
    legend_labels = {label: color for _, (label, color) in disaster_events}
    if legend_labels:
        handles = [
            plt.Line2D([0], [0], color=color, linestyle="--", linewidth=1)
            for label, color in legend_labels.items()
        ]
        fig.legend(
            handles,
            legend_labels.keys(),
            loc="lower center",
            ncol=len(legend_labels),
            fontsize="small",
        )

    fig.autofmt_xdate()
    fig.tight_layout()
    if legend_labels:
        fig.subplots_adjust(bottom=0.15)

    # Hover tooltip: show the disaster name when the cursor is near one of its
    # vertical lines. Only meaningful in the interactive window (no-op on save).
    if disaster_lines:
        annotations = {}
        for ax, _line, _label, _date in disaster_lines:
            if ax not in annotations:
                annotations[ax] = ax.annotate(
                    "",
                    xy=(0, 0),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", ec="0.3"),
                    visible=False,
                    zorder=100,
                )

        def on_move(event):
            for ax, annotation in annotations.items():
                if event.inaxes != ax:
                    if annotation.get_visible():
                        annotation.set_visible(False)
                        event.canvas.draw_idle()
                    continue
                # ~6 pixels of hover tolerance around each line, in display coords.
                # Multiple disasters often land on the same month (e.g. a plane
                # crash triggering a helicopter response + explosion), so collect
                # every line within tolerance rather than stopping at the first
                # match — otherwise the label shown can silently disagree with
                # whichever line color is actually on top at the cursor.
                hits = []
                for line_ax, _line, label, date in disaster_lines:
                    if line_ax is not ax:
                        continue
                    x_display = ax.transData.transform((mdates.date2num(date), 0))[0]
                    distance = abs(event.x - x_display)
                    if distance <= 6:
                        hits.append((distance, date, label))
                if hits:
                    hits.sort()
                    # Same (date, label) pair can repeat if the same disaster type
                    # fires more than once in a month; keep first-seen order.
                    seen = set()
                    lines_text = []
                    for _distance, date, label in hits:
                        key = (date, label)
                        if key in seen:
                            continue
                        seen.add(key)
                        lines_text.append(f"{label} ({date:%b %Y})")
                    annotation.xy = (event.xdata, event.ydata)
                    annotation.set_text("\n".join(lines_text))
                    annotation.set_visible(True)
                else:
                    annotation.set_visible(False)
                event.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

    if output:
        fig.savefig(output, dpi=150)
        print(f"Wrote {output}")
    else:
        plt.show()
    plt.close(fig)


def save_run_plot(sim: CitySimulation) -> str:
    """Plot `sim` into data/micropolis/<city>/, returning the path written."""
    out = sim.get_plot_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_run(sim, output=str(out))
    return str(out)
