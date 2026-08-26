"""Place point labels on a scatter plot without letting them collide.

Shared by the analysis scripts, which all label their points with model names —
long, variable-width strings that overlap readily once several models land close
together.
"""


def place_labels(
    fig,
    ax,
    names: list[str],
    xs: list[float],
    ys: list[float],
    xerr: list[tuple[float, float] | None] | None = None,
    yerr: list[tuple[float, float] | None] | None = None,
) -> None:
    """Annotate each point with its name, nudged clear of the others.

    Models frequently share an x value, and on the smaller subsets they share a
    y too, so the names collide readily. Overlap is tested against the labels'
    rendered pixel extents rather than the points' data coordinates: the names
    are long and their widths vary, so point proximity is a poor proxy for
    whether the text actually overlaps.

    `xerr` and `yerr` give each point's error bar as (below, above) in data
    units, for the figures that draw confidence intervals. A bar is a long
    obstacle that a marker-sized one does not stand in for: without this a label
    clears every marker and still prints straight through the neighboring
    interval it was routed past. None, or a None entry, means that point is just
    its marker in that direction.

    Call after the axes are otherwise final — limits, scales and tight_layout —
    since the extents are measured against the axes as they stand.
    """
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    mid = (min(xs) + max(xs)) / 2
    step = 11  # points; a little over one line at this font size

    # The markers are obstacles too: a label that clears every other label can
    # still be printed across a neighboring point.
    radius = 7

    def obstacle(i: int) -> Bbox:
        """Point i's footprint in pixels: its marker, widened by any error bars.

        The arms are transformed endpoint by endpoint rather than as pixel
        offsets, so they come out right on a log axis, where the two arms of a
        symmetric-in-data interval render at different lengths.
        """
        px, py = ax.transData.transform((xs[i], ys[i]))
        left = right = below = above = 0.0
        err = xerr[i] if xerr else None
        if err is not None:
            lo = ax.transData.transform((xs[i] - err[0], ys[i]))[0]
            hi = ax.transData.transform((xs[i] + err[1], ys[i]))[0]
            left, right = px - lo, hi - px
        err = yerr[i] if yerr else None
        if err is not None:
            lo = ax.transData.transform((xs[i], ys[i] - err[0]))[1]
            hi = ax.transData.transform((xs[i], ys[i] + err[1]))[1]
            below, above = py - lo, hi - py
        return Bbox.from_bounds(
            px - radius - left,
            py - radius - below,
            2 * radius + left + right,
            2 * radius + below + above,
        )

    placed = [obstacle(i) for i in range(len(xs))]

    # Tightest scores first, so the crowded rows are laid out before the
    # isolated points claim space near them.
    order = sorted(range(len(names)), key=lambda i: (ys[i], xs[i]))
    for i in order:
        # Labels on the right half go to the left of their marker, so the text
        # stays inside the axes.
        right = xs[i] > mid
        # Clears the marker's own box, so a point never blocks its own label.
        offset, text = (radius + 3) * (-1 if right else 1), None
        for attempt in range(24):
            # Alternate above and below the marker, widening each time.
            dy = (attempt + 1) // 2 * step * (1 if attempt % 2 else -1)
            if text is not None:
                text.remove()
            text = ax.annotate(
                names[i],
                (xs[i], ys[i]),
                xytext=(offset, dy),
                textcoords="offset points",
                ha="right" if right else "left",
                va="center",
                fontsize=8,
                color="#333333",
            )
            box = text.get_window_extent(renderer=renderer).expanded(1.02, 1.35)
            if not any(box.overlaps(other) for other in placed):
                break
        placed.append(box)
