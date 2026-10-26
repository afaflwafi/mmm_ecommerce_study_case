"""Extract and draw the causal DAG implied by a Google Meridian model's InputData.

Meridian does not learn a causal graph from data -- the DAG is an assumption
baked into the model. But every *element* of that DAG (which variables are
treatments, which are confounders/controls, what the KPI is) lives in
``meridian.data.input_data.InputData``. This module introspects it directly:

    treatments   <- media_channel, rf_channel, organic_media_channel,
                    organic_rf_channel, non_media_channel
    confounders  <- control_variable
    outcome      <- kpi (+ kpi_type)
    lagged edges <- any treatment with adstock (paid/organic media & RF)

Nothing is supplied by hand.

Usage
-----
    from meridian_causal_dag import extract_dag_elements, plot_meridian_causal_dag

    dag = extract_dag_elements(mmm)      # or extract_dag_elements(mmm.input_data)
    print(dag["nodes"]); print(dag["edges"])

    plot_meridian_causal_dag(mmm, n_periods=3)
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# InputData properties holding channel/variable name coordinates, by causal role.
# Adstocked roles get lagged treatment -> future-KPI edges.
_TREATMENT_PROPS = (
    ("media_channel",         "paid media",    True),
    ("rf_channel",            "paid R&F",      True),
    ("organic_media_channel", "organic media", True),
    ("organic_rf_channel",    "organic R&F",   True),
    ("non_media_channel",     "non-media",     False),
)
_CONTROL_PROP = "control_variable"


def _names(data, prop):
    da = getattr(data, prop, None)
    if da is None:
        return []
    return [str(v) for v in da.values]


def _input_data(source):
    """Accept a Meridian model or an InputData object.

    NOTE: the Meridian *model* also has `.kpi`/`.controls` properties (as
    tensors), so we must resolve `.input_data` FIRST -- duck-typing on `.kpi`
    would silently pick the model and every channel lookup would return None.
    """
    data = getattr(source, "input_data", None)
    if data is not None:
        source = data
    is_input_data = any(
        getattr(source, prop, None) is not None
        for prop, _, _ in _TREATMENT_PROPS
    ) or getattr(source, _CONTROL_PROP, None) is not None
    if not is_input_data:
        raise ValueError(
            "Could not find any channel/control coordinates on this object. "
            "Pass a meridian.model.model.Meridian or "
            "meridian.data.input_data.InputData (got "
            f"{type(source).__name__}). If this is a valid InputData with no "
            "media at all, that is not a plottable MMM."
        )
    return source


def extract_dag_elements(source):
    """Read the assumed causal-DAG elements straight out of InputData.

    Returns a dict with:
      nodes : {"treatments": {group: [names]}, "confounders": [names],
               "outcome": {"name", "kpi_type"}}
      edges : list of (src_role, dst_role, kind) template edges, where kind is
              "causal" or "lagged". These repeat for every time period t.
    """
    data = _input_data(source)

    treatments, lagged_groups = {}, []
    for prop, group, adstocked in _TREATMENT_PROPS:
        names = _names(data, prop)
        if names:
            treatments[group] = names
            if adstocked:
                lagged_groups.append(group)

    confounders = _names(data, _CONTROL_PROP)
    kpi_type = getattr(data, "kpi_type", None)
    outcome = {"name": "KPI", "kpi_type": str(kpi_type) if kpi_type else None}

    edges = [("C_t", "T_t", "causal"), ("C_t", "K_t", "causal"),
             ("T_t", "K_t", "causal")]
    if lagged_groups:
        edges.append(("T_t", "K_t+lag", "lagged"))

    return {
        "nodes": {"treatments": treatments, "confounders": confounders,
                  "outcome": outcome},
        "edges": edges,
        "lagged_treatment_groups": lagged_groups,
    }


def _node(ax, x, y, title, subtitle, face, edge, w=1.9, h=0.56):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=1.0, zorder=3))
    ax.text(x, y + 0.10, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=edge, zorder=4)
    if subtitle:
        ax.text(x, y - 0.12, subtitle, ha="center", va="center",
                fontsize=7, color=edge, zorder=4)


def _arrow(ax, p, q, color="#5F5E5A", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=11,
        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw,
        shrinkA=12, shrinkB=12, zorder=2))


def plot_meridian_causal_dag(source, n_periods=2, max_lag_arrows=1,
                             wrap=40, figsize=None,
                             title="Causal DAG"):
    """Draw the time-indexed DAG using only elements extracted from InputData."""
    dag = extract_dag_elements(source)
    treatments = dag["nodes"]["treatments"]
    confounders = dag["nodes"]["confounders"]
    outcome = dag["nodes"]["outcome"]

    t_names = [n for names in treatments.values() for n in names]
    t_sub = textwrap.shorten(", ".join(t_names), wrap, placeholder=" ...")
    c_sub = textwrap.shorten(", ".join(confounders), wrap, placeholder=" ...") \
        if confounders else "(none in model)"
    k_sub = outcome["kpi_type"] or "kpi"

    has_lag = bool(dag["lagged_treatment_groups"]) and max_lag_arrows > 0
    xs = [i * 2.9 for i in range(n_periods)]
    if figsize is None:
        figsize = (2.2 + 2.9 * n_periods, 4.8)
    fig, ax = plt.subplots(figsize=figsize)

    pos = {}
    for t, x in enumerate(xs, start=1):
        ax.text(x, 3.0, f"Period {t}", ha="center", fontsize=10.5,
                fontweight="bold")
        pos[("C", t)], pos[("T", t)], pos[("K", t)] = (x, 2.4), (x, 1.3), (x, 0.0)
        _node(ax, x, 2.4, f"C{t} controls", c_sub, "#FAECE7", "#993C1D")
        _node(ax, x, 1.3, f"T{t} treatments", t_sub, "#EEEDFE", "#534AB7")
        _node(ax, x, 0.0, f"K{t} {outcome['name']}", k_sub, "#E1F5EE", "#0F6E56")
        _arrow(ax, pos[("C", t)], pos[("T", t)])
        _arrow(ax, pos[("C", t)], pos[("K", t)], rad=0.5)
        _arrow(ax, pos[("T", t)], pos[("K", t)])

    if has_lag:
        for t in range(1, n_periods):
            for lag in range(1, max_lag_arrows + 1):
                if t + lag <= n_periods:
                    _arrow(ax, pos[("T", t)], pos[("K", t + lag)],
                           color="#7F77DD", rad=-0.15, lw=1.5)

    handles = [plt.Line2D([], [], color="#5F5E5A", lw=1.2,
                          label="Assumed causal effect")]
    if has_lag:
        handles.append(plt.Line2D([], [], color="#7F77DD", lw=1.5,
                                  label="Lagged effect (adstock)"))
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)

    ax.set_xlim(-1.6, xs[-1] + 1.6)
    ax.set_ylim(-1.0, 3.3)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig