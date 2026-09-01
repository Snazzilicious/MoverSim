"""Interactive globe plotting renderers."""

from __future__ import annotations

import numpy as np

from mover_sim.math.coordinates import A
from mover_sim.plotting.events import map_events_to_positions
from mover_sim.plotting.transforms import (
    assign_platform_colors,
    compute_ecef_equalized_bounds,
    select_platforms,
)


EARTH_RADIUS_M = float(A)


def _import_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        return None
    return go


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_run_globe requires matplotlib or plotly to be installed. "
            "Install one of them with `pip install matplotlib` or `pip install plotly`."
        ) from exc
    return plt


def _resolve_backend(backend):
    if backend not in (None, "auto", "plotly", "matplotlib"):
        raise ValueError("backend must be one of: `auto`, `plotly`, `matplotlib`")

    if backend in (None, "auto"):
        return "plotly" if _import_plotly() is not None else "matplotlib"
    return backend


def _wireframe_sphere(resolution=24, radius=EARTH_RADIUS_M):
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _combined_bounds(platforms):
    bounds = compute_ecef_equalized_bounds(platforms)
    if bounds is None:
        radius = EARTH_RADIUS_M
        center = np.zeros(3, dtype=float)
    else:
        radius = max(float(bounds["radius"]), EARTH_RADIUS_M)
        center = np.zeros(3, dtype=float)

    return {
        "center": center,
        "radius": radius,
        "mins": center - radius,
        "maxs": center + radius,
    }


def _build_event_hover_text(event):
    platform_id = event["platform_id"] if event["platform_id"] is not None else "-"
    return f"event={event['topic']}<br>time={event['time']:.3f}s<br>platform={platform_id}"


def _plot_run_globe_plotly(run, platforms, colors, mapped_events):
    go = _import_plotly()
    if go is None:
        raise ImportError(
            "plot_run_globe backend `plotly` requires plotly to be installed. "
            "Install it with `pip install plotly`, or use `backend='matplotlib'`."
        )

    x, y, z = _wireframe_sphere()
    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0.0, "#d8dde6"], [1.0, "#aeb8c4"]],
            opacity=0.18,
            showscale=False,
            hoverinfo="skip",
        )
    )

    for platform_id, track in platforms.items():
        if track.position_ecef is None or len(track.position_ecef) == 0:
            continue
        position = np.asarray(track.position_ecef, dtype=float)
        hover_text = [f"platform={platform_id}<br>time={t:.3f}s" for t in track.time]
        figure.add_trace(
            go.Scatter3d(
                x=position[:, 0],
                y=position[:, 1],
                z=position[:, 2],
                mode="lines",
                name=platform_id,
                line={"color": colors[platform_id], "width": 4},
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    event_positions = [event for event in mapped_events if event["position_ecef"] is not None]
    if event_positions:
        xyz = np.vstack([event["position_ecef"] for event in event_positions])
        figure.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name="events",
                marker={"color": "black", "size": 4, "symbol": "x"},
                text=[_build_event_hover_text(event) for event in event_positions],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    bounds = _combined_bounds(platforms)
    figure.update_layout(
        title="MoverSim Globe View",
        scene={
            "xaxis_title": "ECEF X (m)",
            "yaxis_title": "ECEF Y (m)",
            "zaxis_title": "ECEF Z (m)",
            "aspectmode": "cube",
            "xaxis": {"range": [bounds["mins"][0], bounds["maxs"][0]]},
            "yaxis": {"range": [bounds["mins"][1], bounds["maxs"][1]]},
            "zaxis": {"range": [bounds["mins"][2], bounds["maxs"][2]]},
        },
        showlegend=True,
    )
    return figure


def _plot_run_globe_matplotlib(run, platforms, colors, mapped_events):
    plt = _import_matplotlib()
    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection="3d")

    x, y, z = _wireframe_sphere()
    ax.plot_wireframe(x, y, z, color="#b0b7c3", linewidth=0.4, alpha=0.25)

    has_trajectory = False
    for platform_id, track in platforms.items():
        if track.position_ecef is None or len(track.position_ecef) == 0:
            continue
        has_trajectory = True
        position = np.asarray(track.position_ecef, dtype=float)
        ax.plot(
            position[:, 0],
            position[:, 1],
            position[:, 2],
            color=colors[platform_id],
            label=platform_id,
        )

    event_positions = [event for event in mapped_events if event["position_ecef"] is not None]
    if event_positions:
        xyz = np.vstack([event["position_ecef"] for event in event_positions])
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color="black",
            marker="x",
            s=32,
            label="events",
            depthshade=False,
        )

    bounds = _combined_bounds(platforms)
    ax.set_xlim(bounds["mins"][0], bounds["maxs"][0])
    ax.set_ylim(bounds["mins"][1], bounds["maxs"][1])
    ax.set_zlim(bounds["mins"][2], bounds["maxs"][2])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_title("MoverSim Globe View")
    ax.set_xlabel("ECEF X (m)")
    ax.set_ylabel("ECEF Y (m)")
    ax.set_zlabel("ECEF Z (m)")
    ax.grid(True)
    if has_trajectory or event_positions:
        ax.legend(loc="best")
    figure.tight_layout()
    return figure


def plot_run_globe(run, platform_ids=None, event_topics=None, backend="auto"):
    """Render a globe-style trajectory view for one run.

    When ``backend`` is ``auto``, plotly is used if installed; otherwise a simpler
    matplotlib 3D fallback is returned.

    Example:
        ```python
        from mover_sim.plotting import load_hdf5_run, plot_run_globe

        run = load_hdf5_run("output/scenario_air_launched_cruise_missile.h5")
        globe = plot_run_globe(run, backend="matplotlib")
        ```
    """

    platforms = select_platforms(run, platform_ids=platform_ids)
    colors = assign_platform_colors(platforms)
    mapped_events = map_events_to_positions(run, event_topics=event_topics, platform_ids=platform_ids)

    resolved_backend = _resolve_backend(backend)
    if resolved_backend == "plotly":
        return _plot_run_globe_plotly(run, platforms, colors, mapped_events)
    return _plot_run_globe_matplotlib(run, platforms, colors, mapped_events)
