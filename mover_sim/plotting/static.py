"""Static plotting renderers."""

from __future__ import annotations

import numpy as np

from mover_sim.plotting.events import summarize_events
from mover_sim.plotting.transforms import (
    assign_platform_colors,
    compute_ecef_equalized_bounds,
    filter_events,
    panel_data_availability,
    select_platforms,
)


DEFAULT_SUMMARY_SECTIONS = ("trajectory", "position", "velocity", "orientation", "events")


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as exc:
        raise ImportError("plot_run_summary requires matplotlib to be installed") from exc
    return plt, GridSpec


def _resolve_sections(sections, availability):
    requested = DEFAULT_SUMMARY_SECTIONS if sections is None else tuple(sections)
    return [section for section in requested if availability.get(section, False)]


def _set_3d_equalized_bounds(ax, bounds):
    if bounds is None:
        return

    mins = bounds["mins"]
    maxs = bounds["maxs"]
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])


def _plot_trajectory_panel(ax, platforms, colors):
    for platform_id, track in platforms.items():
        if track.position_ecef is None or len(track.position_ecef) == 0:
            continue
        position = np.asarray(track.position_ecef, dtype=float)
        ax.plot(
            position[:, 0],
            position[:, 1],
            position[:, 2],
            color=colors[platform_id],
            label=platform_id,
        )

    ax.set_title("Trajectory")
    ax.set_xlabel("ECEF X (m)")
    ax.set_ylabel("ECEF Y (m)")
    ax.set_zlabel("ECEF Z (m)")
    if len(platforms) > 1:
        ax.legend(loc="best")


def _plot_position_panel(ax, platforms, colors):
    components = ("x", "y", "z")
    for platform_id, track in platforms.items():
        if track.position_ecef is None or len(track.position_ecef) == 0:
            continue
        position = np.asarray(track.position_ecef, dtype=float)
        for index, component in enumerate(components):
            ax.plot(
                track.time,
                position[:, index],
                color=colors[platform_id],
                linestyle=("-", "--", ":")[index],
                label=f"{platform_id} {component}",
            )

    ax.set_title("Position vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECEF Position (m)")
    ax.grid(True)


def _plot_velocity_panel(ax, platforms, colors):
    components = ("vx", "vy", "vz")
    for platform_id, track in platforms.items():
        if track.velocity_ecef is None or len(track.velocity_ecef) == 0:
            continue
        velocity = np.asarray(track.velocity_ecef, dtype=float)
        for index, component in enumerate(components):
            ax.plot(
                track.time,
                velocity[:, index],
                color=colors[platform_id],
                linestyle=("-", "--", ":")[index],
                label=f"{platform_id} {component}",
            )

    ax.set_title("Velocity vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECEF Velocity (m/s)")
    ax.grid(True)


def _plot_orientation_panel(ax, platforms, colors):
    components = ("qw", "qx", "qy", "qz")
    linestyles = ("-", "--", ":", "-.")
    for platform_id, track in platforms.items():
        if track.orientation is None or len(track.orientation) == 0:
            continue
        orientation = np.asarray(track.orientation, dtype=float)
        for index, component in enumerate(components):
            ax.plot(
                track.time,
                orientation[:, index],
                color=colors[platform_id],
                linestyle=linestyles[index],
                label=f"{platform_id} {component}",
            )

    ax.set_title("Orientation vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Quaternion Component")
    ax.grid(True)


def _add_event_markers(ax, events):
    for event in events:
        ax.axvline(event.time, color="0.6", alpha=0.35, linewidth=0.8)


def _plot_event_panel(ax, run, event_topics, platform_ids):
    summaries = summarize_events(run, event_topics=event_topics, platform_ids=platform_ids, max_events=12)
    total_events = len(filter_events(run, event_topics=event_topics, platform_ids=platform_ids))

    lines = ["Event Summary"]
    if not summaries:
        lines.append("No matching events")
    else:
        lines.extend(summary["label"] for summary in summaries)
        if total_events > len(summaries):
            lines.append(f"... {total_events - len(summaries)} more")

    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")


def _set_run_title(fig, run):
    title_parts = ["MoverSim Run Summary"]
    sample_interval = run.metadata.get("sample_interval")
    if sample_interval is not None:
        title_parts.append(f"sample_interval={sample_interval}")
    created_utc = run.metadata.get("created_utc")
    if created_utc:
        title_parts.append(str(created_utc))
    fig.suptitle(" | ".join(title_parts))


def plot_run_summary(run, platform_ids=None, event_topics=None, sections=None):
    """Render a composite static summary figure for one run."""

    plt, GridSpec = _import_matplotlib()

    platforms = select_platforms(run, platform_ids=platform_ids)
    events = filter_events(run, event_topics=event_topics, platform_ids=platform_ids)
    availability = panel_data_availability(platforms, events)
    enabled_sections = _resolve_sections(sections, availability)
    colors = assign_platform_colors(platforms)

    time_series_sections = [
        section for section in enabled_sections if section in ("position", "velocity", "orientation")
    ]
    has_top_row = any(section in enabled_sections for section in ("trajectory", "events"))
    row_count = len(time_series_sections) + (1 if has_top_row else 0)
    if row_count == 0:
        row_count = 1

    fig = plt.figure(figsize=(14, 4 + 3 * row_count))
    gs = GridSpec(row_count, 2, figure=fig, width_ratios=[3, 2])
    current_row = 0

    if "trajectory" in enabled_sections and "events" in enabled_sections:
        trajectory_ax = fig.add_subplot(gs[current_row, 0], projection="3d")
        _plot_trajectory_panel(trajectory_ax, platforms, colors)
        _set_3d_equalized_bounds(trajectory_ax, compute_ecef_equalized_bounds(platforms))

        events_ax = fig.add_subplot(gs[current_row, 1])
        _plot_event_panel(events_ax, run, event_topics, platform_ids)
        current_row += 1
    elif "trajectory" in enabled_sections:
        trajectory_ax = fig.add_subplot(gs[current_row, :], projection="3d")
        _plot_trajectory_panel(trajectory_ax, platforms, colors)
        _set_3d_equalized_bounds(trajectory_ax, compute_ecef_equalized_bounds(platforms))
        current_row += 1
    elif "events" in enabled_sections:
        events_ax = fig.add_subplot(gs[current_row, :])
        _plot_event_panel(events_ax, run, event_topics, platform_ids)
        current_row += 1

    for section in time_series_sections:
        ax = fig.add_subplot(gs[current_row, :])
        if section == "position":
            _plot_position_panel(ax, platforms, colors)
        elif section == "velocity":
            _plot_velocity_panel(ax, platforms, colors)
        elif section == "orientation":
            _plot_orientation_panel(ax, platforms, colors)
        _add_event_markers(ax, events)
        if ax.lines:
            ax.legend(loc="best", ncols=2, fontsize="small")
        current_row += 1

    if not enabled_sections:
        ax = fig.add_subplot(gs[0, :])
        ax.axis("off")
        ax.text(0.5, 0.5, "No plottable data available for the selected run", ha="center", va="center")

    _set_run_title(fig, run)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
