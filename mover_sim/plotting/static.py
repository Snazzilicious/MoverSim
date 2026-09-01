"""Static plotting renderers."""

from __future__ import annotations

import numpy as np

from mover_sim.math.coordinates import ecef_to_lla, ecef_to_enu, lla_to_ecef
from mover_sim.math.orientation import (
    rotate_vector_by_quaternion,
)
from mover_sim.plotting.events import map_events_to_positions, summarize_events
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
        raise ImportError(
            "plot_run_summary requires matplotlib to be installed. "
            "Install it with `pip install matplotlib`."
        ) from exc
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
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _plot_trajectory_panel(ax, platforms, colors, mapped_events=None):
    has_trajectory = False
    for platform_id, track in platforms.items():
        if track.position_ecef is None or len(track.position_ecef) == 0:
            continue
        position = np.asarray(track.position_ecef, dtype=float)
        has_trajectory = True
        ax.plot(
            position[:, 0],
            position[:, 1],
            position[:, 2],
            color=colors[platform_id],
            label=platform_id,
        )

    event_positions = [
        event["position_ecef"]
        for event in (mapped_events or [])
        if event["position_ecef"] is not None
    ]
    if event_positions:
        event_xyz = np.vstack(event_positions)
        ax.scatter(
            event_xyz[:, 0],
            event_xyz[:, 1],
            event_xyz[:, 2],
            color="black",
            marker="x",
            s=40,
            label="events",
            depthshade=False,
        )

    ax.set_title("Trajectory (ECEF)")
    ax.set_xlabel("ECEF X (m)")
    ax.set_ylabel("ECEF Y (m)")
    ax.set_zlabel("ECEF Z (m)")
    ax.grid(True)
    if has_trajectory and (len(platforms) > 1 or event_positions):
        ax.legend(loc="best")


def _plot_position_panel(ax, platforms, colors):
    components = ("lat", "lon", "alt")
    linestyles = ("-", "--", ":")
    for platform_id, track in platforms.items():
        lla = _track_lla(track)
        if lla is None or len(lla) == 0:
            continue
        y_scale = (1.0, 1.0, 1.0)
        for index, component in enumerate(components):
            ax.plot(
                track.time,
                lla[:, index] / y_scale[index],
                color=colors[platform_id],
                linestyle=linestyles[index],
                label=f"{platform_id} {component}",
            )

    ax.set_title("Position vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LLA (deg / m)")
    ax.grid(True)


def _track_lla(track):
    if track.lla is not None and len(track.lla) > 0:
        return np.asarray(track.lla, dtype=float)
    if track.position_ecef is None or len(track.position_ecef) == 0:
        return None

    position = np.asarray(track.position_ecef, dtype=float)
    lat, lon, alt = ecef_to_lla(position[:, 0], position[:, 1], position[:, 2])
    return np.column_stack([lat, lon, alt])


def _velocity_to_enu(track):
    if track.velocity_ecef is None or len(track.velocity_ecef) == 0:
        return None

    lla = _track_lla(track)
    if lla is None or len(lla) == 0:
        return None

    velocity_ecef = np.asarray(track.velocity_ecef, dtype=float)
    enu_velocity = np.empty_like(velocity_ecef)
    for index, (lat, lon, alt) in enumerate(lla):
        x_ref, y_ref, z_ref = lla_to_ecef(lat, lon, alt)
        east, north, up = ecef_to_enu(
            x_ref + velocity_ecef[index, 0],
            y_ref + velocity_ecef[index, 1],
            z_ref + velocity_ecef[index, 2],
            lat,
            lon,
            alt,
        )
        enu_velocity[index] = [east, north, up]
    return enu_velocity


def _plot_velocity_panel(ax, platforms, colors):
    components = ("east", "north", "up")
    for platform_id, track in platforms.items():
        velocity = _velocity_to_enu(track)
        if velocity is None or len(velocity) == 0:
            continue
        for index, component in enumerate(components):
            ax.plot(
                track.time,
                velocity[:, index],
                color=colors[platform_id],
                linestyle=("-", "--", ":")[index],
                label=f"{platform_id} {component}",
            )

        speed = np.linalg.norm(velocity, axis=1)
        ax.plot(
            track.time,
            speed,
            color=colors[platform_id],
            linestyle="-.",
            linewidth=1.4,
            label=f"{platform_id} speed",
        )

    ax.set_title("Velocity vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ENU Velocity (m/s)")
    ax.grid(True)


def _ecef_vector_to_enu(vector_ecef, lat_deg, lon_deg, alt_m):
    x_ref, y_ref, z_ref = lla_to_ecef(lat_deg, lon_deg, alt_m)
    east, north, up = ecef_to_enu(
        x_ref + vector_ecef[0],
        y_ref + vector_ecef[1],
        z_ref + vector_ecef[2],
        lat_deg,
        lon_deg,
        alt_m,
    )
    return np.array([east, north, up], dtype=float)


def _enu_body_axes_from_quaternion(quaternion, lat_deg, lon_deg, alt_m):
    forward_ecef = rotate_vector_by_quaternion([1.0, 0.0, 0.0], quaternion)
    right_ecef = rotate_vector_by_quaternion([0.0, 1.0, 0.0], quaternion)
    up_ecef = rotate_vector_by_quaternion([0.0, 0.0, 1.0], quaternion)
    return (
        _ecef_vector_to_enu(forward_ecef, lat_deg, lon_deg, alt_m),
        _ecef_vector_to_enu(right_ecef, lat_deg, lon_deg, alt_m),
        _ecef_vector_to_enu(up_ecef, lat_deg, lon_deg, alt_m),
    )


def _local_aircraft_attitude_from_enu_axes(forward_enu, right_enu):
    east, north, up = forward_enu
    horizontal_norm = np.hypot(east, north)
    yaw = np.arctan2(east, north)
    pitch = np.arctan2(up, horizontal_norm)

    local_up = np.array([0.0, 0.0, 1.0])
    if horizontal_norm < 1e-12:
        right_level = np.array([1.0, 0.0, 0.0])
        up_level = local_up
    else:
        forward_unit = forward_enu / np.linalg.norm(forward_enu)
        right_level = np.cross(local_up, forward_unit)
        right_level = right_level / np.linalg.norm(right_level)
        up_level = np.cross(forward_unit, right_level)
        up_level = up_level / np.linalg.norm(up_level)

    roll = np.arctan2(np.dot(right_enu, up_level), np.dot(right_enu, right_level))
    return np.array([roll, pitch, yaw])


def _orientation_to_local_rpy_degrees(track):
    lla = _track_lla(track)
    if lla is None or track.orientation is None:
        return None

    euler_radians = []
    for quaternion, (lat_deg, lon_deg, alt_m) in zip(np.asarray(track.orientation, dtype=float), lla):
        forward_enu, right_enu, _ = _enu_body_axes_from_quaternion(quaternion, lat_deg, lon_deg, alt_m)
        euler_radians.append(_local_aircraft_attitude_from_enu_axes(forward_enu, right_enu))

    euler_radians = np.vstack(euler_radians)
    return np.degrees(euler_radians)


def _plot_orientation_panel(ax, platforms, colors):
    components = ("roll", "pitch", "yaw")
    linestyles = ("-", "--", ":", "-.")
    for platform_id, track in platforms.items():
        if track.orientation is None or len(track.orientation) == 0:
            continue
        orientation = _orientation_to_local_rpy_degrees(track)
        if orientation is None or len(orientation) == 0:
            continue
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
    ax.set_ylabel("Angle (deg)")
    ax.grid(True)


def _add_event_markers(ax, events):
    for event in events:
        ax.axvline(event.time, color="0.6", alpha=0.35, linewidth=0.8, label="_nolegend_")


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
    """Render a composite static summary figure for one run.

    Parameters are selection-only; saving the returned figure is left to the caller.

    Example:
        ```python
        from mover_sim.plotting import load_hdf5_run, plot_run_summary

        run = load_hdf5_run("output/scenario_air_launched_cruise_missile.h5")
        fig = plot_run_summary(run, sections=["trajectory", "position", "velocity"])
        ```
    """

    plt, GridSpec = _import_matplotlib()

    platforms = select_platforms(run, platform_ids=platform_ids)
    events = filter_events(run, event_topics=event_topics, platform_ids=platform_ids)
    availability = panel_data_availability(platforms, events)
    enabled_sections = _resolve_sections(sections, availability)
    colors = assign_platform_colors(platforms)
    mapped_events = map_events_to_positions(run, event_topics=event_topics, platform_ids=platform_ids)

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
        _plot_trajectory_panel(trajectory_ax, platforms, colors, mapped_events=mapped_events)
        _set_3d_equalized_bounds(trajectory_ax, compute_ecef_equalized_bounds(platforms))

        events_ax = fig.add_subplot(gs[current_row, 1])
        _plot_event_panel(events_ax, run, event_topics, platform_ids)
        current_row += 1
    elif "trajectory" in enabled_sections:
        trajectory_ax = fig.add_subplot(gs[current_row, :], projection="3d")
        _plot_trajectory_panel(trajectory_ax, platforms, colors, mapped_events=mapped_events)
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
