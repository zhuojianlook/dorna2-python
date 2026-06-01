#!/usr/bin/env python3
"""Generate poster figures from local Dorna controller configuration.

The figures are SVGs so they can be used directly in posters. Most are
dependency-light; the offline workspace/manipulability panel uses numpy and the
local Dorna kinematic model when available.
"""

from __future__ import annotations

import json
import csv
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "poster_assets"
POSES_PATH = ROOT / "poses.json"
SETTINGS_PATH = ROOT / "settings.json"
LAUNCHER_PATH = ROOT / ".dorna_launcher.json"
CHARACTERIZATION_DIR = OUT / "characterization"
REPEATABILITY_DIR = OUT / "repeatability"
RIGHT_STICK_DEMO_DIR = OUT / "right_stick_demo"
ROUTINE_PATH = ROOT / "routine.txt"
ROUTINES_PATH = ROOT / "routines.json"


COLORS = {
    "ink": "#1f2933",
    "muted": "#52616b",
    "grid": "#d9e2ec",
    "blue": "#2f80ed",
    "green": "#219653",
    "red": "#d64545",
    "amber": "#f2a900",
    "violet": "#7b61ff",
    "cyan": "#22a6b3",
    "bg": "#ffffff",
}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x, y, text, size=14, color=None, anchor="start", weight="400"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{color or COLORS["ink"]}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{esc(text)}</text>'
    )


def write_svg(name: str, width: int, height: int, body: list[str]):
    OUT.mkdir(exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        *body,
        "</svg>",
    ]
    (OUT / name).write_text("\n".join(content) + "\n")


def metric_text(value, unit="", digits=2, missing="pending"):
    if value is None:
        return missing
    try:
        value = float(value)
    except Exception:
        return missing
    if math.isinf(value) or math.isnan(value):
        return missing
    if abs(value) >= 100:
        text = f"{value:.0f}"
    elif abs(value) >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.{digits}f}"
    return f"{text} {unit}".strip()


def linear_map(v, lo, hi, out_lo, out_hi):
    if hi == lo:
        return 0.5 * (out_lo + out_hi)
    return out_lo + (v - lo) * (out_hi - out_lo) / (hi - lo)


def parse_float(value):
    try:
        if value in ("", None):
            return None
        value = float(value)
        if math.isinf(value) or math.isnan(value):
            return None
        return value
    except Exception:
        return None


def latest_characterization_rows():
    if not CHARACTERIZATION_DIR.exists():
        return [], None
    candidates = sorted(
        CHARACTERIZATION_DIR.glob("*/motion_characterization.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return [], None
    path = candidates[0]
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for key in (
                "t", "sample", "alarm",
                "j0", "j1", "j2", "j3", "j4", "j5",
                "x", "y", "z", "a", "b", "c",
                "q_speed_norm_deg_s", "tcp_speed_norm_mm_s",
                "joint_target_error_norm_deg", "joint_target_error_max_deg",
                "sigma_min", "sigma_max", "condition", "manip_6d", "manip_trans",
            ):
                if key in row:
                    row[key] = parse_float(row.get(key))
            rows.append(row)
    return rows, path


def characterization_metadata(path):
    if not path:
        return {}
    meta_path = Path(path).with_name("metadata.json")
    return load_json(meta_path)


def latest_repeatability_rows():
    if not REPEATABILITY_DIR.exists():
        return [], None
    candidates = sorted(
        REPEATABILITY_DIR.glob("*/pose_repeatability.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return [], None
    path = candidates[0]
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for key in (
                "t", "cycle", "move_index", "settle_s", "alarm",
                "joint_error_norm_deg", "joint_error_max_deg",
                "j0", "j1", "j2", "j3", "j4", "j5",
                "target_j0", "target_j1", "target_j2", "target_j3", "target_j4", "target_j5",
                "x", "y", "z", "a", "b", "c",
            ):
                if key in row:
                    row[key] = parse_float(row.get(key))
            rows.append(row)
    return rows, path


def latest_right_stick_demo_rows():
    if not RIGHT_STICK_DEMO_DIR.exists():
        return [], None
    candidates = sorted(
        RIGHT_STICK_DEMO_DIR.glob("*/right_stick_tcp_demo.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return [], None
    path = candidates[0]
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for key in (
                "t", "right_stick_x", "right_stick_y", "sample", "alarm",
                "j0", "j1", "j2", "j3", "j4", "j5",
                "x", "y", "z", "a", "b", "c",
                "target_x", "target_y", "target_z", "target_a", "target_b", "target_c",
                "joint_delta_norm_deg", "joint_delta_max_deg",
                "tcp_drift_norm_mm", "tcp_drift_x_mm", "tcp_drift_y_mm", "tcp_drift_z_mm",
                "q_speed_norm_deg_s", "tcp_speed_norm_mm_s",
            ):
                if key in row:
                    row[key] = parse_float(row.get(key))
            rows.append(row)
    return rows, path


def load_routine_text():
    routines = load_json(ROUTINES_PATH)
    if isinstance(routines, dict) and isinstance(routines.get("Default"), str):
        return routines["Default"]
    try:
        return ROUTINE_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


def parse_routine_steps(text: str):
    steps = []
    for raw in str(text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            steps.append(line)
    return steps


def nominal_characterization_rows(poses: dict):
    default = poses.get("Default", {})
    reload = poses.get("Reload", {})
    if not default or not reload:
        return []
    sample_hz = 20.0
    vel = 20.0
    sequence = [
        ("nominal_default_hold", "Default", default, default),
        ("nominal_default_to_reload", "Reload", default, reload),
        ("nominal_reload_to_default", "Default", reload, default),
    ]
    rows = []
    t = 0.0
    for segment, target, start_pose, end_pose in sequence:
        dq_max = max(abs(float(end_pose[f"j{i}"]) - float(start_pose[f"j{i}"])) for i in range(6))
        duration = 1.0 if "hold" in segment else max(2.5, dq_max / vel + 1.0)
        n = max(2, int(duration * sample_hz))
        for idx in range(n):
            u = idx / max(1, n - 1)
            s = 3 * u * u - 2 * u * u * u
            row = {"t": t, "segment": segment, "target": target, "sample": idx, "alarm": 0}
            for j in range(6):
                key = f"j{j}"
                row[key] = float(start_pose[key]) + (float(end_pose[key]) - float(start_pose[key])) * s
            rows.append(row)
            t += 1.0 / sample_hz
    return rows


def points_from_rows(rows, x_key, y_key):
    pts = []
    for row in rows:
        x = parse_float(row.get(x_key))
        y = parse_float(row.get(y_key))
        if x is not None and y is not None:
            pts.append((x, y))
    return pts


def finite_diff_series(rows, y_key):
    pts = []
    prev_t = prev_y = None
    for row in rows:
        t = parse_float(row.get("t"))
        y = parse_float(row.get(y_key))
        if t is None or y is None:
            continue
        if prev_t is not None and t > prev_t:
            pts.append((t, (y - prev_y) / (t - prev_t)))
        prev_t, prev_y = t, y
    return pts


def derivative_norm_series(rows, keys):
    pts = []
    prev_t = None
    prev_vals = None
    for row in rows:
        t = parse_float(row.get("t"))
        vals = [parse_float(row.get(key)) for key in keys]
        if t is None or any(v is None for v in vals):
            continue
        if prev_t is not None and t > prev_t and prev_vals is not None:
            norm = math.sqrt(sum((vals[i] - prev_vals[i]) ** 2 for i in range(len(vals)))) / (t - prev_t)
            pts.append((t, norm))
        prev_t, prev_vals = t, vals
    return pts


def target_error_series(rows, path, metric_key):
    direct = points_from_rows(rows, "t", metric_key)
    if direct:
        return direct
    meta = characterization_metadata(path)
    pose_map = meta.get("poses", {}) if isinstance(meta, dict) else {}
    pts = []
    for row in rows:
        t = parse_float(row.get("t"))
        target = row.get("target")
        target_pose = pose_map.get(str(target), {}) if isinstance(pose_map, dict) else {}
        if t is None or not isinstance(target_pose, dict):
            continue
        diffs = []
        for idx in range(6):
            q = parse_float(row.get(f"j{idx}"))
            q_target = parse_float(target_pose.get(f"j{idx}"))
            if q is None or q_target is None:
                diffs = []
                break
            diffs.append(q - q_target)
        if not diffs:
            continue
        if metric_key == "joint_target_error_max_deg":
            value = max(abs(v) for v in diffs)
        else:
            value = math.sqrt(sum(v * v for v in diffs))
        pts.append((t, value))
    return pts


def normalize_series(pts):
    vals = [y for _x, y in pts if y is not None]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [(x, 1.0) for x, _y in pts]
    return [(x, (y - lo) / (hi - lo)) for x, y in pts if y is not None]


def normalize_to_peak(pts):
    vals = [abs(y) for _x, y in pts if y is not None]
    peak = max(vals) if vals else 0.0
    if peak <= 0.0:
        return []
    return [(x, y / peak) for x, y in pts if y is not None]


def peak_point(pts):
    clean = [(x, y) for x, y in pts if y is not None]
    if not clean:
        return None
    return max(clean, key=lambda p: abs(p[1]))


def characterization_stats(rows):
    if not rows:
        return {}
    times = [parse_float(row.get("t")) for row in rows]
    times = [t for t in times if t is not None]
    if not times:
        return {}

    def values(key):
        vals = [parse_float(row.get(key)) for row in rows]
        return [v for v in vals if v is not None]

    xyz = {key: values(key) for key in ("x", "y", "z")}
    q_speed = values("q_speed_norm_deg_s")
    tcp_speed = values("tcp_speed_norm_mm_s")
    sigma_min = values("sigma_min")
    manip_6d = values("manip_6d")
    alarms = values("alarm")

    phases = []
    cur = None
    for row in rows:
        t = parse_float(row.get("t"))
        seg = row.get("segment")
        if t is None or not seg:
            continue
        if cur is None or cur["segment"] != seg:
            if cur is not None:
                phases.append(cur)
            cur = {"segment": seg, "start": t, "end": t}
        else:
            cur["end"] = t
    if cur is not None:
        phases.append(cur)

    return {
        "samples": len(rows),
        "duration": max(times) - min(times),
        "phase_count": len(phases),
        "alarm_max": max(alarms) if alarms else None,
        "max_q_speed": max(q_speed) if q_speed else None,
        "max_tcp_speed": max(tcp_speed) if tcp_speed else None,
        "x_excursion": max(xyz["x"]) - min(xyz["x"]) if xyz["x"] else None,
        "y_excursion": max(xyz["y"]) - min(xyz["y"]) if xyz["y"] else None,
        "z_excursion": max(xyz["z"]) - min(xyz["z"]) if xyz["z"] else None,
        "sigma_min_low": min(sigma_min) if sigma_min else None,
        "manip_6d_low": min(manip_6d) if manip_6d else None,
        "jacobian_rows": len(sigma_min),
    }


def grouped_rows(rows, key):
    groups = {}
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        groups.setdefault(str(value), []).append(row)
    return groups


def make_line_chart(
    name,
    title,
    subtitle,
    series,
    ylabel,
    xlabel="time (s)",
    width=980,
    height=450,
    no_data_message=None,
):
    left, right, top, bottom = 76, 28, 76, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_pts = [p for _, _, pts in series for p in pts if p[1] is not None]
    body = [
        svg_text(28, 34, title, 21, weight="700"),
        svg_text(28, 56, subtitle, 13, COLORS["muted"]),
    ]
    if not all_pts:
        body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(
            width / 2,
            top + plot_h / 2,
            no_data_message or "No measured data yet. Use the launcher Motion Characterization button, then rerun this generator.",
            15,
            COLORS["muted"],
            anchor="middle",
        ))
        write_svg(name, width, height, body)
        return

    x_vals = [p[0] for p in all_pts]
    y_vals = [p[1] for p in all_pts]
    x0, x1 = min(x_vals), max(x_vals)
    y0, y1 = min(y_vals), max(y_vals)
    if x0 == x1:
        x1 = x0 + 1.0
    if y0 == y1:
        pad = max(1.0, abs(y0) * 0.1)
        y0 -= pad
        y1 += pad
    y_pad = (y1 - y0) * 0.08
    y0 -= y_pad
    y1 += y_pad

    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    for i in range(6):
        gx = left + plot_w * i / 5
        gy = top + plot_h * i / 5
        body.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}"/>')
        xv = x0 + (x1 - x0) * i / 5
        yv = y1 - (y1 - y0) * i / 5
        body.append(svg_text(gx, top + plot_h + 20, f"{xv:.1f}", 11, COLORS["muted"], anchor="middle"))
        body.append(svg_text(left - 8, gy + 4, f"{yv:.2g}", 11, COLORS["muted"], anchor="end"))

    def sx(x):
        return linear_map(x, x0, x1, left, left + plot_w)

    def sy(y):
        return linear_map(y, y0, y1, top + plot_h, top)

    legend_x = left + 10
    legend_y = top + 16
    for idx, (label, color, pts) in enumerate(series):
        clean = [(sx(x), sy(y)) for x, y in pts if y is not None]
        if len(clean) >= 2:
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in clean)
            body.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        elif len(clean) == 1:
            body.append(f'<circle cx="{clean[0][0]:.1f}" cy="{clean[0][1]:.1f}" r="3" fill="{color}"/>')
        lx = legend_x + (idx % 3) * 150
        ly = legend_y + (idx // 3) * 18
        body.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 20}" y2="{ly - 4}" stroke="{color}" stroke-width="3"/>')
        body.append(svg_text(lx + 26, ly, label, 11, COLORS["ink"]))

    body.append(svg_text(left + plot_w / 2, height - 18, xlabel, 13, COLORS["ink"], anchor="middle"))
    body.append(svg_text(20, top + plot_h / 2, ylabel, 13, COLORS["ink"], anchor="middle"))
    write_svg(name, width, height, body)


def make_endpoint_repeatability(rows, path):
    width, height = 1040, 520
    body = [
        svg_text(28, 34, "Figure 16. Settled endpoint repeatability", 21, weight="700"),
        svg_text(
            28,
            56,
            f"Latest repeatability log: {path.relative_to(ROOT)}" if path else
            "Run launcher Repeatability Study to measure repeated settled endpoints.",
            13,
            COLORS["muted"],
        ),
    ]
    groups = grouped_rows(rows, "target")
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        body.append(f'<rect x="72" y="92" width="894" height="300" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, 238, "No repeatability dataset yet.", 17, COLORS["muted"], anchor="middle"))
        body.append(svg_text(width / 2, 268, "Use the startup launcher Repeatability Study button, then rerun this generator.", 14, COLORS["muted"], anchor="middle"))
        write_svg("endpoint_repeatability.svg", width, height, body)
        return

    summaries = []
    all_centered = []
    for target, target_rows in groups.items():
        pts = []
        for row in target_rows:
            x = parse_float(row.get("x"))
            y = parse_float(row.get("y"))
            z = parse_float(row.get("z"))
            if x is not None and y is not None and z is not None:
                pts.append((x, y, z, row))
        if len(pts) < 2:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        centered = []
        radial = []
        settle = []
        joint_residual = []
        for x, y, z, row in pts:
            dx, dy, dz = x - cx, y - cy, z - cz
            r3 = math.sqrt(dx * dx + dy * dy + dz * dz)
            centered.append((dx, dy, dz))
            radial.append(r3)
            val = parse_float(row.get("settle_s"))
            if val is not None:
                settle.append(val)
            val = parse_float(row.get("joint_error_norm_deg"))
            if val is not None:
                joint_residual.append(val)
        all_centered.extend((target, dx, dy, dz) for dx, dy, dz in centered)
        summaries.append({
            "target": target,
            "n": len(centered),
            "rms_mm": math.sqrt(sum(v * v for v in radial) / len(radial)),
            "max_mm": max(radial),
            "mean_settle_s": sum(settle) / len(settle) if settle else None,
            "mean_joint_error_deg": sum(joint_residual) / len(joint_residual) if joint_residual else None,
            "centered": centered,
        })

    if not summaries:
        body.append(f'<rect x="72" y="92" width="894" height="300" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, 238, "Repeatability rows exist, but no complete TCP endpoints were found.", 15, COLORS["muted"], anchor="middle"))
        write_svg("endpoint_repeatability.svg", width, height, body)
        return

    max_abs = max(max(abs(dx), abs(dy)) for _target, dx, dy, _dz in all_centered) if all_centered else 1.0
    max_abs = max(max_abs, 0.05)
    plot_left, plot_top, plot_w, plot_h = 72, 96, 430, 330
    body.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    for i in range(5):
        x = plot_left + plot_w * i / 4
        y = plot_top + plot_h * i / 4
        body.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_top + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')

    def sx(dx):
        return linear_map(dx, -max_abs, max_abs, plot_left + 22, plot_left + plot_w - 22)

    def sy(dy):
        return linear_map(dy, -max_abs, max_abs, plot_top + plot_h - 22, plot_top + 22)

    target_colors = [COLORS["blue"], COLORS["green"], COLORS["violet"], COLORS["amber"], COLORS["red"], COLORS["cyan"]]
    for idx, summary in enumerate(summaries):
        color = target_colors[idx % len(target_colors)]
        for dx, dy, _dz in summary["centered"]:
            body.append(f'<circle cx="{sx(dx):.1f}" cy="{sy(dy):.1f}" r="4" fill="{color}" opacity="0.78"/>')
        lx = plot_left + 12 + (idx % 2) * 180
        ly = plot_top + 18 + (idx // 2) * 18
        body.append(f'<circle cx="{lx}" cy="{ly - 4}" r="4" fill="{color}"/>')
        body.append(svg_text(lx + 12, ly, summary["target"], 11, COLORS["ink"]))
    body.append(svg_text(plot_left + plot_w / 2, plot_top + plot_h + 34, "centered x error (mm)", 13, anchor="middle"))
    body.append(svg_text(24, plot_top + plot_h / 2, "centered y error (mm)", 13, anchor="middle"))
    body.append(svg_text(plot_left + plot_w - 6, plot_top + plot_h + 18, f"+/- {max_abs:.3g} mm", 11, COLORS["muted"], anchor="end"))

    card_x, card_y = 550, 106
    row_h = 74
    for idx, summary in enumerate(summaries[:4]):
        y = card_y + idx * row_h
        color = target_colors[idx % len(target_colors)]
        body.append(f'<rect x="{card_x}" y="{y}" width="410" height="58" rx="5" fill="#f8fafc" stroke="{color}" stroke-width="2"/>')
        body.append(svg_text(card_x + 14, y + 23, summary["target"], 13, color, weight="700"))
        body.append(svg_text(card_x + 150, y + 23, f"n={summary['n']}", 12, COLORS["muted"]))
        body.append(svg_text(card_x + 14, y + 45, f"RMS {summary['rms_mm']:.3g} mm   max {summary['max_mm']:.3g} mm", 14, COLORS["ink"], weight="700"))
        extra = []
        if summary["mean_settle_s"] is not None:
            extra.append(f"settle {summary['mean_settle_s']:.2f}s")
        if summary["mean_joint_error_deg"] is not None:
            extra.append(f"joint residual {summary['mean_joint_error_deg']:.3g} deg")
        if extra:
            body.append(svg_text(card_x + 230, y + 45, " | ".join(extra), 11, COLORS["muted"]))

    body.append(svg_text(width / 2, 482, "Use this figure to support repeatability claims; it is more defensible than showing only stored pose values.", 14, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("endpoint_repeatability.svg", width, height, body)


def make_target_tracking_residual(rows, path, source_label):
    make_line_chart(
        "target_tracking_residual.svg",
        "Figure 17. Joint target residual during pose transitions",
        source_label,
        [
            ("||q - q_target||", COLORS["red"], target_error_series(rows, path, "joint_target_error_norm_deg")),
            ("max axis residual", COLORS["amber"], target_error_series(rows, path, "joint_target_error_max_deg")),
        ],
        "joint residual (deg)",
        no_data_message="No target residual can be computed. Rerun Motion Characterization with the updated launcher logger.",
    )


def make_characterization_summary(rows, path, dexterity_metrics=None):
    width, height = 980, 520
    stats = characterization_stats(rows)
    body = [
        svg_text(28, 34, "Figure 13. Measured characterization summary", 21, weight="700"),
        svg_text(
            28,
            56,
            f"Latest log: {path.relative_to(ROOT)}" if path else "Run launcher Motion Characterization to populate this quantitative panel.",
            13,
            COLORS["muted"],
        ),
    ]
    if not stats:
        body.append(f'<rect x="64" y="94" width="852" height="320" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, 255, "No measured characterization log found.", 16, COLORS["muted"], anchor="middle"))
        write_svg("characterization_summary.svg", width, height, body)
        return

    sigma_low = stats["sigma_min_low"]
    manip_low = stats["manip_6d_low"]
    dexterity_source = "pending"
    dexterity_color = COLORS["amber"]
    if dexterity_metrics and dexterity_metrics.get("available"):
        sigma_vals = [v for _t, v in dexterity_metrics.get("sigma_min", [])]
        manip_vals = [v for _t, v in dexterity_metrics.get("manip", [])]
        if sigma_vals:
            sigma_low = min(sigma_vals)
        if manip_vals:
            manip_low = min(manip_vals)
        source = dexterity_metrics.get("source", "")
        if str(source).startswith("logged"):
            dexterity_source = "logged"
            dexterity_color = COLORS["green"]
        else:
            dexterity_source = "offline FK"
            dexterity_color = COLORS["blue"]
    elif stats.get("jacobian_rows"):
        dexterity_source = "logged"
        dexterity_color = COLORS["green"]

    cards = [
        ("Samples", metric_text(stats["samples"], "", 0), COLORS["blue"]),
        ("Duration", metric_text(stats["duration"], "s"), COLORS["green"]),
        ("Phases", metric_text(stats["phase_count"], "", 0), COLORS["violet"]),
        ("Alarm latch", "none" if stats["alarm_max"] == 0 else metric_text(stats["alarm_max"], "", 0), COLORS["red"] if stats["alarm_max"] else COLORS["green"]),
        ("Max ||dq/dt||", metric_text(stats["max_q_speed"], "deg/s"), COLORS["violet"]),
        ("Max ||dx/dt||", metric_text(stats["max_tcp_speed"], "mm/s"), COLORS["blue"]),
        ("TCP x span", metric_text(stats["x_excursion"], "mm"), COLORS["cyan"]),
        ("TCP y span", metric_text(stats["y_excursion"], "mm"), COLORS["cyan"]),
        ("TCP z span", metric_text(stats["z_excursion"], "mm"), COLORS["cyan"]),
        ("Dexterity source", dexterity_source, dexterity_color),
        ("Min sigma", metric_text(sigma_low, ""), COLORS["red"]),
        ("Min manip.", metric_text(manip_low, ""), COLORS["red"]),
    ]
    x0, y0, w, h = 54, 94, 205, 82
    gap_x, gap_y = 28, 28
    for idx, (label, value, color) in enumerate(cards):
        col = idx % 4
        row = idx // 4
        x = x0 + col * (w + gap_x)
        y = y0 + row * (h + gap_y)
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="#f8fafc" stroke="{color}" stroke-width="2"/>')
        body.append(svg_text(x + 14, y + 26, label, 13, COLORS["muted"], weight="700"))
        body.append(svg_text(x + 14, y + 58, value, 22, COLORS["ink"], weight="700"))

    note = "Interpretation: this panel separates measured robot behavior from future biological endpoint validation."
    if dexterity_metrics and dexterity_metrics.get("available"):
        note = f"Dexterity metrics use {dexterity_metrics.get('source')}; trajectory and speed are measured."
    elif not stats.get("jacobian_rows"):
        note = "Note: trajectory/speed data are measured; install numpy/Dorna model or rerun characterization for dexterity metrics."
    body.append(svg_text(width / 2, 462, note, 14, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("characterization_summary.svg", width, height, body)


def make_phase_timeline(rows, source_label):
    width, height = 980, 430
    left, right, top = 170, 270, 72
    bar_h, gap = 26, 12
    segments = []
    cur = None
    for row in rows:
        t = parse_float(row.get("t"))
        seg = row.get("segment", "")
        if t is None or not seg:
            continue
        if cur is None or cur["segment"] != seg:
            if cur is not None:
                segments.append(cur)
            cur = {"segment": seg, "start": t, "end": t}
        else:
            cur["end"] = t
    if cur is not None:
        segments.append(cur)
    t0 = min((s["start"] for s in segments), default=0.0)
    t1 = max((s["end"] for s in segments), default=1.0)
    if t1 <= t0:
        t1 = t0 + 1.0
    plot_w = width - left - right
    body = [
        svg_text(28, 34, "Figure 9. Supervisory phase timeline", 21, weight="700"),
        svg_text(28, 56, source_label, 13, COLORS["muted"]),
    ]
    durations = [(seg["segment"], max(0.0, seg["end"] - seg["start"])) for seg in segments]
    total_duration = max(0.0, t1 - t0)
    dwell_duration = sum(d for name, d in durations if "baseline" in name or "hold" in name or "settle" in name)
    transit_duration = max(0.0, total_duration - dwell_duration)
    longest = max(durations, key=lambda item: item[1]) if durations else ("none", 0.0)
    card_x, card_y, card_w = 742, 90, 200
    cards = [
        ("Total cycle", f"{total_duration:.2f} s"),
        ("Transit fraction", f"{(100.0 * transit_duration / total_duration):.0f}%" if total_duration else "pending"),
        ("Dwell/hold", f"{dwell_duration:.2f} s"),
        ("Longest phase", f"{longest[1]:.2f} s"),
    ]
    for idx, (label, value) in enumerate(cards):
        y = card_y + idx * 62
        body.append(f'<rect x="{card_x}" y="{y}" width="{card_w}" height="48" rx="5" fill="#f8fafc" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(card_x + 12, y + 19, label, 12, COLORS["muted"], weight="700"))
        body.append(svg_text(card_x + 12, y + 40, value, 17, COLORS["ink"], weight="700"))
    body.append(svg_text(card_x + 12, card_y + 4 * 62 + 20, longest[0].replace("_", " "), 11, COLORS["muted"]))
    colors = [COLORS["green"], COLORS["blue"], COLORS["violet"], COLORS["amber"], COLORS["cyan"]]
    for idx, seg in enumerate(segments):
        y = top + idx * (bar_h + gap)
        x = linear_map(seg["start"], t0, t1, left, left + plot_w)
        x2 = linear_map(seg["end"], t0, t1, left, left + plot_w)
        body.append(svg_text(left - 10, y + 18, seg["segment"].replace("_", " "), 12, anchor="end"))
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{max(1.0, x2 - x):.1f}" height="{bar_h}" fill="{colors[idx % len(colors)]}"/>')
        body.append(svg_text(x + 4, y + 18, f"{seg['end'] - seg['start']:.2f}s", 11, "#ffffff"))
    axis_y = top + len(segments) * (bar_h + gap) + 10
    body.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="{COLORS["grid"]}"/>')
    body.append(svg_text(left, axis_y + 20, f"{t0:.1f}", 11, COLORS["muted"], anchor="middle"))
    body.append(svg_text(left + plot_w, axis_y + 20, f"{t1:.1f} s", 11, COLORS["muted"], anchor="middle"))
    body.append(svg_text(left + plot_w / 2, axis_y + 42, "phase timing supports cycle-time and supervision-duty reporting", 12, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("characterization_phase_timeline.svg", width, max(height, axis_y + 64), body)


def make_tcp_path(rows, source_label):
    pts = []
    for row in rows:
        t = parse_float(row.get("t"))
        x = parse_float(row.get("x"))
        y = parse_float(row.get("y"))
        z = parse_float(row.get("z"))
        if x is not None and y is not None:
            pts.append((t, x, y, z))
    width, height = 980, 580
    left, right, top, bottom = 82, 340, 88, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    body = [
        svg_text(28, 34, "Figure 7. TCP task-space trajectory and return closure", 21, weight="700"),
        svg_text(28, 56, source_label, 13, COLORS["muted"]),
    ]
    if len(pts) < 2:
        body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, top + plot_h / 2, "No TCP measurements yet. Run launcher Motion Characterization.", 15, COLORS["muted"], anchor="middle"))
        write_svg("characterization_tcp_path.svg", width, height, body)
        return
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    zs = [p[3] for p in pts if p[3] is not None]
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    if lo_x == hi_x:
        lo_x -= 1
        hi_x += 1
    if lo_y == hi_y:
        lo_y -= 1
        hi_y += 1
    pad_x = (hi_x - lo_x) * 0.08
    pad_y = (hi_y - lo_y) * 0.08
    lo_x -= pad_x
    hi_x += pad_x
    lo_y -= pad_y
    hi_y += pad_y
    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    for i in range(6):
        gx = left + plot_w * i / 5
        gy = top + plot_h * i / 5
        body.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}"/>')
        xv = lo_x + (hi_x - lo_x) * i / 5
        yv = hi_y - (hi_y - lo_y) * i / 5
        body.append(svg_text(gx, top + plot_h + 20, f"{xv:.0f}", 10, COLORS["muted"], anchor="middle"))
        body.append(svg_text(left - 8, gy + 4, f"{yv:.0f}", 10, COLORS["muted"], anchor="end"))
    coords = [
        (
            linear_map(x, lo_x, hi_x, left, left + plot_w),
            linear_map(y, lo_y, hi_y, top + plot_h, top),
        )
        for _t, x, y, _z in pts
    ]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    body.append(f'<polyline points="{d}" fill="none" stroke="{COLORS["blue"]}" stroke-width="2.5"/>')
    body.append(f'<circle cx="{coords[0][0]:.1f}" cy="{coords[0][1]:.1f}" r="5" fill="{COLORS["green"]}"/>')
    body.append(f'<circle cx="{coords[-1][0]:.1f}" cy="{coords[-1][1]:.1f}" r="5" fill="{COLORS["red"]}"/>')
    body.append(svg_text(coords[0][0] + 8, coords[0][1] - 8, "start", 11, COLORS["green"], weight="700"))
    body.append(svg_text(coords[-1][0] + 8, coords[-1][1] + 14, "end", 11, COLORS["red"], weight="700"))
    body.append(svg_text(left + plot_w / 2, height - 20, "x (mm)", 13, anchor="middle"))
    body.append(svg_text(24, top + plot_h / 2, "y (mm)", 13, anchor="middle"))

    def dist3(a, b):
        if a[3] is None or b[3] is None:
            return math.hypot(b[1] - a[1], b[2] - a[2])
        return math.sqrt((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2 + (b[3] - a[3]) ** 2)

    path_length = sum(dist3(a, b) for a, b in zip(pts, pts[1:]))
    closure = dist3(pts[0], pts[-1])
    card_x, card_y, card_w = 674, 96, 260
    cards = [
        ("Integrated TCP path", f"{path_length:.1f} mm", "sample-to-sample FK arc"),
        ("Return closure", f"{closure:.3g} mm", "start/end TCP distance"),
        ("x-y envelope", f"{max(xs) - min(xs):.1f} x {max(ys) - min(ys):.1f} mm", "horizontal workspace span"),
        ("z span", f"{(max(zs) - min(zs)):.1f} mm" if zs else "pending", "vertical task motion"),
        ("Samples", f"{len(pts)}", "feedback-derived TCP poses"),
    ]
    for idx, (label, value, detail) in enumerate(cards):
        y = card_y + idx * 62
        body.append(f'<rect x="{card_x}" y="{y}" width="{card_w}" height="50" rx="5" fill="#f8fafc" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(card_x + 12, y + 19, label, 12, COLORS["muted"], weight="700"))
        body.append(svg_text(card_x + 12, y + 40, value, 17, COLORS["ink"], weight="700"))
        body.append(svg_text(card_x + 158, y + 40, detail, 10, COLORS["muted"]))

    z_pts = [(t, z) for t, _x, _y, z in pts if t is not None and z is not None]
    if len(z_pts) >= 2:
        z_left, z_top, z_w, z_h = card_x, 430, card_w, 92
        t0, t1 = min(t for t, _z in z_pts), max(t for t, _z in z_pts)
        z0, z1 = min(z for _t, z in z_pts), max(z for _t, z in z_pts)
        if z0 == z1:
            z0 -= 1.0
            z1 += 1.0
        body.append(svg_text(z_left, z_top - 12, "z(t) profile", 12, COLORS["muted"], weight="700"))
        body.append(f'<rect x="{z_left}" y="{z_top}" width="{z_w}" height="{z_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        z_poly = " ".join(
            f"{linear_map(t, t0, t1, z_left + 8, z_left + z_w - 8):.1f},"
            f"{linear_map(z, z0, z1, z_top + z_h - 8, z_top + 8):.1f}"
            for t, z in z_pts
        )
        body.append(f'<polyline points="{z_poly}" fill="none" stroke="{COLORS["violet"]}" stroke-width="2"/>')
        body.append(svg_text(z_left + z_w, z_top + z_h + 18, f"{t1 - t0:.1f} s", 10, COLORS["muted"], anchor="end"))
        body.append(svg_text(z_left, z_top + z_h + 18, f"{z0:.0f}-{z1:.0f} mm", 10, COLORS["muted"]))
    write_svg("characterization_tcp_path.svg", width, height, body)


def make_speed_norms_figure(rows, source_label):
    q_pts = points_from_rows(rows, "t", "q_speed_norm_deg_s") or derivative_norm_series(rows, [f"j{i}" for i in range(6)])
    tcp_pts = points_from_rows(rows, "t", "tcp_speed_norm_mm_s") or derivative_norm_series(rows, ["x", "y", "z"])
    q_norm = normalize_to_peak(q_pts)
    tcp_norm = normalize_to_peak(tcp_pts)
    width, height = 980, 460
    left, right, top, bottom = 78, 300, 78, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    body = [
        svg_text(28, 34, "Figure 8. Peak-normalized joint and TCP speed envelopes", 21, weight="700"),
        svg_text(28, 56, f"{source_label} Curves are normalized separately; physical units are reported at right.", 13, COLORS["muted"]),
    ]
    all_pts = q_norm + tcp_norm
    if not all_pts:
        body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, top + plot_h / 2, "No speed data available.", 15, COLORS["muted"], anchor="middle"))
        write_svg("characterization_speed_norms.svg", width, height, body)
        return

    x0 = min(x for x, _y in all_pts)
    x1 = max(x for x, _y in all_pts)
    if x0 == x1:
        x1 = x0 + 1.0
    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    for i in range(6):
        gx = left + plot_w * i / 5
        gy = top + plot_h * i / 5
        body.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(gx, top + plot_h + 20, f"{x0 + (x1 - x0) * i / 5:.1f}", 11, COLORS["muted"], anchor="middle"))
        body.append(svg_text(left - 8, gy + 4, f"{1.0 - i / 5:.1f}", 11, COLORS["muted"], anchor="end"))

    def sx(t):
        return linear_map(t, x0, x1, left, left + plot_w)

    def sy(v):
        return linear_map(v, 0.0, 1.0, top + plot_h, top)

    for label, color, pts in (
        ("||dq/dt|| / peak", COLORS["violet"], q_norm),
        ("||dx/dt|| / peak", COLORS["blue"], tcp_norm),
    ):
        clean = [(sx(t), sy(v)) for t, v in pts]
        if len(clean) < 2:
            continue
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in clean)
        body.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lx = left + 12 if "dq" in label else left + 200
        body.append(f'<line x1="{lx}" y1="{top + 18}" x2="{lx + 24}" y2="{top + 18}" stroke="{color}" stroke-width="3"/>')
        body.append(svg_text(lx + 30, top + 22, label, 12, COLORS["ink"]))

    q_peak = peak_point(q_pts)
    tcp_peak = peak_point(tcp_pts)
    card_x, card_y, card_w = 720, 96, 218
    cards = [
        ("Peak joint speed", metric_text(q_peak[1], "deg/s") if q_peak else "pending", f"t={q_peak[0]:.2f}s" if q_peak else "run characterization"),
        ("Peak TCP speed", metric_text(tcp_peak[1], "mm/s") if tcp_peak else "pending", f"t={tcp_peak[0]:.2f}s" if tcp_peak else "requires TCP FK"),
        ("Interpretation", "shape, not units", "separate normalization"),
    ]
    for idx, (label, value, detail) in enumerate(cards):
        y = card_y + idx * 82
        body.append(f'<rect x="{card_x}" y="{y}" width="{card_w}" height="62" rx="5" fill="#f8fafc" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(card_x + 12, y + 22, label, 12, COLORS["muted"], weight="700"))
        body.append(svg_text(card_x + 12, y + 46, value, 18, COLORS["ink"], weight="700"))
        body.append(svg_text(card_x + 124, y + 46, detail, 10, COLORS["muted"]))
    body.append(svg_text(left + plot_w / 2, height - 20, "time (s)", 13, anchor="middle"))
    body.append(svg_text(22, top + plot_h / 2, "normalized speed", 13, anchor="middle"))
    write_svg("characterization_speed_norms.svg", width, height, body)


def make_right_stick_decoupling_demo(rows, path):
    width, height = 980, 460
    left, right, top, bottom = 78, 300, 78, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    source_label = (
        f"Measured log: {path.parent.name}/right_stick_tcp_demo.csv." if path else
        "Run launcher Right-Stick TCP Demo to collect fixed-TCP orientation data."
    )
    body = [
        svg_text(28, 34, "Figure 19. Right-stick fixed-TCP response envelopes", 21, weight="700"),
        svg_text(
            28,
            56,
            f"{source_label} Joint response is peak-normalized; TCP drift is shown relative to 1 mm.",
            13,
            COLORS["muted"],
        ),
    ]
    if not rows:
        body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, top + plot_h / 2 - 12, "No right-stick fixed-TCP demo dataset yet.", 17, COLORS["muted"], anchor="middle"))
        body.append(svg_text(width / 2, top + plot_h / 2 + 18, "Use the startup launcher Right-Stick TCP Demo button, then rerun this generator.", 14, COLORS["muted"], anchor="middle"))
        write_svg("right_stick_tcp_decoupling.svg", width, height, body)
        return

    command_colors = {
        "up": COLORS["green"],
        "down": COLORS["amber"],
        "left": COLORS["violet"],
        "right": COLORS["blue"],
    }

    def normalized_command(value):
        value = str(value or "").strip().lower()
        if value.startswith("right_stick_"):
            value = value.replace("right_stick_", "", 1)
        return value if value in command_colors else None

    records = []
    for row in rows:
        t = parse_float(row.get("t"))
        q = parse_float(row.get("joint_delta_norm_deg"))
        drift = parse_float(row.get("tcp_drift_norm_mm"))
        if t is None or q is None or drift is None:
            continue
        records.append({
            "t": t,
            "q": q,
            "drift": drift,
            "command": normalized_command(row.get("command")),
            "segment": row.get("segment"),
            "q_speed": parse_float(row.get("q_speed_norm_deg_s")),
            "tcp_speed": parse_float(row.get("tcp_speed_norm_mm_s")),
        })

    if len(records) < 2:
        body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, top + plot_h / 2, "Right-stick demo rows exist, but joint/TCP metrics are incomplete.", 15, COLORS["muted"], anchor="middle"))
        write_svg("right_stick_tcp_decoupling.svg", width, height, body)
        return

    t_origin = min(item["t"] for item in records)
    for item in records:
        item["tr"] = item["t"] - t_origin
    x0 = 0.0
    x1 = max(item["tr"] for item in records)
    if x1 <= x0:
        x1 = x0 + 1.0

    max_joint = max(item["q"] for item in records)
    max_drift = max(item["drift"] for item in records)
    q_peak = max(records, key=lambda item: item["q"])
    drift_peak = max(records, key=lambda item: item["drift"])
    q_ref = max_joint if max_joint > 0 else 1.0
    drift_ref = 1.0
    q_series = [(item["tr"], item["q"] / q_ref) for item in records]
    drift_series = [(item["tr"], item["drift"] / drift_ref) for item in records]
    y_hi = max(1.0, *(v for _t, v in q_series), *(v for _t, v in drift_series))
    y_hi = math.ceil(y_hi * 10.0) / 10.0

    def sx(t):
        return linear_map(t, x0, x1, left, left + plot_w)

    def sy(v):
        return linear_map(v, 0.0, y_hi, top + plot_h, top)

    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')

    intervals = []
    active = None
    start_t = None
    prev_t = None
    for item in records:
        cmd = item["command"]
        if cmd != active:
            if active and start_t is not None and prev_t is not None and prev_t > start_t:
                intervals.append((active, start_t, prev_t))
            active = cmd
            start_t = item["tr"] if cmd else None
        prev_t = item["tr"]
    if active and start_t is not None and prev_t is not None and prev_t > start_t:
        intervals.append((active, start_t, prev_t))

    for cmd, start, end in intervals:
        x = sx(start)
        w = max(1.0, sx(end) - x)
        body.append(f'<rect x="{x:.1f}" y="{top}" width="{w:.1f}" height="{plot_h}" fill="{command_colors[cmd]}" opacity="0.08"/>')
        if w > 30:
            body.append(svg_text(x + w / 2, top - 8, cmd, 10, command_colors[cmd], anchor="middle", weight="700"))

    for i in range(6):
        gx = left + plot_w * i / 5
        gy = top + plot_h * i / 5
        body.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(gx, top + plot_h + 20, f"{x0 + (x1 - x0) * i / 5:.1f}", 11, COLORS["muted"], anchor="middle"))
        body.append(svg_text(left - 8, gy + 4, f"{y_hi * (1.0 - i / 5):.1f}", 11, COLORS["muted"], anchor="end"))

    def draw_series(points, color, width_px=2.5):
        clean = [(sx(t), sy(v)) for t, v in points]
        if len(clean) < 2:
            return
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in clean)
        body.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="{width_px}"/>')

    draw_series(q_series, COLORS["violet"], 2.7)
    draw_series(drift_series, COLORS["blue"], 2.7)

    for lx, label, color in (
        (left + 12, "||q - q0|| / peak", COLORS["violet"]),
        (left + 205, "||p - p0|| / 1 mm", COLORS["blue"]),
    ):
        body.append(f'<line x1="{lx}" y1="{top + 18}" x2="{lx + 24}" y2="{top + 18}" stroke="{color}" stroke-width="3"/>')
        body.append(svg_text(lx + 30, top + 22, label, 12, COLORS["ink"]))

    q_speed_values = [item["q_speed"] for item in records if item["q_speed"] is not None]
    tcp_speed_values = [item["tcp_speed"] for item in records if item["tcp_speed"] is not None]
    baseline_drift = [
        item["drift"]
        for item in records
        if item.get("segment") == "baseline_fixed_tcp"
    ]
    ratio = max_drift / max_joint if max_joint else None

    card_x, card_y, card_w = 720, 88, 218
    cards = [
        ("Samples", f"{len(records)}", f"{x1:.2f} s sequence"),
        ("Peak joint motion", f"{max_joint:.2f} deg", f"t={q_peak['tr']:.2f}s"),
        ("Peak TCP drift", f"{max_drift:.3f} mm", f"{max_drift / drift_ref:.2f} of 1 mm"),
        ("Drift / joint", f"{ratio:.3g} mm/deg" if ratio is not None else "pending", "peak ratio"),
        ("Baseline drift", f"{max(baseline_drift):.3f} mm" if baseline_drift else "pending", "before pulses"),
        (
            "Peak rates",
            f"{max(q_speed_values):.1f} deg/s" if q_speed_values else "pending",
            f"{max(tcp_speed_values):.2f} mm/s residual" if tcp_speed_values else "",
        ),
    ]
    for idx, (label, value, detail) in enumerate(cards):
        y = card_y + idx * 56
        body.append(f'<rect x="{card_x}" y="{y}" width="{card_w}" height="48" rx="5" fill="#f8fafc" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(card_x + 12, y + 17, label, 11, COLORS["muted"], weight="700"))
        body.append(svg_text(card_x + 12, y + 36, value, 15, COLORS["ink"], weight="700"))
        if detail:
            body.append(svg_text(card_x + 126, y + 36, detail, 9, COLORS["muted"]))

    body.append(svg_text(left + plot_w / 2, height - 20, "time from demo start (s)", 13, anchor="middle"))
    body.append(svg_text(22, top + plot_h / 2, "normalized response", 13, anchor="middle"))
    body.append(svg_text(
        left + plot_w / 2,
        428,
        "Right-stick pitch/yaw pulses reconfigure the arm while TCP translational drift remains below the 1 mm reference.",
        12,
        COLORS["ink"],
        anchor="middle",
        weight="700",
    ))
    write_svg("right_stick_tcp_decoupling.svg", width, height, body)


def make_characterization_figures(poses: dict, settings: dict):
    rows, path = latest_characterization_rows()
    measured = bool(rows)
    if not rows:
        rows = nominal_characterization_rows(poses)
    source_label = (
        f"Measured log: {path.parent.name}/motion_characterization.csv." if measured and path else
        "Nominal trajectory synthesized from poses.json; run launcher Motion Characterization for measured data."
    )
    joint_colors = [COLORS["blue"], COLORS["green"], COLORS["red"], COLORS["violet"], COLORS["amber"], COLORS["cyan"]]
    make_line_chart(
        "characterization_joint_trajectory.svg",
        "Figure 5. Joint trajectory during characterization sweep",
        source_label,
        [(f"j{i}", joint_colors[i], points_from_rows(rows, "t", f"j{i}")) for i in range(6)],
        "joint angle (deg)",
    )
    make_line_chart(
        "characterization_joint_velocity.svg",
        "Figure 6. Joint velocity profile",
        source_label,
        [(f"dj{i}/dt", joint_colors[i], finite_diff_series(rows, f"j{i}")) for i in range(6)],
        "joint velocity (deg/s)",
    )
    make_tcp_path(rows, source_label)
    make_speed_norms_figure(rows, source_label)
    dexterity_metrics = jacobian_metric_series(rows if measured else [], settings)
    make_jacobian_metrics_figure(rows if measured else [], settings, source_label, dexterity_metrics)
    make_phase_timeline(rows, source_label)
    make_characterization_summary(rows if measured else [], path, dexterity_metrics if measured else None)
    make_target_tracking_residual(rows if measured else [], path, source_label)


def selected_pose_names(poses: dict) -> list[str]:
    order = [
        "Default",
        "Reload",
        "Injector_Pose",
        "Safe_Pose",
        "Wash1",
        "Wash2",
        "Calibration",
    ]
    return [name for name in order if name in poses]


def pose_joint_vector(poses: dict, name: str):
    try:
        return [float(poses[name][f"j{i}"]) for i in range(6)]
    except Exception:
        return None


def load_offline_dorna_model(settings: dict):
    try:
        import numpy as np
        from dorna2 import Dorna

        robot = Dorna(model="dorna_ta")
        cx = float(settings.get("tool_cx", 0.0) or 0.0)
        cy = float(settings.get("tool_cy", 0.0) or 0.0)
        lz = float(settings.get("tool_lz", 30.0) or 30.0)
        robot.kinematic.set_tcp_xyzabc([cx, cy, lz, 0.0, 0.0, 0.0])
        return robot, np
    except Exception as e:
        return None, str(e)


def offline_tcp_xyz(robot, np, q):
    T_flange = np.array(robot.kinematic.t_flange_r_world(joint=q), dtype=float)
    T_tcp = T_flange @ robot.kinematic.T_tcp_r_flange
    return np.array(T_tcp[:3, 3], dtype=float).reshape(3)


def offline_trans_jacobian(robot, np, q, delta_deg=0.05):
    q0 = np.array([float(v) for v in q[:6]], dtype=float)
    delta_rad = math.radians(float(delta_deg))
    J = np.zeros((3, 6), dtype=float)
    for idx in range(6):
        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[idx] += float(delta_deg)
        q_minus[idx] -= float(delta_deg)
        p_plus = offline_tcp_xyz(robot, np, q_plus.tolist())
        p_minus = offline_tcp_xyz(robot, np, q_minus.tolist())
        J[:, idx] = (p_plus - p_minus) / (2.0 * delta_rad)
    return J


def offline_jacobian_metric_series(rows, settings: dict):
    robot, np_or_error = load_offline_dorna_model(settings)
    if robot is None:
        return [], [], f"offline FK unavailable: {np_or_error}"
    np = np_or_error
    sigma_min = []
    manip_trans = []
    for row in rows:
        t = parse_float(row.get("t"))
        if t is None:
            continue
        q = [parse_float(row.get(f"j{i}")) for i in range(6)]
        if any(v is None for v in q):
            continue
        try:
            Jv = offline_trans_jacobian(robot, np, q)
            sigma = np.linalg.svd(Jv, compute_uv=False)
            sigma_min.append((t, float(np.min(sigma))))
            det_trans = float(np.linalg.det(Jv @ Jv.T))
            manip_trans.append((t, math.sqrt(max(0.0, det_trans))))
        except Exception:
            continue
    return sigma_min, manip_trans, "offline finite-difference FK from measured joint feedback"


def jacobian_metric_series(rows, settings: dict):
    sigma_min = points_from_rows(rows, "t", "sigma_min")
    manip = points_from_rows(rows, "t", "manip_trans") or points_from_rows(rows, "t", "manip_6d")
    metric_source = "logged Jacobian metrics"
    available = bool(sigma_min and manip)
    if not sigma_min or not manip:
        sigma_min, manip, metric_source = offline_jacobian_metric_series(rows, settings)
        available = bool(sigma_min and manip)
    return {
        "sigma_min": sigma_min,
        "manip": manip,
        "source": metric_source,
        "available": available,
    }


def make_jacobian_metrics_figure(rows, settings: dict, source_label, metrics=None):
    metrics = metrics or jacobian_metric_series(rows, settings)
    sigma_min = metrics.get("sigma_min", [])
    manip = metrics.get("manip", [])
    metric_source = metrics.get("source", "unavailable")
    sigma_vals = [v for _t, v in sigma_min]
    manip_vals = [v for _t, v in manip]
    plot_source = "logged metrics" if str(metric_source).startswith("logged") else "offline FK from measured q"
    if sigma_vals and manip_vals:
        subtitle = (
            f"{source_label} Metrics: {plot_source}; "
            f"sigma_min {min(sigma_vals):.3g}-{max(sigma_vals):.3g}, "
            f"manip {min(manip_vals):.3g}-{max(manip_vals):.3g}."
        )
    else:
        subtitle = source_label
    make_line_chart(
        "characterization_manipulability.svg",
        "Figure 10. Normalized translational dexterity along measured trajectory",
        subtitle,
        [
            ("sigma_min norm.", COLORS["red"], normalize_series(sigma_min)),
            ("manip_trans norm.", COLORS["green"], normalize_series(manip)),
        ],
        "normalized index",
        no_data_message="No Jacobian/dexterity metrics available. Rerun Motion Characterization after installing numpy.",
    )


def make_local_workspace_manipulability(poses: dict, settings: dict):
    width, height = 980, 620
    body = [
        svg_text(28, 34, "Figure 18. Local task-space reachability and translational manipulability", 21, weight="700"),
        svg_text(28, 56, "Offline finite-difference FK around Injector_Pose; color encodes translational manipulability.", 13, COLORS["muted"]),
    ]
    robot, np_or_error = load_offline_dorna_model(settings)
    q0 = pose_joint_vector(poses, "Injector_Pose")
    if robot is None or q0 is None:
        msg = f"Kinematic model unavailable: {np_or_error}" if robot is None else "Injector_Pose missing from poses.json."
        body.append(f'<rect x="70" y="96" width="840" height="330" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, 250, msg, 15, COLORS["muted"], anchor="middle"))
        write_svg("local_workspace_manipulability.svg", width, height, body)
        return
    np = np_or_error

    samples = []
    offsets = [v for v in range(-10, 11, 2)]
    for dj0 in offsets:
        for dj1 in offsets:
            q = list(q0)
            q[0] += dj0
            q[1] += dj1
            try:
                p = offline_tcp_xyz(robot, np, q)
                Jv = offline_trans_jacobian(robot, np, q)
                gram = Jv @ Jv.T
                manip = math.sqrt(max(0.0, float(np.linalg.det(gram))))
                samples.append({
                    "dj0": dj0,
                    "dj1": dj1,
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "manip": manip,
                })
            except Exception:
                continue

    if not samples:
        body.append(f'<rect x="70" y="96" width="840" height="330" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(width / 2, 250, "No valid local workspace samples could be computed.", 15, COLORS["muted"], anchor="middle"))
        write_svg("local_workspace_manipulability.svg", width, height, body)
        return

    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]
    zs = [s["z"] for s in samples]
    ms = [s["manip"] for s in samples]
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    lo_m, hi_m = min(ms), max(ms)
    if lo_x == hi_x:
        lo_x -= 1
        hi_x += 1
    if lo_y == hi_y:
        lo_y -= 1
        hi_y += 1
    pad_x = (hi_x - lo_x) * 0.08
    pad_y = (hi_y - lo_y) * 0.08
    lo_x -= pad_x
    hi_x += pad_x
    lo_y -= pad_y
    hi_y += pad_y
    plot_left, plot_top, plot_w, plot_h = 76, 96, 560, 380
    body.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    for i in range(6):
        gx = plot_left + plot_w * i / 5
        gy = plot_top + plot_h * i / 5
        body.append(f'<line x1="{gx:.1f}" y1="{plot_top}" x2="{gx:.1f}" y2="{plot_top + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{plot_left}" y1="{gy:.1f}" x2="{plot_left + plot_w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}"/>')

    def sx(x):
        return linear_map(x, lo_x, hi_x, plot_left + 22, plot_left + plot_w - 22)

    def sy(y):
        return linear_map(y, lo_y, hi_y, plot_top + plot_h - 22, plot_top + 22)

    def manip_color(m):
        t = 0.5 if hi_m == lo_m else (m - lo_m) / (hi_m - lo_m)
        red = int(214 - 130 * t)
        green = int(69 + 120 * t)
        blue = int(69 + 120 * t)
        return f"#{red:02x}{green:02x}{blue:02x}"

    for s in samples:
        body.append(f'<circle cx="{sx(s["x"]):.1f}" cy="{sy(s["y"]):.1f}" r="5" fill="{manip_color(s["manip"])}" opacity="0.82"/>')

    try:
        p0 = offline_tcp_xyz(robot, np, q0)
        body.append(f'<circle cx="{sx(float(p0[0])):.1f}" cy="{sy(float(p0[1])):.1f}" r="8" fill="none" stroke="{COLORS["ink"]}" stroke-width="2"/>')
        body.append(svg_text(sx(float(p0[0])) + 10, sy(float(p0[1])) - 8, "Injector", 11, COLORS["ink"], weight="700"))
    except Exception:
        pass

    body.append(svg_text(plot_left + plot_w / 2, plot_top + plot_h + 34, "TCP x (mm)", 13, anchor="middle"))
    body.append(svg_text(24, plot_top + plot_h / 2, "TCP y (mm)", 13, anchor="middle"))

    card_x, card_y = 690, 108
    cards = [
        ("Samples", f"{len(samples)}", "j0/j1 grid"),
        ("x span", f"{max(xs) - min(xs):.1f} mm", f"{min(xs):.1f} to {max(xs):.1f}"),
        ("y span", f"{max(ys) - min(ys):.1f} mm", f"{min(ys):.1f} to {max(ys):.1f}"),
        ("z span", f"{max(zs) - min(zs):.1f} mm", f"{min(zs):.1f} to {max(zs):.1f}"),
        ("Manipulability", f"{lo_m:.2g}-{hi_m:.2g}", "sqrt(det(Jv Jv^T))"),
    ]
    for idx, (label, value, detail) in enumerate(cards):
        y = card_y + idx * 66
        body.append(f'<rect x="{card_x}" y="{y}" width="230" height="52" rx="5" fill="#f8fafc" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(card_x + 12, y + 20, label, 12, COLORS["muted"], weight="700"))
        body.append(svg_text(card_x + 12, y + 42, value, 17, COLORS["ink"], weight="700"))
        body.append(svg_text(card_x + 118, y + 42, detail, 10, COLORS["muted"]))

    legend_x, legend_y = plot_left, 536
    for i in range(9):
        t = i / 8
        m = lo_m + (hi_m - lo_m) * t
        x = legend_x + i * 34
        body.append(f'<rect x="{x}" y="{legend_y}" width="32" height="14" fill="{manip_color(m)}"/>')
    body.append(svg_text(legend_x, legend_y + 34, "low manipulability", 11, COLORS["muted"]))
    body.append(svg_text(legend_x + 8 * 34 + 32, legend_y + 34, "high", 11, COLORS["muted"], anchor="end"))
    body.append(svg_text(width / 2, 594, "This approximates the local dexterity envelope around the injection posture; it should be replaced by measured repeatability and visual-servo data when available.", 13, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("local_workspace_manipulability.svg", width, height, body)


def make_pose_transition_matrix(poses: dict):
    names = selected_pose_names(poses)
    if not names:
        write_svg("pose_transition_matrix.svg", 860, 380, [
            svg_text(28, 34, "Figure 1. Pose-transition burden matrix", 21, weight="700"),
            svg_text(430, 200, "No named poses found.", 15, COLORS["muted"], anchor="middle"),
        ])
        return
    values = []
    max_val = 1.0
    for src in names:
        row = []
        src_vec = pose_joint_vector(poses, src)
        for dst in names:
            dst_vec = pose_joint_vector(poses, dst)
            if src_vec is None or dst_vec is None:
                value = 0.0
            else:
                value = math.sqrt(sum((dst_vec[i] - src_vec[i]) ** 2 for i in range(6)))
            row.append(value)
            max_val = max(max_val, value)
        values.append(row)

    width, height = 980, 650
    left, top = 190, 104
    cell = min(72, int((width - left - 80) / max(1, len(names))))
    body = [
        svg_text(28, 34, "Figure 1. Pose-transition burden matrix", 21, weight="700"),
        svg_text(28, 56, "Pairwise joint-space motion burden; M marks destinations with a defined midpoint gate.", 13, COLORS["muted"]),
    ]
    for c, name in enumerate(names):
        x = left + c * cell + cell / 2
        label = name.replace("_", " ")
        body.append(f'<text x="{x:.1f}" y="{top - 12:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{COLORS["ink"]}" text-anchor="end" transform="rotate(-42 {x:.1f} {top - 12:.1f})">{esc(label)}</text>')
    for r, src in enumerate(names):
        y = top + r * cell
        body.append(svg_text(left - 12, y + cell / 2 + 4, src.replace("_", " "), 12, anchor="end"))
        for c, dst in enumerate(names):
            x = left + c * cell
            value = values[r][c]
            t = value / max_val if max_val else 0.0
            red = int(247 - 105 * t)
            green = int(250 - 92 * t)
            blue = int(252 - 10 * t)
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" fill="{fill}" stroke="#ffffff"/>')
            label = f"{value:.0f}"
            body.append(svg_text(x + cell / 2 - 1, y + cell / 2 + 3, label, 11, COLORS["ink"], anchor="middle"))
            if f"{dst}__midway" in poses and src != dst:
                body.append(svg_text(x + cell - 13, y + 15, "M", 10, COLORS["blue"], anchor="middle", weight="700"))
    legend_y = top + len(names) * cell + 34
    body.append(svg_text(28, legend_y, f"Scale: 0 to {max_val:.0f} deg Euclidean norm across j0..j5. High-burden transitions justify staged motion and operator-gated final approach.", 13, COLORS["muted"]))
    write_svg("pose_transition_matrix.svg", width, max(height, legend_y + 36), body)


def make_joint_pose_heatmap(poses: dict):
    names = selected_pose_names(poses)
    joints = [f"j{i}" for i in range(6)]
    values = [[float(poses[name].get(j, 0.0)) for j in joints] for name in names]
    flat = [v for row in values for v in row]
    lo, hi = min(flat), max(flat)

    width, height = 940, 470
    left, top = 170, 70
    cell_w, cell_h = 105, 42
    body = [
        svg_text(28, 34, "Figure 1. Calibrated joint-space pose library", 21, weight="700"),
        svg_text(28, 56, f"Joint angles from poses.json; range {lo:.1f} to {hi:.1f} deg.", 13, COLORS["muted"]),
    ]
    for c, joint in enumerate(joints):
        body.append(svg_text(left + c * cell_w + cell_w / 2, top - 14, joint, 13, anchor="middle", weight="700"))
    for r, name in enumerate(names):
        y = top + r * cell_h
        body.append(svg_text(left - 12, y + 27, name.replace("_", " "), 13, anchor="end"))
        for c, v in enumerate(values[r]):
            x = left + c * cell_w
            t = linear_map(v, lo, hi, 0.0, 1.0)
            if v >= 0:
                red = int(235 - 80 * t)
                green = int(248 - 80 * t)
                blue = int(255 - 10 * t)
            else:
                red = int(255 - 60 * (1 - t))
                green = int(236 - 120 * (1 - t))
                blue = int(217 - 130 * (1 - t))
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" fill="{fill}" stroke="#ffffff"/>')
            body.append(svg_text(x + cell_w / 2 - 2, y + 25, f"{v:.1f}", 12, anchor="middle"))
    body.append(svg_text(28, 440, "Positive and negative cells encode different regions of the six-axis posture manifold used for injection, washing, reload, and calibration.", 13, COLORS["muted"]))
    write_svg("joint_pose_heatmap.svg", width, height, body)


def make_pose_transition_chart(poses: dict):
    names = selected_pose_names(poses)
    default = poses.get("Default", {})
    distances = []
    for name in names:
        vals = [float(poses[name].get(f"j{i}", 0.0)) - float(default.get(f"j{i}", 0.0)) for i in range(6)]
        distances.append((name, math.sqrt(sum(v * v for v in vals)), max(abs(v) for v in vals), f"{name}__midway" in poses))

    width, height = 980, 440
    left, right, top = 210, 40, 72
    bar_h, gap = 30, 18
    max_v = max(v for _, v, _max_axis, _mid in distances) or 1.0
    body = [
        svg_text(28, 34, "Figure 2. Staged transition burden from Default", 21, weight="700"),
        svg_text(28, 56, "Bars show joint-space reconfiguration load; labels report max single-axis move and midpoint availability.", 13, COLORS["muted"]),
    ]
    for idx, (name, value, max_axis, has_midway) in enumerate(distances):
        y = top + idx * (bar_h + gap)
        body.append(svg_text(left - 16, y + 21, name.replace("_", " "), 13, anchor="end"))
        bw = linear_map(value, 0, max_v, 0, width - left - right - 240)
        color = COLORS["blue"] if name != "Default" else COLORS["green"]
        body.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}"/>')
        gate = "midpoint" if has_midway and name != "Default" else "direct"
        body.append(svg_text(left + bw + 8, y + 21, f"{value:.1f} deg norm | max axis {max_axis:.1f} deg | {gate}", 12, COLORS["ink"]))
    axis_y = top + len(distances) * (bar_h + gap) + 12
    body.append(f'<line x1="{left}" y1="{axis_y}" x2="{width - right - 240}" y2="{axis_y}" stroke="{COLORS["grid"]}"/>')
    body.append(svg_text(left, axis_y + 22, "0", 12, COLORS["muted"], anchor="middle"))
    body.append(svg_text(width - right - 240, axis_y + 22, f"{max_v:.0f} deg", 12, COLORS["muted"], anchor="middle"))
    write_svg("pose_transition_distance.svg", width, height, body)


def make_system_architecture(settings: dict, launcher: dict):
    width, height = 980, 500
    uvc_fps = float(launcher.get("startup_uvc_fps", 30) or 30)
    rs_fps = float(launcher.get("startup_rs_fps", 30) or 30)
    rates = [
        ("Joystick sampling", 240.0, "input thread", COLORS["green"]),
        ("Robot control loop", 120.0, "state/update loop", COLORS["blue"]),
        ("Tool-axis jog stream", 80.0, "local TCP insertion", COLORS["violet"]),
        ("General TCP jog stream", 40.0, "lmove rate limit", COLORS["cyan"]),
        ("UVC microscope streams", uvc_fps, f"{int(launcher.get('startup_uvc_width', 640) or 640)}x{int(launcher.get('startup_uvc_height', 480) or 480)}", COLORS["amber"]),
        ("RealSense stream", rs_fps, f"{int(launcher.get('startup_rs_width', 1280) or 1280)}x{int(launcher.get('startup_rs_height', 720) or 720)}", COLORS["red"]),
    ]
    max_rate = max(rate for _label, rate, _detail, _color in rates)
    body = [
        svg_text(28, 34, "Figure 3. Control and acquisition timing budget", 21, weight="700"),
        svg_text(28, 56, "Thread and command rates define what the supervised controller can react to and record.", 13, COLORS["muted"]),
    ]
    left, top, bar_w = 230, 96, 560
    for idx, (label, rate, detail, color) in enumerate(rates):
        y = top + idx * 54
        width_px = linear_map(rate, 0, max_rate, 0, bar_w)
        period_ms = 1000.0 / rate if rate > 0 else 0.0
        body.append(svg_text(left - 18, y + 23, label, 13, anchor="end"))
        body.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="28" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(f'<rect x="{left}" y="{y}" width="{width_px:.1f}" height="28" fill="{color}" opacity="0.86"/>')
        body.append(svg_text(left + bar_w + 14, y + 20, f"{rate:.0f} Hz ({period_ms:.1f} ms)", 12, COLORS["ink"], weight="700"))
        body.append(svg_text(left + bar_w + 14, y + 38, detail, 11, COLORS["muted"]))
    body.append(f'<line x1="{left}" y1="{top + 348}" x2="{left + bar_w}" y2="{top + 348}" stroke="{COLORS["grid"]}"/>')
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + bar_w * frac
        val = max_rate * frac
        body.append(f'<line x1="{x:.1f}" y1="{top + 342}" x2="{x:.1f}" y2="{top + 354}" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(x, top + 374, f"{val:.0f}", 11, COLORS["muted"], anchor="middle"))
    body.append(svg_text(left + bar_w / 2, top + 398, "rate (Hz)", 13, anchor="middle"))
    body.append(svg_text(28, 462, f"Halt settings: threshold={settings.get('alarm_threshold', 30)}, duration={settings.get('alarm_duration', 50)}; command transport={launcher.get('startup_host', '10.42.0.11')}:{launcher.get('startup_port', 443)}", 13, COLORS["muted"]))
    write_svg("system_architecture.svg", width, height, body)


def make_field_positioning_matrix():
    width, height = 1120, 560
    left, top = 280, 86
    col_w, row_h = 185, 50
    columns = [
        "Manual\nmicromanipulation",
        "High-throughput\nautonomous batch",
        "This\nplatform",
        "Next\nvalidation",
    ]
    rows = [
        ("Operator-in-the-loop visual judgement", ["high", "low", "high", "high"]),
        ("Programmable pose/routine execution", ["low", "high", "high", "high"]),
        ("Reconfigurable arbitrary sample workflow", ["high", "medium", "high", "high"]),
        ("Synchronized telemetry/video evidence", ["low", "medium", "high", "high"]),
        ("Volume-calibrated fluid actuation", ["medium", "high", "high", "high"]),
        ("Closed-loop visual servoing to biological target", ["low", "high", "partial", "target"]),
        ("Force/tactile puncture confirmation", ["operator", "emerging", "none", "target"]),
    ]
    score_colors = {
        "high": COLORS["green"],
        "medium": COLORS["amber"],
        "partial": COLORS["amber"],
        "low": "#cbd5e1",
        "operator": COLORS["blue"],
        "emerging": COLORS["violet"],
        "none": COLORS["red"],
        "target": COLORS["cyan"],
    }
    score_text = {
        "high": "High",
        "medium": "Med.",
        "partial": "Partial",
        "low": "Low",
        "operator": "Human",
        "emerging": "Emerging",
        "none": "None",
        "target": "Target",
    }
    body = [
        svg_text(28, 34, "Figure 11. Field positioning and contribution boundary", 21, weight="700"),
        svg_text(28, 56, "The contribution is a supervised, reconfigurable experimental workstation rather than a fully autonomous high-throughput batch injector.", 13, COLORS["muted"]),
    ]
    for c, label in enumerate(columns):
        x = left + c * col_w
        for i, line in enumerate(label.split("\n")):
            body.append(svg_text(x + col_w / 2, top - 36 + i * 16, line, 13, anchor="middle", weight="700"))
    for r, (label, vals) in enumerate(rows):
        y = top + r * row_h
        body.append(svg_text(left - 14, y + 31, label, 13, anchor="end"))
        for c, value in enumerate(vals):
            x = left + c * col_w
            fill = score_colors[value]
            body.append(f'<rect x="{x + 9}" y="{y + 7}" width="{col_w - 18}" height="{row_h - 14}" rx="4" fill="{fill}" opacity="0.88"/>')
            text_color = "#ffffff" if value not in ("low",) else COLORS["ink"]
            body.append(svg_text(x + col_w / 2, y + 31, score_text[value], 13, text_color, anchor="middle", weight="700"))
    foot_y = top + len(rows) * row_h + 34
    body.append(svg_text(28, foot_y, "Interpretation: high-throughput systems optimize autonomy and throughput; this platform optimizes protocol development, traceability, and supervised microscope-guided operation.", 13, COLORS["muted"]))
    write_svg("field_positioning_matrix.svg", width, height, body)


def make_contribution_pipeline():
    width, height = 1080, 420
    stages = [
        ("Sensing", "Dual UVC microscopy\nRealSense context\nthermal monitor", COLORS["blue"]),
        ("State Estimation", "joint feedback\nFK TCP pose\nJacobian metrics", COLORS["violet"]),
        ("Control", "SO(3) jog control\nlocal TCP insertion\nmidpoint gating", COLORS["green"]),
        ("Protocol", "pose library\nroutine DSL\nwash/inject state machine", COLORS["amber"]),
        ("Evidence", "telemetry CSV\nmulti-camera video\nUI recording", COLORS["red"]),
    ]
    body = [
        svg_text(28, 34, "Figure 12. Contribution stack for reproducible supervised microinjection", 21, weight="700"),
        svg_text(28, 56, "The system converts manual microscope work into a logged, programmable, and characterizable robotic workflow.", 13, COLORS["muted"]),
        '<defs><marker id="arrow2" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#1f2933"/></marker></defs>',
    ]
    x0, y0, w, h, gap = 36, 118, 178, 145, 34
    for idx, (title, detail, color) in enumerate(stages):
        x = x0 + idx * (w + gap)
        body.append(f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" rx="6" fill="#f8fafc" stroke="{color}" stroke-width="2"/>')
        body.append(svg_text(x + w / 2, y0 + 30, title, 15, color, anchor="middle", weight="700"))
        for i, line in enumerate(detail.split("\n")):
            body.append(svg_text(x + w / 2, y0 + 65 + i * 20, line, 12, COLORS["ink"], anchor="middle"))
        if idx < len(stages) - 1:
            x1 = x + w + 5
            x2 = x + w + gap - 8
            y = y0 + h / 2
            body.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{COLORS["ink"]}" stroke-width="2" marker-end="url(#arrow2)"/>')
    body.append(svg_text(540, 330, "Poster thesis: practical novelty is integration, traceability, and task-frame control for low-throughput protocol development where full autonomy is premature.", 14, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("contribution_pipeline.svg", width, height, body)


def make_routine_state_machine(routine_text: str, poses: dict):
    steps = parse_routine_steps(routine_text)
    nodes = []
    omitted_unknown = 0
    idx = 0
    while idx < len(steps):
        step = steps[idx]
        parts = step.split()
        cmd = parts[0].upper() if parts else ""
        if cmd in ("POSE", "GO") and len(parts) > 1:
            raw_pose_name = " ".join(parts[1:])
            if raw_pose_name not in poses:
                omitted_unknown += 1
                idx += 1
                continue
            pose_name = raw_pose_name.replace("_", " ")
            detail = "midway gate"
            if idx + 1 < len(steps):
                nxt = steps[idx + 1].upper()
                if nxt.startswith("ADVANCE_AUTO"):
                    detail = "auto final"
                    idx += 1
                elif nxt.startswith("ADVANCE_WAIT"):
                    detail = "operator final"
                    idx += 1
            nodes.append((pose_name, detail, COLORS["blue"]))
        elif cmd == "WASH":
            repeat = parts[1] if len(parts) > 1 else "1"
            nodes.append((f"Wash x{repeat}", "BWD/FWD endstop", COLORS["amber"]))
        elif cmd == "ADVANCE_CANCEL":
            nodes.append(("Cancel gate", "direct next pose", COLORS["red"]))
        elif cmd.startswith("ADVANCE"):
            nodes.append(("Advance", cmd.lower(), COLORS["green"]))
        else:
            nodes.append((step[:18], "routine token", COLORS["violet"]))
        idx += 1

    if not nodes:
        nodes = [("No routine", "routine.txt missing", COLORS["red"])]

    width, height = 1080, 520
    body = [
        svg_text(28, 34, "Figure 14. Supervisory routine state machine", 21, weight="700"),
        svg_text(28, 56, "Pose programs combine midpoint gating, operator authority, and endstop-bounded fluid handling.", 13, COLORS["muted"]),
        '<defs><marker id="arrow3" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#1f2933"/></marker></defs>',
    ]
    x0, y0, w, h, gap = 42, 102, 150, 76, 28
    per_row = 5
    coords = []
    for idx, (title, detail, color) in enumerate(nodes[:10]):
        row = idx // per_row
        col = idx % per_row
        x = x0 + col * (w + gap)
        y = y0 + row * 160
        coords.append((x, y))
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="#f8fafc" stroke="{color}" stroke-width="2"/>')
        body.append(svg_text(x + w / 2, y + 29, title, 13, color, anchor="middle", weight="700"))
        body.append(svg_text(x + w / 2, y + 54, detail, 11, COLORS["muted"], anchor="middle"))
        if idx > 0:
            px, py = coords[idx - 1]
            if idx % per_row == 0:
                body.append(f'<path d="M {px + w / 2:.1f} {py + h + 8:.1f} C {px + w / 2:.1f} {py + 130:.1f}, {x + w / 2:.1f} {y - 36:.1f}, {x + w / 2:.1f} {y - 8:.1f}" fill="none" stroke="{COLORS["ink"]}" stroke-width="2" marker-end="url(#arrow3)"/>')
            else:
                body.append(f'<line x1="{px + w + 4:.1f}" y1="{py + h / 2:.1f}" x2="{x - 8:.1f}" y2="{y + h / 2:.1f}" stroke="{COLORS["ink"]}" stroke-width="2" marker-end="url(#arrow3)"/>')
    footer = "Novelty claim: protocol structure, not only robot motion, is recorded and repeatable across microscope-guided sessions."
    if omitted_unknown:
        footer = "Defined-pose schematic; development placeholders are excluded from this poster figure."
    body.append(svg_text(540, 454, footer, 14, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("routine_state_machine.svg", width, height, body)


def make_data_provenance_schema(settings: dict, launcher: dict):
    width, height = 1040, 500
    columns = [
        ("Robot", "j0..j5 feedback\nFK TCP pose\nalarms and phases", COLORS["blue"]),
        ("Operator", "joystick axes\nbuttons\nmanual interventions", COLORS["green"]),
        ("Vision", f"UVC x2: {launcher.get('startup_uvc_fps', 30)} fps\nRealSense: {launcher.get('startup_rs_fps', 30)} fps\nUI screen video", COLORS["violet"]),
        ("Fluidics", f"step volume: {settings.get('syringe_step_ul', 1.0)} uL\nendstop state\nwash cycles", COLORS["amber"]),
    ]
    outputs = [
        ("telemetry.csv", "time-aligned robot and operator state"),
        ("frames.csv", "camera frame timestamps"),
        ("rs/uvc/ui mp4", "visual evidence streams"),
        ("metadata", "study, subject, material, volume"),
    ]
    body = [
        svg_text(28, 34, "Figure 15. Multimodal evidence and provenance schema", 21, weight="700"),
        svg_text(28, 56, "Each injection step can be reconstructed from synchronized robot, operator, vision, and fluidic records.", 13, COLORS["muted"]),
        '<defs><marker id="arrow4" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#1f2933"/></marker></defs>',
    ]
    x0, y0, w, h, gap = 42, 100, 205, 110, 36
    center_x, center_y = width / 2, 276
    for idx, (title, detail, color) in enumerate(columns):
        x = x0 + idx * (w + gap)
        body.append(f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" rx="5" fill="#f8fafc" stroke="{color}" stroke-width="2"/>')
        body.append(svg_text(x + w / 2, y0 + 28, title, 14, color, anchor="middle", weight="700"))
        for line_idx, line in enumerate(detail.split("\n")):
            body.append(svg_text(x + w / 2, y0 + 58 + line_idx * 18, line, 11, COLORS["ink"], anchor="middle"))
        body.append(f'<line x1="{x + w / 2:.1f}" y1="{y0 + h + 6:.1f}" x2="{center_x:.1f}" y2="{center_y - 44:.1f}" stroke="{COLORS["ink"]}" stroke-width="1.8" marker-end="url(#arrow4)"/>')

    body.append(f'<rect x="{center_x - 150:.1f}" y="{center_y - 42:.1f}" width="300" height="84" rx="5" fill="#ffffff" stroke="{COLORS["ink"]}" stroke-width="2"/>')
    body.append(svg_text(center_x, center_y - 10, "Step recording directory", 15, COLORS["ink"], anchor="middle", weight="700"))
    body.append(svg_text(center_x, center_y + 18, "~/Desktop/RobotInjectionData", 12, COLORS["muted"], anchor="middle"))
    out_x0, out_y, out_w, out_h = 66, 385, 210, 58
    for idx, (title, detail) in enumerate(outputs):
        x = out_x0 + idx * (out_w + 26)
        body.append(f'<line x1="{center_x:.1f}" y1="{center_y + 46:.1f}" x2="{x + out_w / 2:.1f}" y2="{out_y - 10:.1f}" stroke="{COLORS["ink"]}" stroke-width="1.8" marker-end="url(#arrow4)"/>')
        body.append(f'<rect x="{x}" y="{out_y}" width="{out_w}" height="{out_h}" rx="5" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(x + out_w / 2, out_y + 23, title, 13, COLORS["ink"], anchor="middle", weight="700"))
        body.append(svg_text(x + out_w / 2, out_y + 43, detail, 10, COLORS["muted"], anchor="middle"))
    write_svg("data_provenance_schema.svg", width, height, body)


def make_syringe_calibration(settings: dict):
    total_ul = float(settings.get("syringe_volume_ul", 10.0) or 10.0)
    step_ul = float(settings.get("syringe_step_ul", 1.0) or 1.0)
    total_units = float(settings.get("syringe_rotations_total", 0.0) or 0.0)
    full_time = float(settings.get("syringe_full_travel_time_s", 0.0) or 0.0)
    remaining_ul = float(settings.get("syringe_remaining_ul", total_ul) or 0.0)
    step_units = total_units * step_ul / total_ul if total_ul else 0.0
    step_time = full_time * step_ul / total_ul if total_ul else 0.0
    steps_total = total_ul / step_ul if step_ul else 0.0
    steps_remaining = remaining_ul / step_ul if step_ul else 0.0

    width, height = 900, 440
    left, top = 90, 72
    plot_w, plot_h = 520, 230
    body = [
        svg_text(28, 34, "Figure 4. Syringe operating envelope", 21, weight="700"),
        svg_text(28, 56, "Calibration converts requested volume into actuator travel/time and exposes the remaining volume budget.", 13, COLORS["muted"]),
    ]
    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    for i in range(6):
        x = left + plot_w * i / 5
        y = top + plot_h * i / 5
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
    x1, y1 = left, top + plot_h
    x2, y2 = left + plot_w, top
    body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS["red"]}" stroke-width="4"/>')
    if total_ul > 0:
        rem_x = linear_map(max(0.0, min(total_ul, remaining_ul)), 0, total_ul, left, left + plot_w)
        body.append(f'<line x1="{rem_x:.1f}" y1="{top}" x2="{rem_x:.1f}" y2="{top + plot_h}" stroke="{COLORS["green"]}" stroke-width="3" stroke-dasharray="6 5"/>')
        body.append(svg_text(rem_x, top - 8, "remaining register", 11, COLORS["green"], anchor="middle", weight="700"))
    body.append(svg_text(left + plot_w / 2, top + plot_h + 42, "commanded volume (uL)", 13, COLORS["ink"], anchor="middle"))
    body.append(svg_text(18, top + plot_h / 2, "actuator units", 13, COLORS["ink"]))
    for i in range(6):
        vol = total_ul * i / 5
        units = total_units * i / 5
        x = left + plot_w * i / 5
        y = top + plot_h - plot_h * i / 5
        body.append(svg_text(x, top + plot_h + 20, f"{vol:.0f}", 11, COLORS["muted"], anchor="middle"))
        body.append(svg_text(left - 8, y + 4, f"{units:.0f}", 11, COLORS["muted"], anchor="end"))

    card_x, card_y = 650, 84
    cards = [
        ("Full stroke", f"{total_ul:.2f} uL", f"{total_units:.0f} units | {full_time:.2f}s"),
        ("Step command", f"{step_ul:.2f} uL", f"{step_units:.0f} units | {step_time:.3f}s"),
        ("Total steps", f"{steps_total:.1f}", "at configured step size"),
        ("Remaining", f"{remaining_ul:.2f} uL", f"{steps_remaining:.1f} steps"),
    ]
    for idx, (label, value, detail) in enumerate(cards):
        y = card_y + idx * 72
        color = COLORS["green"] if label != "Remaining" or remaining_ul > 0 else COLORS["red"]
        body.append(f'<rect x="{card_x}" y="{y}" width="210" height="56" rx="5" fill="#f8fafc" stroke="{color}" stroke-width="2"/>')
        body.append(svg_text(card_x + 12, y + 21, label, 12, COLORS["muted"], weight="700"))
        body.append(svg_text(card_x + 12, y + 43, value, 18, COLORS["ink"], weight="700"))
        body.append(svg_text(card_x + 112, y + 43, detail, 10, COLORS["muted"]))
    body.append(svg_text(450, 396, "More informative than a calibration line alone: this reports resolution, full-stroke timing, and whether the syringe state is ready for another experiment.", 13, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("syringe_calibration.svg", width, height, body)


def make_xbox_arduino_control_stack(settings: dict):
    """Technical schematic of the Xbox-controller-to-Arduino syringe path."""
    fwd_rate = int(float(settings.get("plunger_fwd_rate", 1600) or 1600))
    bwd_rate = int(float(settings.get("plunger_bwd_rate", 1600) or 1600))
    dir_sign = int(float(settings.get("plunger_dir_sign", 1) or 1))
    max_rate = 1600
    total_ul = float(settings.get("syringe_volume_ul", 10.0) or 10.0)
    step_ul = float(settings.get("syringe_step_ul", 1.0) or 1.0)
    total_units = float(settings.get("syringe_rotations_total", 0.0) or 0.0)
    full_time = float(settings.get("syringe_full_travel_time_s", 0.0) or 0.0)
    step_units = total_units * step_ul / total_ul if total_ul > 0 else 0.0
    step_time = full_time * step_ul / total_ul if total_ul > 0 else 0.0
    timeout_ms = int(float(settings.get("endstop_timeout_ms", 0) or 0))
    expel_endstop = str(settings.get("expel_endstop", "BWD") or "BWD").upper()

    width, height = 1080, 650
    body = [
        svg_text(28, 34, "Figure 20. Xbox-to-Arduino syringe control stack", 21, weight="700"),
        svg_text(28, 56, "Controller triggers are converted into guarded serial velocity commands and calibrated volume accounting.", 13, COLORS["muted"]),
        '<defs><marker id="arrow5" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#1f2933"/></marker></defs>',
    ]

    def text_block(x, y, lines, size=11, color=None, anchor="start", line_h=16, weight="400"):
        for idx, line in enumerate(lines):
            body.append(svg_text(x, y + idx * line_h, line, size, color or COLORS["ink"], anchor=anchor, weight=weight))

    def card(x, y, w, h, title, lines, stroke, fill="#f8fafc"):
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        body.append(svg_text(x + 12, y + 22, title, 13, stroke, weight="700"))
        text_block(x + 12, y + 46, lines, 10.5, COLORS["ink"], line_h=15)

    def arrow(x1, y1, x2, y2, color=None, width_px=1.8):
        body.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color or COLORS["ink"]}" stroke-width="{width_px}" marker-end="url(#arrow5)"/>'
        )

    # Top-level signal path.
    boxes = [
        (
            34,
            "Xbox controller",
            [
                "pygame joystick 0",
                "LT axis 2, RT axis 5",
                "A/B/X/Y = 0/1/2/3",
                "poll thread: 240 Hz",
            ],
            COLORS["green"],
        ),
        (
            246,
            "Input conditioning",
            [
                "deadzone = 0.02",
                "trigger normalize u in [0,1]",
                "A/B/X/Y edge detection",
                "axis guard delta = 0.20",
            ],
            COLORS["blue"],
        ),
        (
            458,
            "Supervisory gates",
            [
                "block while at __midway",
                "block before pose settles",
                "block during calibration",
                "branch: manual vs injection",
            ],
            COLORS["violet"],
        ),
        (
            670,
            "Command synthesis",
            [
                "manual: LT -> +V",
                "manual: RT -> -V",
                "injection: RT drives dose",
                "release/complete -> V0",
            ],
            COLORS["amber"],
        ),
        (
            882,
            "Arduino bridge",
            [
                "pyserial 115200 baud",
                "ASCII: V<rate>",
                "E queries endstops",
                "R resets controller",
            ],
            COLORS["red"],
        ),
    ]
    y_top, box_w, box_h = 88, 170, 118
    for x, title, lines, color in boxes:
        card(x, y_top, box_w, box_h, title, lines, color)
    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][0] + box_w
        x2 = boxes[idx + 1][0]
        arrow(x1 + 8, y_top + box_h / 2, x2 - 10, y_top + box_h / 2)

    # Trigger-to-rate transfer function.
    plot_x, plot_y, plot_w, plot_h = 72, 265, 410, 230
    rate_hi = max(max_rate, abs(fwd_rate), abs(bwd_rate), 1)
    body.append(svg_text(plot_x, plot_y - 22, "Proportional velocity map", 14, COLORS["ink"], weight="700"))
    body.append(f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    for i in range(6):
        gx = plot_x + plot_w * i / 5
        gy = plot_y + plot_h * i / 5
        body.append(f'<line x1="{gx:.1f}" y1="{plot_y}" x2="{gx:.1f}" y2="{plot_y + plot_h}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<line x1="{plot_x}" y1="{gy:.1f}" x2="{plot_x + plot_w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(gx, plot_y + plot_h + 18, f"{i/5:.1f}", 10, COLORS["muted"], anchor="middle"))
        rate_label = rate_hi - 2 * rate_hi * i / 5
        body.append(svg_text(plot_x - 8, gy + 4, f"{rate_label:.0f}", 10, COLORS["muted"], anchor="end"))
    zero_y = linear_map(0, -rate_hi, rate_hi, plot_y + plot_h, plot_y)
    body.append(f'<line x1="{plot_x}" y1="{zero_y:.1f}" x2="{plot_x + plot_w}" y2="{zero_y:.1f}" stroke="{COLORS["ink"]}" stroke-width="1.4"/>')

    def px(u):
        return linear_map(u, 0, 1, plot_x, plot_x + plot_w)

    def py(rate):
        return linear_map(rate, -rate_hi, rate_hi, plot_y + plot_h, plot_y)

    lt_rate = max_rate * dir_sign
    rt_rate = -max_rate * dir_sign
    inj_rate = -abs(fwd_rate) * dir_sign
    body.append(f'<polyline points="{px(0):.1f},{py(0):.1f} {px(1):.1f},{py(lt_rate):.1f}" fill="none" stroke="{COLORS["green"]}" stroke-width="3"/>')
    body.append(f'<polyline points="{px(0):.1f},{py(0):.1f} {px(1):.1f},{py(rt_rate):.1f}" fill="none" stroke="{COLORS["red"]}" stroke-width="3"/>')
    body.append(f'<polyline points="{px(0):.1f},{py(0):.1f} {px(1):.1f},{py(inj_rate):.1f}" fill="none" stroke="{COLORS["violet"]}" stroke-width="2.4" stroke-dasharray="6 5"/>')
    body.append(f'<line x1="{px(0.02):.1f}" y1="{plot_y}" x2="{px(0.02):.1f}" y2="{plot_y + plot_h}" stroke="{COLORS["amber"]}" stroke-width="2" stroke-dasharray="5 4"/>')
    body.append(svg_text(px(0.02) + 4, plot_y + 14, "deadzone", 10, COLORS["amber"], weight="700"))
    body.append(svg_text(plot_x + plot_w / 2, plot_y + plot_h + 40, "trigger command u", 12, COLORS["ink"], anchor="middle"))
    body.append(svg_text(plot_x - 42, plot_y + plot_h / 2, "serial rate", 12, COLORS["ink"], anchor="middle"))
    body.append(svg_text(plot_x + 18, plot_y + 20, "LT manual reverse/prime", 10, COLORS["green"], weight="700"))
    body.append(svg_text(plot_x + 18, plot_y + 38, "RT manual/inject expel", 10, COLORS["red"], weight="700"))
    body.append(svg_text(plot_x + 18, plot_y + 56, "injection uses configured FWD rate", 10, COLORS["violet"], weight="700"))

    # Calibrated injection branch.
    panel_x, panel_y, panel_w, panel_h = 530, 252, 500, 255
    body.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="5" fill="#ffffff" stroke="{COLORS["grid"]}"/>')
    body.append(svg_text(panel_x + 18, panel_y + 30, "Calibrated dose integrator", 14, COLORS["ink"], weight="700"))
    formulas = [
        f"s_step = s_full * Q_step / Q_full = {step_units:.1f} units",
        "s_k = s_(k-1) + |V_cmd| * dt",
        "stop when s_k >= s_step, then write V0",
        f"current step = {step_ul:.3g} uL, nominal time = {step_time:.3f} s",
    ]
    text_block(panel_x + 18, panel_y + 58, formulas, 12, COLORS["ink"], line_h=22)

    bar_x, bar_y, bar_w, bar_h = panel_x + 28, panel_y + 164, 300, 28
    step_frac = max(0.0, min(1.0, step_units / total_units)) if total_units > 0 else 0.0
    body.append(svg_text(bar_x, bar_y - 12, "full calibrated syringe stroke", 11, COLORS["muted"], weight="700"))
    body.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="#fbfdff" stroke="{COLORS["grid"]}"/>')
    body.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w * step_frac:.1f}" height="{bar_h}" fill="{COLORS["blue"]}" opacity="0.9"/>')
    body.append(svg_text(bar_x + bar_w * step_frac + 8, bar_y + 20, f"{step_frac*100:.1f}% per step", 11, COLORS["blue"], weight="700"))
    body.append(svg_text(bar_x, bar_y + 50, f"0 units", 10, COLORS["muted"]))
    body.append(svg_text(bar_x + bar_w, bar_y + 50, f"{total_units:.0f} units", 10, COLORS["muted"], anchor="end"))

    metric_cards = [
        ("Manual max", f"+/-{max_rate} units/s"),
        ("FWD/BWD cfg", f"{fwd_rate}/{bwd_rate} units/s"),
        ("Endstop timeout", f"{timeout_ms} ms" if timeout_ms else "operator-monitored"),
        ("Final expel", expel_endstop),
    ]
    for idx, (label, value) in enumerate(metric_cards):
        x = panel_x + 350
        y = panel_y + 58 + idx * 43
        body.append(f'<rect x="{x}" y="{y}" width="128" height="34" rx="5" fill="#f8fafc" stroke="{COLORS["grid"]}"/>')
        body.append(svg_text(x + 8, y + 14, label, 9, COLORS["muted"], weight="700"))
        body.append(svg_text(x + 8, y + 29, value, 11, COLORS["ink"], weight="700"))

    # Serial protocol and endstop feedback layer.
    lane_y = 555
    body.append(svg_text(70, lane_y - 30, "Serial protocol and feedback parser", 14, COLORS["ink"], weight="700"))
    host_x, ard_x, out_x = 70, 430, 790
    card(host_x, lane_y - 8, 230, 72, "Python host", ["writes V<rate>, V0, E, R", "keeps last_rate to avoid repeats"], COLORS["blue"])
    card(ard_x, lane_y - 8, 230, 72, "Arduino firmware", ["velocity command execution", "endstop status returned as text"], COLORS["amber"])
    card(out_x, lane_y - 8, 230, 72, "Parser / safety stop", ["ACT L/R, EL/ER, keywords", "routine/calibration end at V0"], COLORS["green"])
    arrow(host_x + 238, lane_y + 20, ard_x - 10, lane_y + 20)
    arrow(ard_x + 238, lane_y + 20, out_x - 10, lane_y + 20)
    body.append(svg_text(342, lane_y + 8, "ASCII @ 115200", 10, COLORS["muted"], anchor="middle"))
    body.append(svg_text(700, lane_y + 8, "E responses", 10, COLORS["muted"], anchor="middle"))
    body.append(svg_text(width / 2, 636, "The Arduino is not given a volume command directly; the host closes the calibrated dose step by integrating commanded actuator travel and issuing V0.", 12, COLORS["ink"], anchor="middle", weight="700"))
    write_svg("xbox_arduino_control_stack.svg", width, height, body)


def main():
    poses = load_json(POSES_PATH)
    settings = load_json(SETTINGS_PATH)
    launcher = load_json(LAUNCHER_PATH)
    routine_text = load_routine_text()
    repeat_rows, repeat_path = latest_repeatability_rows()
    right_rows, right_path = latest_right_stick_demo_rows()
    make_pose_transition_matrix(poses)
    make_local_workspace_manipulability(poses, settings)
    make_pose_transition_chart(poses)
    make_system_architecture(settings, launcher)
    make_field_positioning_matrix()
    make_contribution_pipeline()
    make_routine_state_machine(routine_text, poses)
    make_data_provenance_schema(settings, launcher)
    make_syringe_calibration(settings)
    make_xbox_arduino_control_stack(settings)
    make_characterization_figures(poses, settings)
    make_endpoint_repeatability(repeat_rows, repeat_path)
    make_right_stick_decoupling_demo(right_rows, right_path)
    print(f"Wrote SVG figures to {OUT}")


if __name__ == "__main__":
    main()
