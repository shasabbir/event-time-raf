from __future__ import annotations

import csv
import html
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "tmp" / "run_20260821" / "tables" / "metrics.csv"
FIGURE_DIR = ROOT / "paper" / "figures"
WIDTH, HEIGHT = 720, 500
LEFT, RIGHT, TOP, BOTTOM = 74, 22, 24, 118

MODELS = {
    "M00_persistence": "Persistence",
    "C01_ridge_context": "Ridge",
    "C05_lightgbm_context": "LightGBM",
    "C06_context_ensemble": "Context ensemble",
    "M09_event_timeraf_full": "Event-feature XGBoost",
    "M13_trace_raf": "TRACE-RAF",
    "M10_frozen_chronos": "Chronos-Bolt",
}

STYLES = [
    ("#6B7280", "7 5"),
    ("#111827", ""),
    ("#2F6B3C", ""),
    ("#4E79A7", ""),
    ("#B55C7A", "8 4 2 4"),
    ("#B8860B", ""),
    ("#7A5195", "2 4"),
]


def load_metric(metric: str) -> dict[str, list[tuple[int, float]]]:
    values = {model: [] for model in MODELS}
    with METRICS_PATH.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["subset"] != "all" or row["model"] not in values:
                continue
            if not row["horizon"].isdigit():
                continue
            values[row["model"]].append((int(row["horizon"]), float(row[metric])))
    for points in values.values():
        points.sort()
    return values


def line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{html.escape(value)}"' for key, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attributes}/>'


def text(x: float, y: float, value: str, anchor: str = "middle", css: str = "tick", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{css}"{transform}>{html.escape(value)}</text>'


def build(metric: str, ylabel: str, stem: str) -> None:
    series = load_metric(metric)
    all_values = [value for points in series.values() for _, value in points]
    if metric == "mse":
        ymax = math.ceil(max(all_values) / 10.0) * 10.0
        tick_values = list(range(0, int(ymax) + 1, 10))
    else:
        ymax = math.ceil(max(all_values) * 2.0) / 2.0 + 0.5
        tick_values = list(range(0, int(math.floor(ymax)) + 1))
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM

    def x_pos(horizon: int) -> float:
        return LEFT + (horizon - 1) / 23 * plot_width

    def y_pos(value: float) -> float:
        return TOP + (ymax - value) / ymax * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        ".tick{font-family:'Times New Roman',serif;font-size:15px;fill:#263238}",
        ".axis{font-family:'Times New Roman',serif;font-size:17px;fill:#111827}",
        ".legend{font-family:'Times New Roman',serif;font-size:15px;fill:#263238}",
        "</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>',
    ]
    for value in tick_values:
        y = y_pos(value)
        parts.append(line(LEFT, y, WIDTH - RIGHT, y, stroke="#D1D5DB", stroke_width="1"))
        parts.append(text(LEFT - 10, y + 5, f"{value:.0f}" if metric == "mse" else f"{value:.1f}", anchor="end"))
    for horizon in (1, 4, 8, 12, 16, 20, 24):
        x = x_pos(horizon)
        parts.append(line(x, TOP, x, HEIGHT - BOTTOM, stroke="#E5E7EB", stroke_width="1"))
        parts.append(text(x, HEIGHT - BOTTOM + 24, str(horizon)))
    parts.append(line(LEFT, TOP, LEFT, HEIGHT - BOTTOM, stroke="#263238", stroke_width="1.5"))
    parts.append(line(LEFT, HEIGHT - BOTTOM, WIDTH - RIGHT, HEIGHT - BOTTOM, stroke="#263238", stroke_width="1.5"))

    for (model, label), (color, dash) in zip(MODELS.items(), STYLES):
        points = series[model]
        coordinates = " ".join(f"{x_pos(h):.1f},{y_pos(v):.1f}" for h, v in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )
        for horizon, value in points[::3]:
            parts.append(f'<circle cx="{x_pos(horizon):.1f}" cy="{y_pos(value):.1f}" r="3.0" fill="{color}"/>')

    parts.append(text((LEFT + WIDTH - RIGHT) / 2, HEIGHT - 78, "Forecast horizon (hours)", css="axis"))
    parts.append(text(20, (TOP + HEIGHT - BOTTOM) / 2, ylabel, css="axis", rotate=-90))
    for index, ((_, label), (color, dash)) in enumerate(zip(MODELS.items(), STYLES)):
        column, row = index % 4, index // 4
        x = 34 + column * 171
        y = HEIGHT - 48 + row * 25
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 35}" y2="{y}" stroke="{color}" stroke-width="2.4"{dash_attr}/>' )
        parts.append(text(x + 43, y + 5, label, anchor="start", css="legend"))
    parts.append("</svg>")

    svg_path = FIGURE_DIR / f"{stem}.svg"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    browser = next(
        (path for path in (
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        ) if path.exists()),
        None,
    )
    if browser is None:
        raise RuntimeError("Chrome or Edge is required for vector PDF conversion")
    print_path = svg_path.with_suffix(".print.html")
    print_path.write_text(
        "<!doctype html><style>@page{size:7.2in 5in;margin:0}body{margin:0}</style>"
        + svg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    subprocess.run(
        [str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", print_path.resolve().as_uri()],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print_path.unlink()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    build("mse", "MSE", "mse_by_horizon")
    build("mae", "MAE", "mae_by_horizon")


if __name__ == "__main__":
    main()
