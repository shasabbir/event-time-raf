from __future__ import annotations

import html
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
WIDTH = 1200
HEIGHT = 640

INK = "#263238"
MUTED = "#5F6B73"
BLUE = "#DDECF8"
BLUE_DARK = "#4E79A7"
GOLD = "#F7E8B2"
GOLD_DARK = "#B8860B"
PINK = "#F6DDE5"
PINK_DARK = "#B55C7A"
GREEN = "#DDEEDC"
GREEN_DARK = "#4E7D57"
GRAY = "#EEF1F3"
WHITE = "#FFFFFF"


class Svg:
    def __init__(self, title: str):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(title)}">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>',
            '<style><![CDATA[',
            ".label{font-family:Arial,Helvetica,sans-serif;font-size:18px;fill:#263238}",
            ".small{font-family:Arial,Helvetica,sans-serif;font-size:15px;fill:#5F6B73}",
            ".mono{font-family:Consolas,'Courier New',monospace;font-size:14px;fill:#263238}",
            ".group{font-family:Arial,Helvetica,sans-serif;font-size:17px;font-weight:700;fill:#263238}",
            ".title{font-family:Arial,Helvetica,sans-serif;font-size:21px;font-weight:700;fill:#263238}",
            "]]></style>",
            "</defs>",
            f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{WHITE}"/>',
        ]

    def rect(self, x, y, w, h, fill, title, subtitle=None, stroke=INK, radius=10, dashed=False):
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.7"{dash}/>'
        )
        lines = title if isinstance(title, list) else [title]
        line_height = 22
        title_y = y + h / 2 - ((len(lines) - 1) * line_height) / 2 - (10 if subtitle else 0)
        for index, line in enumerate(lines):
            self.text(x + w / 2, title_y + index * line_height, line, "label", anchor="middle")
        if subtitle:
            self.text(x + w / 2, y + h - 14, subtitle, "mono", anchor="middle")

    def group(self, x, y, w, h, title, stroke=MUTED):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" '
            f'stroke="{stroke}" stroke-width="1.7" stroke-dasharray="8 6"/>'
        )
        self.parts.append(f'<rect x="{x + 14}" y="{y - 12}" width="{max(150, len(title) * 10)}" height="24" fill="{WHITE}"/>')
        self.text(x + 22, y + 5, title, "group")

    def text(self, x, y, value, css="small", anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}" '
            f'dominant-baseline="middle">{html.escape(str(value))}</text>'
        )

    def arrow(self, points, label=None, label_xy=None, dashed=False, color=INK):
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.1" '
            f'stroke-linejoin="round" stroke-linecap="round" marker-end="url(#arrow)"{dash}/>'
        )
        if label and label_xy:
            x, y = label_xy
            width = max(48, len(label) * 8.5)
            self.parts.append(f'<rect x="{x - width / 2}" y="{y - 12}" width="{width}" height="24" fill="{WHITE}" opacity="0.94"/>')
            self.text(x, y, label, "mono", anchor="middle")

    def line(self, x1, y1, x2, y2, dashed=False, color=MUTED):
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.7"{dash}/>'
        )

    def circle(self, cx, cy, r, fill=WHITE, stroke=INK, text_value=None):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
        if text_value:
            self.text(cx, cy, text_value, "label", anchor="middle")

    def finish(self, path: Path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def pipeline_figure(path: Path) -> None:
    s = Svg("TRACE-RAF end-to-end pipeline")
    s.group(24, 52, 240, 520, "Official source records")
    s.rect(52, 100, 184, 76, BLUE, ["EPA AirData", "PM2.5 records"], "T x 1")
    s.rect(52, 220, 184, 76, BLUE, ["NOAA GHCNh", "weather"], "T x 6")
    s.rect(52, 340, 184, 76, BLUE, ["NOAA Storm Events", "verified cache"], "127 event days")
    s.rect(52, 460, 184, 70, GRAY, ["Calendar rules"], "T x 9")

    s.group(292, 52, 250, 520, "SACB: context construction", stroke=BLUE_DARK)
    s.rect(320, 118, 194, 78, BLUE, ["Audit and align", "hourly records"], "UTC + local time")
    s.rect(320, 254, 194, 88, BLUE, ["Causal feature", "construction"], "q_t in R^85")
    s.rect(320, 404, 194, 92, BLUE, ["Chronological", "windowing"], "X: Bx168; Y: Bx24")
    s.arrow([(236, 138), (280, 138), (280, 157), (320, 157)])
    s.arrow([(236, 258), (280, 258), (280, 157), (320, 157)])
    s.arrow([(236, 378), (280, 378), (280, 293), (320, 293)])
    s.arrow([(236, 495), (280, 495), (280, 293), (320, 293)])
    s.arrow([(417, 196), (417, 254)])
    s.arrow([(417, 342), (417, 404)])

    s.group(570, 52, 270, 520, "LSER: historical retrieval", stroke=GOLD_DARK)
    s.rect(600, 112, 210, 82, GOLD, ["Training-history", "knowledge base"], "N_K=1,824")
    s.rect(600, 254, 210, 96, GOLD, ["Embargo filter +", "hybrid ranking"], "k=8; alpha,beta,gamma,delta")
    s.rect(600, 414, 210, 82, GOLD, ["Retrieved futures", "and summaries"], "Bx8x24 -> Bx51")
    s.arrow([(514, 450), (564, 450), (564, 153), (600, 153)], "train only", (563, 224))
    s.arrow([(514, 293), (600, 293)], "query", (557, 278))
    s.arrow([(705, 194), (705, 254)])
    s.arrow([(705, 350), (705, 414)])

    s.group(868, 52, 308, 520, "TRACE + DFEH", stroke=PINK_DARK)
    s.rect(898, 92, 248, 76, PINK, ["Context base ensemble"], "B x 55 -> B x 24")
    s.rect(898, 210, 248, 76, GOLD, ["Residual analogue", "alignment"], "B x 8 x 24")
    s.rect(898, 328, 248, 76, GREEN, ["Reliability gate", "and correction"], "z: B x 18; g: B x 1")
    s.rect(898, 462, 248, 76, GREEN, ["Forecast + evidence"], "Yhat_TRACE: B x 24")
    s.arrow([(810, 440), (850, 440), (850, 248), (898, 248)], "residual IDs", (850, 309))
    s.arrow([(514, 293), (560, 293), (560, 592), (884, 592), (884, 130), (898, 130)], "q_B, C+", (724, 592))
    s.arrow([(1022, 168), (1022, 210)])
    s.arrow([(1022, 286), (1022, 328)])
    s.arrow([(1022, 404), (1022, 462)])
    s.text(600, 618, "Every candidate target precedes its query; event inputs obey the declared retrospective availability assumption.", "small", anchor="middle")
    s.finish(path)


def context_figure(path: Path) -> None:
    s = Svg("Source-Audited Context Builder internals")
    s.group(24, 48, 260, 538, "Source audit")
    s.rect(52, 96, 204, 72, BLUE, ["EPA PM2.5", "county monitors"], "coverage=0.9997")
    s.rect(52, 208, 204, 72, BLUE, ["NOAA GHCNh", "station 72297023129"], "coverage=0.9923")
    s.rect(52, 320, 204, 72, BLUE, ["NOAA events", "hash + manifest"], "127 event days")
    s.rect(52, 432, 204, 88, GRAY, ["Availability and", "duplicate checks"], "event-start caveat")
    s.arrow([(154, 168), (154, 208)])
    s.arrow([(154, 280), (154, 320)])
    s.arrow([(154, 392), (154, 432)])

    s.group(316, 48, 430, 538, "Hourly context construction", stroke=BLUE_DARK)
    s.rect(346, 92, 170, 74, BLUE, ["PM2.5 lags, rolling", "and differences"], "23 features")
    s.rect(546, 92, 170, 74, BLUE, ["Weather values,", "flags, rolling means"], "14 features")
    s.rect(346, 222, 170, 74, GRAY, ["Cyclic calendar", "and holidays"], "9 features")
    s.rect(546, 222, 170, 74, GOLD, ["Event counts, active", "states, burst ratio"], "39 features")
    s.circle(531, 390, 30, GREEN, text_value="||")
    s.arrow([(431, 166), (431, 190), (330, 190), (330, 390), (501, 390)])
    s.arrow([(631, 166), (631, 190), (732, 190), (732, 390), (561, 390)])
    s.arrow([(431, 296), (431, 350), (501, 350), (501, 390)])
    s.arrow([(631, 296), (631, 350), (561, 350), (561, 390)])
    s.rect(406, 444, 250, 76, GREEN, ["Origin context vector"], "q_t in R^85")
    s.arrow([(531, 420), (531, 444)])
    s.text(531, 548, "Fitted and checked on chronological training history", "small", anchor="middle")

    s.group(778, 48, 398, 538, "Window and split contract", stroke=GREEN_DARK)
    s.rect(812, 94, 330, 76, GREEN, ["Aligned hourly panel"], "T x (1 + 85)")
    s.rect(812, 226, 146, 82, GREEN, ["Lookback"], "X_t: 168 x 1")
    s.rect(996, 226, 146, 82, PINK, ["Future target"], "Y_t: 24 x 1")
    s.line(977, 170, 977, 206, color=INK)
    s.line(885, 206, 1069, 206, color=INK)
    s.arrow([(885, 206), (885, 226)])
    s.arrow([(1069, 206), (1069, 226)])
    s.rect(812, 372, 330, 90, GRAY, ["Chronological partitions"], "45,784 train | 7,950 val | 2,460 test")
    s.line(885, 308, 885, 342, color=INK)
    s.line(1069, 308, 1069, 342, color=INK)
    s.line(885, 342, 1069, 342, color=INK)
    s.arrow([(977, 342), (977, 372)])
    s.text(977, 510, "56,194 valid input-target windows", "mono", anchor="middle")
    s.text(977, 542, "targets are never interpolated", "small", anchor="middle")
    s.finish(path)


def retrieval_figure(path: Path) -> None:
    s = Svg("Leakage-Safe Event-Context Retriever internals")
    s.group(24, 48, 344, 538, "Temporal eligibility")
    s.text(48, 92, "past", "small")
    s.text(340, 92, "query time", "small", anchor="end")
    s.line(52, 116, 338, 116, color=INK)
    s.arrow([(52, 116), (338, 116)])
    s.rect(58, 154, 118, 60, GRAY, ["candidate", "input"], "168 h", radius=5)
    s.rect(176, 154, 68, 60, PINK, ["target"], "24 h", radius=5)
    s.rect(274, 154, 68, 60, BLUE, ["query"], "168 h", radius=5)
    s.line(244, 132, 244, 232, dashed=True)
    s.line(274, 132, 274, 232, dashed=True)
    s.text(259, 250, "strict gap", "mono", anchor="middle")
    s.rect(58, 316, 284, 94, GOLD, ["Eligibility set A_t"], "candidate_end < query_start")
    s.arrow([(200, 214), (200, 316)])
    s.rect(58, 466, 284, 72, GRAY, ["Training-history restriction"], "validation/test queries")
    s.arrow([(200, 410), (200, 466)])

    s.group(398, 48, 470, 538, "Hybrid similarity and top-k", stroke=GOLD_DARK)
    labels = [
        (430, 98, "PM2.5 shape", "s_ts", BLUE),
        (650, 98, "Weather context", "s_w", BLUE),
        (430, 218, "Calendar context", "s_c", GRAY),
        (650, 218, "Event context", "s_e", GOLD),
    ]
    for x, y, title, sub, fill in labels:
        s.rect(x, y, 186, 72, fill, [title], sub)
    s.circle(633, 350, 34, GREEN, text_value="sum")
    # Top-row channels use the side gutters instead of crossing the lower blocks.
    s.arrow([(523, 170), (523, 190), (414, 190), (414, 350), (599, 350)])
    s.arrow([(743, 170), (743, 190), (852, 190), (852, 350), (667, 350)])
    s.arrow([(523, 290), (523, 318), (609, 318), (609, 326)])
    s.arrow([(743, 290), (743, 318), (657, 318), (657, 326)])
    s.text(633, 402, "0.5 s_ts + 0.2 s_w + 0.1 s_c + 0.2 s_e", "mono", anchor="middle")
    s.rect(474, 454, 318, 76, GREEN, ["TopK over eligible candidates"], "G_t: B x 8")
    s.arrow([(633, 384), (633, 454)])

    s.group(898, 48, 278, 538, "Retrieved evidence", stroke=GREEN_DARK)
    s.rect(928, 104, 218, 82, GREEN, ["Aligned candidate", "future trajectories"], "B x 8 x 24")
    s.rect(928, 252, 218, 94, GREEN, ["Uniform / score-weighted", "forecast"], "Yhat_R: B x 24")
    s.rect(928, 412, 218, 98, GRAY, ["Trajectory, spread,", "similarity, count"], "rho(G_t): B x 51")
    s.arrow([(792, 492), (884, 492), (884, 145), (928, 145)], "candidate IDs", (884, 380))
    s.arrow([(1037, 186), (1037, 252)])
    s.arrow([(1037, 346), (1037, 412)])
    s.finish(path)


def forecast_figure(path: Path) -> None:
    s = Svg("TRACE residual correction and evidence head internals")
    s.group(24, 48, 278, 538, "Conditioning tensors")
    s.rect(52, 92, 222, 68, BLUE, ["Base origin context"], "q_B: B x 46")
    s.rect(52, 212, 222, 68, BLUE, ["Future calendar"], "C+: B x 24 x 9")
    s.rect(52, 332, 222, 68, GOLD, ["Residual neighbours"], "Delta_G: B x 8 x 24")
    s.rect(52, 452, 222, 68, PINK, ["Reliability inputs"], "z_t: B x 18")

    s.group(330, 48, 540, 538, "Trust-gated residual correction", stroke=PINK_DARK)
    s.rect(372, 92, 206, 80, PINK, ["Context XGBoost", "24 horizon heads"], "Yhat_X: B x 24")
    s.rect(622, 92, 206, 80, PINK, ["Context LightGBM", "24 horizon heads"], "Yhat_L: B x 24")
    s.circle(600, 250, 34, GREEN, text_value="base")
    s.text(600, 300, "0.3479 Yhat_X + 0.6521 Yhat_L", "mono", anchor="middle")
    s.rect(372, 354, 206, 76, GOLD, ["Robust residual", "alignment"], "r_t: B x 24")
    s.rect(622, 354, 206, 76, GREEN, ["Depth-2 trust gate"], "g_t: B x 1")
    s.circle(600, 504, 34, GREEN, text_value="+")
    s.text(600, 554, "Yhat_TRACE = Yhat_B + g_t r_t", "mono", anchor="middle")
    s.arrow([(274, 126), (338, 126), (338, 132), (372, 132)], "q_B", (338, 94))
    s.arrow([(274, 246), (350, 246), (350, 188), (846, 188), (846, 132), (828, 132)], "C+", (350, 214))
    s.arrow([(350, 188), (350, 152), (372, 152)])
    s.arrow([(338, 126), (338, 76), (846, 76), (846, 112), (828, 112)])
    s.arrow([(475, 172), (475, 224), (566, 224), (566, 238)])
    s.arrow([(725, 172), (725, 224), (634, 224), (634, 238)])
    s.arrow([(274, 366), (372, 366)])
    s.arrow([(274, 486), (606, 486), (606, 392), (622, 392)])
    s.arrow([(600, 284), (600, 470)])
    s.arrow([(475, 430), (475, 456), (575, 456), (575, 480)])
    s.arrow([(725, 430), (725, 456), (625, 456), (625, 480)])

    s.group(900, 48, 276, 538, "DFEH evidence output", stroke=GREEN_DARK)
    s.rect(930, 92, 216, 66, GRAY, ["Base feature effects"], "top fields")
    s.rect(930, 198, 216, 66, GRAY, ["Residual window IDs"], "top 8 analogues")
    s.rect(930, 304, 216, 66, GOLD, ["Linked event IDs"], "prior 72 h")
    s.rect(930, 410, 216, 66, PINK, ["Gate, drift, uncertainty"], "machine-readable")
    s.arrow([(634, 504), (884, 504), (884, 125), (930, 125)], "forecast", (884, 544))
    s.line(910, 125, 910, 443, color=INK)
    s.arrow([(910, 231), (930, 231)])
    s.arrow([(910, 337), (930, 337)])
    s.arrow([(910, 443), (930, 443)])
    s.text(1038, 532, "2,460 evidence records", "mono", anchor="middle")
    s.finish(path)


def convert_to_pdf(svg_path: Path) -> None:
    chrome_candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    browser = next((candidate for candidate in chrome_candidates if candidate.exists()), None)
    if browser is None:
        raise RuntimeError("Chrome or Edge is required to convert SVG figures to vector PDF")
    html_path = svg_path.with_suffix(".print.html")
    pdf_path = svg_path.with_suffix(".pdf")
    html_path.write_text(
        "<!doctype html><html><head><style>"
        "@page{size:12in 6.4in;margin:0}html,body{margin:0;width:12in;height:6.4in;overflow:hidden}"
        "svg{display:block;width:12in;height:6.4in}"
        "</style></head><body>"
        + svg_path.read_text(encoding="utf-8")
        + "</body></html>",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(browser),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    html_path.unlink()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "event_timeraf_pipeline_overview.svg": pipeline_figure,
        "source_audited_context_builder.svg": context_figure,
        "leakage_safe_event_context_retriever.svg": retrieval_figure,
        "drift_aware_forecast_evidence_head.svg": forecast_figure,
    }
    for name, builder in builders.items():
        path = FIGURE_DIR / name
        builder(path)
        convert_to_pdf(path)
        print(path.name)


if __name__ == "__main__":
    main()
