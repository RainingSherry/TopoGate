#!/usr/bin/env python3
"""Generate the scMAE-mainline TopoGate V18 architecture figure."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics import renderPDF, renderPM, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Ellipse, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "papers" / "figures"
FONT_REGULAR = "V18MainDejaVuSans"
FONT_BOLD = "V18MainDejaVuSansBold"
pdfmetrics.registerFont(TTFont(FONT_REGULAR, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BOLD, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

W, H = 1600, 1050
INK = HexColor("#17202A")
MUTED = HexColor("#52616B")
LINE = HexColor("#B8C2CC")
PANEL = HexColor("#F7F9FB")
DATA_FILL, DATA_STROKE = HexColor("#EEF2F5"), HexColor("#52616B")
MAE_FILL, MAE_STROKE = HexColor("#DCECF7"), HexColor("#2B6CB0")
GRAPH_FILL, GRAPH_STROKE = HexColor("#DFF2E5"), HexColor("#2F855A")
GATE_FILL, GATE_STROKE = HexColor("#FCE8C9"), HexColor("#B45309")
OUT_FILL, OUT_STROKE = HexColor("#E9E1F3"), HexColor("#6B46A1")
OPTION_FILL, OPTION_STROKE = HexColor("#FFF5D6"), HexColor("#9C6B16")
RED = HexColor("#B83232")


def text(d, x, y, value, size=20, color=INK, bold=False, anchor="start"):
    d.add(
        String(
            x,
            y,
            value,
            fontName=FONT_BOLD if bold else FONT_REGULAR,
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def lines(d, x, y, values, size=18, leading=25, color=INK, bold_first=False):
    for i, value in enumerate(values):
        text(d, x, y - i * leading, value, size, color, bold_first and i == 0)


def box(d, x, y, w, h, fill, stroke, dashed=False, radius=10, width=2.2):
    shape = Rect(x, y, w, h, rx=radius, ry=radius, fillColor=fill, strokeColor=stroke, strokeWidth=width)
    if dashed:
        shape.strokeDashArray = [9, 6]
    d.add(shape)


def arrow(d, x1, y1, x2, y2, color=INK, dashed=False, width=2.5):
    line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width)
    if dashed:
        line.strokeDashArray = [9, 6]
    d.add(line)
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head, half = 13, 6
    d.add(
        Polygon(
            [x2, y2, x2 - head * ux + half * px, y2 - head * uy + half * py,
             x2 - head * ux - half * px, y2 - head * uy - half * py],
            fillColor=color,
            strokeColor=color,
        )
    )


def matrix(d, x, y, color, n=8, cell=10):
    active = {(0, 1), (0, 5), (1, 3), (2, 0), (2, 6), (3, 4), (4, 2), (4, 7), (5, 5), (6, 1), (6, 6), (7, 3)}
    for r in range(n):
        for c in range(n):
            d.add(Rect(x + c * cell, y + (n - 1 - r) * cell, cell - 1, cell - 1,
                       fillColor=color if (r, c) in active else white,
                       strokeColor=LINE, strokeWidth=0.6))


def latent_icon(d, x, y):
    groups = [([(0, 12), (22, 0), (38, 24), (15, 37)], MAE_STROKE),
              ([(71, 10), (90, 30), (112, 6), (102, 39)], GRAPH_STROKE),
              ([(44, 67), (62, 87), (84, 69), (57, 51)], GATE_STROKE)]
    for points, color in groups:
        cx = sum(p[0] for p in points) / len(points) + x
        cy = sum(p[1] for p in points) / len(points) + y
        d.add(Ellipse(cx - 28, cy - 25, 65, 55, fillColor=None, strokeColor=color, strokeWidth=1.5))
        for px, py in points:
            d.add(Circle(x + px, y + py, 6, fillColor=color, strokeColor=white, strokeWidth=1))


def graph_icon(d, x, y):
    pts = [(0, 20), (42, 48), (85, 30), (18, 78), (72, 88), (110, 68)]
    edges = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (4, 5)]
    for a, b in edges:
        d.add(Line(x + pts[a][0], y + pts[a][1], x + pts[b][0], y + pts[b][1],
                   strokeColor=GRAPH_STROKE, strokeWidth=2))
    for px, py in pts:
        d.add(Circle(x + px, y + py, 7, fillColor=white, strokeColor=GRAPH_STROKE, strokeWidth=2))


def cluster_icon(d, x, y):
    for cx, cy, color, w, h in [(30, 28, MAE_STROKE, 65, 60), (110, 26, GRAPH_STROKE, 70, 62),
                                 (72, 92, GATE_STROKE, 62, 58)]:
        d.add(Ellipse(x + cx - w / 2, y + cy - h / 2, w, h, fillColor=None, strokeColor=color, strokeWidth=1.6))
    groups = [([(8, 28), (25, 15), (42, 35), (22, 48)], MAE_STROKE),
              ([(90, 20), (108, 38), (128, 16), (116, 50)], GRAPH_STROKE),
              ([(55, 82), (72, 101), (91, 82), (66, 64)], GATE_STROKE)]
    for points, color in groups:
        for px, py in points:
            d.add(Circle(x + px, y + py, 5.5, fillColor=color, strokeColor=white, strokeWidth=1))


def build():
    d = Drawing(W, H)
    d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))

    text(d, W / 2, 1008, "TopoGate V18: scMAE-Latent Topological Gating", 36, bold=True, anchor="middle")
    text(d, W / 2, 975, "scMAE is the main representation learner; the topology gate operates on its latent relation graph", 20, MUTED, anchor="middle")

    # Mainline boundary.
    box(d, 24, 420, 1552, 520, PANEL, LINE, radius=15, width=1.8)
    text(d, 48, 910, "V18 MAINLINE", 21, MUTED, bold=True)

    # A: input.
    box(d, 42, 490, 215, 365, DATA_FILL, DATA_STROKE)
    text(d, 149, 818, "Input X", 28, bold=True, anchor="middle")
    text(d, 149, 788, "N x D sparse matrix", 18, MUTED, anchor="middle")
    matrix(d, 104, 674, MAE_STROKE, cell=10)
    lines(d, 62, 638, ["Count: source-declared", "log1p + row L2", "continuous: row L2", "labels excluded from fit"], 17, 27)

    # B: scMAE.
    box(d, 285, 475, 335, 395, MAE_FILL, MAE_STROKE, width=3)
    text(d, 452.5, 835, "A. scMAE Encoder", 26, bold=True, anchor="middle")
    text(d, 452.5, 806, "masked representation learning", 17, MUTED, anchor="middle")
    box(d, 308, 700, 289, 75, white, MAE_STROKE, radius=7, width=1.6)
    text(d, 452.5, 742, "T_m X -> Encoder E_theta -> Z", 19, bold=True, anchor="middle")
    text(d, 452.5, 716, "Z in R^(N x d), dense latent code", 17, anchor="middle")
    matrix(d, 330, 575, MAE_STROKE, cell=10)
    lines(d, 438, 648, ["Decoder -> Xhat", "L_MAE = masked recon", "Z trainable in V18"], 16, 25)
    text(d, 452.5, 535, "latent perturbation views H^(m)", 16, MUTED, anchor="middle")

    # C: candidates.
    box(d, 648, 475, 235, 395, GRAPH_FILL, GRAPH_STROKE)
    text(d, 765.5, 835, "B. Latent Graph", 25, bold=True, anchor="middle")
    text(d, 765.5, 806, "candidate support E0", 18, MUTED, anchor="middle")
    graph_icon(d, 704, 685)
    lines(d, 669, 642, ["Z -> cosine / SNN kNN", "E0(i): allowed donor edges", "no forced non-zero edge", "no labels or K in fit"], 16, 25)
    box(d, 669, 480, 193, 78, white, GRAPH_STROKE, radius=7, width=1.5)
    lines(d, 682, 551, ["edge features phi_ij", "cosine + mutual + SNN", "recurrence + stability"], 14, 19)

    # D: gate and relation.
    box(d, 910, 455, 385, 425, GATE_FILL, GATE_STROKE, width=3)
    text(d, 1102.5, 845, "C. Latent Topology Gate", 25, bold=True, anchor="middle")
    text(d, 1102.5, 817, "edge decision + sparse relation", 17, MUTED, anchor="middle")
    box(d, 933, 690, 339, 100, white, GATE_STROKE, radius=7, width=1.6)
    lines(d, 950, 762, ["G_ij = HardConcrete(phi_ij)", "W: candidate-restricted relation", "C = G o W"], 18, 27)
    matrix(d, 950, 563, GATE_STROKE, cell=10)
    lines(d, 1062, 641, ["unrolled sparse solver", "implicit gradient", "supp(C): topology gate", "|C_ij|: relation strength"], 15, 23)
    lines(d, 933, 528, ["L_topo = mean_m Huber(H^m - C H^m)", "+ lambda1 ||C||_1 + lambda2 ||W||_F^2", "view stability + anti-collapse safeguards"], 15, 22, MUTED)

    # E: readout.
    box(d, 1322, 490, 230, 365, OUT_FILL, OUT_STROKE)
    text(d, 1437, 818, "D. Readout", 25, bold=True, anchor="middle")
    lines(d, 1342, 775, ["A = |C| + |C.T|", "L_sym(A)", "spectral U -> KMeans", "or Leiden readout"], 16, 28)
    cluster_icon(d, 1365, 565)
    text(d, 1437, 536, "final clusters", 20, bold=True, anchor="middle")
    text(d, 1437, 510, "degree-zero -> abstain", 15, MUTED, anchor="middle")

    # Main arrows.
    arrow(d, 257, 670, 285, 670, DATA_STROKE)
    arrow(d, 620, 670, 648, 670, MAE_STROKE)
    arrow(d, 883, 670, 910, 670, GRAPH_STROKE)
    arrow(d, 1295, 670, 1322, 670, GATE_STROKE)

    # Loss feedback arrows make the mainline meaning explicit.
    arrow(d, 1102, 455, 560, 475, RED, dashed=True, width=2)
    text(d, 780, 438, "L_topo gradient updates E_theta", 14, RED, bold=True, anchor="middle")

    # Training / control band.
    box(d, 24, 170, 1552, 205, OPTION_FILL, OPTION_STROKE, dashed=True, radius=12)
    text(d, 48, 345, "TRAINING SCHEDULE, CONTROLS, AND SAFEGUARDS", 20, OPTION_STROKE, bold=True)

    box(d, 45, 195, 375, 118, white, OPTION_STROKE, radius=7, width=1.6)
    text(d, 232.5, 287, "Joint optimization", 20, bold=True, anchor="middle")
    lines(d, 65, 260, ["1. MAE warm-up", "2. topology fine-tuning", "3. readout from the same A(C)"], 16, 23)

    box(d, 445, 195, 535, 118, white, OPTION_STROKE, radius=7, width=1.6)
    text(d, 712.5, 287, "Required ablations", 20, bold=True, anchor="middle")
    lines(d, 465, 260, ["scMAE-only | no-Gate candidate graph", "V17 exact-zero C | shuffled E0", "full V18: scMAE + G + W"], 16, 23)

    box(d, 1005, 195, 550, 118, white, OPTION_STROKE, radius=7, width=1.6)
    text(d, 1280, 287, "Unsupervised safeguards", 20, bold=True, anchor="middle")
    lines(d, 1025, 260, ["continuation for gate sparsity and temperature", "variance / anchor / view-stability checks", "K only at readout; labels only for metrics"], 16, 23)

    # Conditional transfer callouts.
    box(d, 24, 32, 1552, 105, PANEL, LINE, radius=10, width=1.5)
    text(d, 48, 108, "EXPECTED CLAIM", 19, MUTED, bold=True)
    text(d, 48, 75, "scMAE repairs the input representation; latent topology gating selects and weights relations.", 18, bold=True)
    text(d, 48, 49, "The same C must improve the final graph readout; later extensions include bounded adapter / ZEUS transfer.", 16, MUTED)

    return d


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d = build()
    svg = OUTPUT_DIR / "v18_scmae_mainline.svg"
    pdf = OUTPUT_DIR / "v18_scmae_mainline.pdf"
    png = OUTPUT_DIR / "v18_scmae_mainline.png"
    renderSVG.drawToFile(d, str(svg))
    renderPDF.drawToFile(d, str(pdf))
    png_written = False
    try:
        renderPM.drawToFile(d, str(png), fmt="PNG", dpi=160)
        png_written = True
    except Exception as exc:
        print(f"PNG preview skipped: {exc.__class__.__name__}")
    svg.chmod(0o644)
    pdf.chmod(0o644)
    if png_written:
        png.chmod(0o644)
    print(svg)
    print(pdf)
    if png_written:
        print(png)


if __name__ == "__main__":
    main()
