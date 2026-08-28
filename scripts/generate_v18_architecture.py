#!/usr/bin/env python3
"""Generate the staged TopoGate V18 methodology overview figure."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics import renderPDF, renderPM, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Ellipse, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "papers" / "figures"

FONT_REGULAR = "V18DejaVuSans"
FONT_BOLD = "V18DejaVuSansBold"
pdfmetrics.registerFont(TTFont(FONT_REGULAR, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BOLD, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

WIDTH = 1440
HEIGHT = 980

INK = HexColor("#17202A")
MUTED = HexColor("#52616B")
LIGHT_LINE = HexColor("#B8C2CC")
PANEL = HexColor("#F7F9FB")

DATA_FILL = HexColor("#EEF2F5")
DATA_STROKE = HexColor("#52616B")
REP_FILL = HexColor("#DCECF7")
REP_STROKE = HexColor("#2B6CB0")
GRAPH_FILL = HexColor("#DFF2E5")
GRAPH_STROKE = HexColor("#2F855A")
CORE_FILL = HexColor("#FCE8C9")
CORE_STROKE = HexColor("#B45309")
READOUT_FILL = HexColor("#E9E1F3")
READOUT_STROKE = HexColor("#6B46A1")
OPTION_FILL = HexColor("#FFF5D6")
OPTION_STROKE = HexColor("#9C6B16")
WARNING_FILL = HexColor("#FDE2E2")
WARNING_STROKE = HexColor("#B83232")


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 22,
    color=INK,
    bold: bool = False,
    anchor: str = "start",
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName=FONT_BOLD if bold else FONT_REGULAR,
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def add_lines(
    drawing: Drawing,
    x: float,
    y: float,
    lines: list[str],
    *,
    size: float = 20,
    leading: float = 27,
    color=INK,
    bold_first: bool = False,
) -> None:
    for idx, line in enumerate(lines):
        add_text(
            drawing,
            x,
            y - idx * leading,
            line,
            size=size,
            color=color,
            bold=bold_first and idx == 0,
        )


def add_box(
    drawing: Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill,
    stroke,
    radius: float = 10,
    dashed: bool = False,
    stroke_width: float = 2.2,
) -> None:
    box = Rect(
        x,
        y,
        width,
        height,
        rx=radius,
        ry=radius,
        fillColor=fill,
        strokeColor=stroke,
        strokeWidth=stroke_width,
    )
    if dashed:
        box.strokeDashArray = [9, 6]
    drawing.add(box)


def add_arrow(
    drawing: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color=INK,
    dashed: bool = False,
    width: float = 2.5,
) -> None:
    line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width)
    if dashed:
        line.strokeDashArray = [9, 6]
    drawing.add(line)

    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    head = 12
    half = 5.5
    drawing.add(
        Polygon(
            [
                x2,
                y2,
                x2 - head * ux + half * px,
                y2 - head * uy + half * py,
                x2 - head * ux - half * px,
                y2 - head * uy - half * py,
            ],
            fillColor=color,
            strokeColor=color,
        )
    )


def add_sparse_matrix(drawing: Drawing, x: float, y: float, cell: float = 10) -> None:
    active = {(0, 1), (0, 5), (1, 3), (2, 0), (2, 6), (3, 4), (4, 2), (4, 7), (5, 5), (6, 1), (6, 6), (7, 3)}
    for row in range(8):
        for col in range(8):
            drawing.add(
                Rect(
                    x + col * cell,
                    y + (7 - row) * cell,
                    cell - 1,
                    cell - 1,
                    fillColor=REP_STROKE if (row, col) in active else white,
                    strokeColor=LIGHT_LINE,
                    strokeWidth=0.6,
                )
            )


def add_graph_icon(drawing: Drawing, x: float, y: float) -> None:
    nodes = [(0, 28), (42, 55), (85, 34), (18, 82), (72, 92), (110, 72)]
    edges = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (4, 5)]
    for left, right in edges:
        drawing.add(
            Line(
                x + nodes[left][0],
                y + nodes[left][1],
                x + nodes[right][0],
                y + nodes[right][1],
                strokeColor=GRAPH_STROKE,
                strokeWidth=2,
            )
        )
    for nx, ny in nodes:
        drawing.add(
            Circle(
                x + nx,
                y + ny,
                7,
                fillColor=white,
                strokeColor=GRAPH_STROKE,
                strokeWidth=2.2,
            )
        )


def add_coefficient_icon(drawing: Drawing, x: float, y: float, cell: float = 9) -> None:
    active = {(0, 2), (0, 6), (1, 0), (2, 4), (3, 1), (4, 7), (5, 3), (6, 5), (7, 2)}
    for row in range(8):
        for col in range(8):
            fill = CORE_STROKE if (row, col) in active else white
            if row == col:
                fill = DATA_FILL
            drawing.add(
                Rect(
                    x + col * cell,
                    y + (7 - row) * cell,
                    cell - 1,
                    cell - 1,
                    fillColor=fill,
                    strokeColor=LIGHT_LINE,
                    strokeWidth=0.5,
                )
            )


def add_cluster_icon(drawing: Drawing, x: float, y: float) -> None:
    groups = [
        ([(5, 15), (22, 2), (34, 22), (16, 35)], REP_STROKE),
        ([(70, 10), (88, 28), (105, 8), (112, 35)], GRAPH_STROKE),
        ([(48, 72), (66, 88), (82, 68), (58, 54)], CORE_STROKE),
    ]
    drawing.add(Ellipse(x - 4, y - 8, 48, 52, fillColor=None, strokeColor=REP_STROKE, strokeWidth=1.5))
    drawing.add(Ellipse(x + 64, y - 8, 58, 54, fillColor=None, strokeColor=GRAPH_STROKE, strokeWidth=1.5))
    drawing.add(Ellipse(x + 43, y + 48, 50, 50, fillColor=None, strokeColor=CORE_STROKE, strokeWidth=1.5))
    for points, color in groups:
        for nx, ny in points:
            drawing.add(Circle(x + nx, y + ny, 5.5, fillColor=color, strokeColor=white, strokeWidth=1))


def build_figure() -> Drawing:
    drawing = Drawing(WIDTH, HEIGHT)
    drawing.add(Rect(0, 0, WIDTH, HEIGHT, fillColor=white, strokeColor=None))

    add_text(
        drawing,
        WIDTH / 2,
        940,
        "TopoGate V18: Staged Topology-Native Clustering Architecture",
        size=34,
        bold=True,
        anchor="middle",
    )
    add_text(
        drawing,
        WIDTH / 2,
        906,
        "Solid path: reference model    Dashed path: conditional extension after promotion",
        size=20,
        color=MUTED,
        anchor="middle",
    )

    # Main reference boundary.
    add_box(drawing, 258, 438, 1160, 430, fill=PANEL, stroke=LIGHT_LINE, radius=14, stroke_width=1.8)
    add_text(drawing, 278, 839, "V18-R REFERENCE PATH", size=20, color=MUTED, bold=True)

    # Input semantics.
    add_box(drawing, 25, 465, 205, 370, fill=DATA_FILL, stroke=DATA_STROKE)
    add_text(drawing, 127.5, 797, "Input X", size=25, bold=True, anchor="middle")
    add_text(drawing, 127.5, 768, "N x D matrix", size=18, color=MUTED, anchor="middle")
    add_sparse_matrix(drawing, 87, 657, cell=10)
    add_lines(
        drawing,
        45,
        620,
        [
            "Count: source-declared",
            "log1p + row L2",
            "Continuous: row L2",
            "",
            "No labels in fit",
        ],
        size=17,
        leading=25,
        color=INK,
    )

    # Fixed representation views.
    add_box(drawing, 278, 485, 220, 325, fill=REP_FILL, stroke=REP_STROKE)
    add_text(drawing, 388, 774, "A. Fixed Views", size=25, bold=True, anchor="middle")
    add_text(drawing, 388, 744, "H^(1), ..., H^(V)", size=22, bold=True, anchor="middle")
    add_box(drawing, 300, 612, 176, 106, fill=white, stroke=REP_STROKE, radius=7, stroke_width=1.6)
    add_lines(
        drawing,
        316,
        686,
        ["Reference backend", "sparse projections", "fixed, label-free"],
        size=19,
        leading=25,
        bold_first=True,
    )
    add_lines(
        drawing,
        300,
        575,
        ["Multiple geometry", "views", "Sparse-safe execution", "No latent KMeans"],
        size=17,
        leading=24,
        color=MUTED,
    )

    # Candidate graph.
    add_box(drawing, 528, 485, 225, 325, fill=GRAPH_FILL, stroke=GRAPH_STROKE)
    add_text(drawing, 640.5, 774, "B. Candidate E0", size=25, bold=True, anchor="middle")
    add_graph_icon(drawing, 584, 635)
    add_lines(
        drawing,
        550,
        608,
        [
            "Reference: cosine union",
            "Ablation: cosine / SNN",
            "Support constraint only",
            "No forced non-zero edge",
            "No full N x N matrix",
        ],
        size=18,
        leading=27,
        color=INK,
    )

    # Topology-native core.
    add_box(drawing, 783, 465, 365, 365, fill=CORE_FILL, stroke=CORE_STROKE, stroke_width=3)
    add_text(drawing, 965.5, 795, "C. Relation Solver", size=24, bold=True, anchor="middle")
    add_text(drawing, 965.5, 764, "Candidate-restricted", size=16, color=MUTED, anchor="middle")
    add_text(drawing, 965.5, 741, "robust self-expression", size=16, color=MUTED, anchor="middle")
    add_box(drawing, 806, 620, 319, 112, fill=white, stroke=CORE_STROKE, radius=7, stroke_width=1.6)
    add_lines(
        drawing,
        821,
        710,
        [
            "min_C mean_v Huber(H^v - C H^v)",
            "+ lambda1 ||C||_1",
            "+ .5 lambda2 ||C||_F^2",
            "diag(C)=0; supp(C) subset E0",
        ],
        size=15,
        leading=23,
    )
    add_coefficient_icon(drawing, 821, 525, cell=9)
    add_text(drawing, 912, 583, "FISTA -> sparse C", size=18, bold=True)
    add_text(drawing, 912, 558, "supp(C) = exact-zero gate", size=16)
    add_text(drawing, 912, 535, "edge strength = |C_ij|", size=16)
    add_lines(
        drawing,
        806,
        493,
        ["No second utility / teacher / binary G", "One relation object closes the objective"],
        size=14,
        leading=18,
        color=MUTED,
    )

    # Same-C readout.
    add_box(drawing, 1178, 485, 220, 325, fill=READOUT_FILL, stroke=READOUT_STROKE)
    add_text(drawing, 1288, 774, "D. Same-C Readout", size=19, bold=True, anchor="middle")
    add_lines(
        drawing,
        1197,
        731,
        [
            "A = |C| + |C.T|",
            "L_sym = I - D^-1/2",
            "        A D^-1/2",
            "spectral U -> KMeans",
        ],
        size=14,
        leading=26,
    )
    add_cluster_icon(drawing, 1223, 545)
    add_text(drawing, 1288, 520, "Final clusters", size=18, bold=True, anchor="middle")
    add_text(drawing, 1288, 496, "degree-zero -> -1", size=14, color=MUTED, anchor="middle")

    # Solid data-flow arrows.
    add_arrow(drawing, 230, 650, 278, 650, color=DATA_STROKE)
    add_arrow(drawing, 498, 650, 528, 650, color=REP_STROKE)
    add_arrow(drawing, 753, 650, 783, 650, color=GRAPH_STROKE)
    add_arrow(drawing, 1148, 650, 1178, 650, color=CORE_STROKE)

    # Training boundary panel.
    add_box(drawing, 25, 170, 205, 250, fill=DATA_FILL, stroke=DATA_STROKE)
    add_text(drawing, 127.5, 386, "Training", size=20, bold=True, anchor="middle")
    add_text(drawing, 127.5, 361, "Boundary", size=20, bold=True, anchor="middle")
    add_lines(
        drawing,
        45,
        325,
        [
            "Fit: no y, no K",
            "Readout: K only",
            "Post-hoc: y -> ARI/NMI",
            "Labels excluded",
        ],
        size=14,
        leading=27,
        color=INK,
    )

    # Conditional extension boundary.
    add_box(drawing, 258, 170, 1160, 250, fill=OPTION_FILL, stroke=OPTION_STROKE, radius=12, dashed=True)
    add_text(drawing, 278, 390, "CONDITIONAL EXTENSIONS: TEST ONE AT A TIME", size=20, color=OPTION_STROKE, bold=True)

    add_box(drawing, 282, 198, 330, 158, fill=white, stroke=OPTION_STROKE, radius=8, dashed=True, stroke_width=1.8)
    add_text(drawing, 447, 326, "V18-Latent / Adapter", size=23, bold=True, anchor="middle")
    add_lines(
        drawing,
        302,
        292,
        [
            "Frozen scMAE -> latent Z",
            "H = Norm(Z + gamma R(Z))",
            "anchor + variance + view stability",
            "never add jointly in the first test",
        ],
        size=17,
        leading=25,
    )

    add_box(drawing, 640, 198, 390, 158, fill=white, stroke=WARNING_STROKE, radius=8, dashed=True, stroke_width=1.8)
    add_text(drawing, 835, 326, "V18-BinaryGate Ablation", size=23, bold=True, anchor="middle")
    add_lines(
        drawing,
        660,
        292,
        [
            "phi_ij: cosine, mutual, SNN-Jaccard,",
            "view recurrence, view stability",
            "HardConcrete G; compare C = G o W",
            "only after L1 support is useful",
        ],
        size=17,
        leading=25,
    )

    add_box(drawing, 1058, 198, 330, 158, fill=white, stroke=OPTION_STROKE, radius=8, dashed=True, stroke_width=1.8)
    add_text(drawing, 1223, 326, "V18-ZEUS Transfer", size=23, bold=True, anchor="middle")
    add_lines(
        drawing,
        1078,
        292,
        [
            "Freeze the original ZEUS encoder",
            "Z1: native embedding -> KMeans",
            "Z2: embedding -> same V18 core",
            "report as TopoGate-ZEUS if adapted",
        ],
        size=17,
        leading=25,
    )

    # Conditional arrows point to the module they would replace or augment.
    add_arrow(drawing, 447, 356, 388, 485, color=OPTION_STROKE, dashed=True, width=2)
    add_arrow(drawing, 835, 356, 965, 465, color=WARNING_STROKE, dashed=True, width=2)
    add_arrow(drawing, 1223, 356, 470, 485, color=OPTION_STROKE, dashed=True, width=2)

    # Promotion strip.
    add_box(drawing, 25, 25, 1393, 110, fill=PANEL, stroke=LIGHT_LINE, radius=10, stroke_width=1.5)
    add_text(drawing, 48, 101, "PROMOTION GATES BEFORE ANY DASHED MODULE", size=20, bold=True, color=MUTED)
    checks = [
        (48, "candidate support is usable"),
        (370, "C is non-degenerate"),
        (648, "retained support improves"),
        (1002, "same-C clustering beats ungated / shuffled"),
    ]
    for x, label in checks:
        drawing.add(Rect(x, 49, 19, 19, fillColor=white, strokeColor=GRAPH_STROKE, strokeWidth=2))
        drawing.add(Line(x + 4, 58, x + 8, 53, strokeColor=GRAPH_STROKE, strokeWidth=2.4))
        drawing.add(Line(x + 8, 53, x + 16, 65, strokeColor=GRAPH_STROKE, strokeWidth=2.4))
        add_text(drawing, x + 28, 51, label, size=17, color=INK)

    return drawing


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    drawing = build_figure()

    svg_path = OUTPUT_DIR / "v18_expected_architecture.svg"
    pdf_path = OUTPUT_DIR / "v18_expected_architecture.pdf"
    png_path = OUTPUT_DIR / "v18_expected_architecture.png"

    renderSVG.drawToFile(drawing, str(svg_path))
    renderPDF.drawToFile(drawing, str(pdf_path))

    png_written = False
    try:
        renderPM.drawToFile(drawing, str(png_path), fmt="PNG", dpi=160)
        png_written = True
    except Exception as exc:
        print(f"PNG preview skipped: {exc.__class__.__name__}")

    svg_path.chmod(0o644)
    pdf_path.chmod(0o644)
    if png_written:
        png_path.chmod(0o644)

    print(svg_path)
    print(pdf_path)
    if png_written:
        print(png_path)


if __name__ == "__main__":
    main()
