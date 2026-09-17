# -*- coding: utf-8 -*-
"""
Independently verify agent3's claimed ranking highlights:
1) values unchanged (re-check vs reference: values only)
2) bold/underline counts per dataset x metric, recomputed from scratch
3) folder-root deliverable formatting status (expected: none)
"""
import os
import sys
import re
from openpyxl import load_workbook

WORK = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(WORK)
sys.path.insert(0, WORK)
import spec_canonical as S

AG3 = os.path.join(WORK, "agent_3", S.OUTPUT_FILENAME)
REF = os.path.join(WORK, "reference", S.OUTPUT_FILENAME)
FINAL = os.path.join(BASE, "合并汇总_3源_56模型×147数据集_均值±std.xlsx")

out = []
def log(*a):
    out.append(" ".join(str(x) for x in a))

def load_ws(path, sheet):
    wb = load_workbook(path, read_only=False, data_only=False)
    return wb[sheet]

# ---- 1) value-level comparison agent3 vs reference (both sheets) ----
def matrix_vals(ws):
    return [["" if c.value is None else str(c.value) for c in row] for row in ws.iter_rows()]

diff = 0
for sheet in [S.SHEET1_NAME, S.SHEET2_NAME]:
    a = matrix_vals(load_ws(AG3, sheet))
    b = matrix_vals(load_ws(REF, sheet))
    if len(a) != len(b):
        diff += 1
        continue
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            diff += 1
            continue
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]:
                diff += 1
log("VALUES: agent3 vs reference diff cells =", diff)

# ---- 2) formatting audit on agent3 sheet1 ----
ws = load_ws(AG3, S.SHEET1_NAME)
rows = list(ws.iter_rows())
header = [c.value for c in rows[0]]
# columns: dataset, then per model 3 cols (ARI,NMI,ACC). Identify metric blocks.
metric_cols = {"ARI": [], "NMI": [], "ACC": []}
for idx, h in enumerate(header[1:], start=1):
    if isinstance(h, str) and h.endswith("_ARI"):
        metric_cols["ARI"].append(idx)
    elif isinstance(h, str) and h.endswith("_NMI"):
        metric_cols["NMI"].append(idx)
    elif isinstance(h, str) and h.endswith("_ACC"):
        metric_cols["ACC"].append(idx)

bold_cells = []
underline_cells = []
for r in range(1, len(rows)):
    for met, idxs in metric_cols.items():
        # recompute rank by mean
        entries = []
        for idx in idxs:
            cell = rows[r][idx]
            v = cell.value
            if v is None or str(v) == "NA":
                continue
            m = re.match(r"^(-?\d+\.\d{6})(?:±(-?\d+\.\d{6}))?$", str(v))
            if not m:
                continue
            mean = float(m.group(1))
            entries.append((mean, idx, cell))
        entries.sort(key=lambda t: (-t[0], t[1]))
        if len(entries) >= 1:
            c1 = entries[0][2]
            f1 = c1.font
            if f1.bold:
                bold_cells.append((r, entries[0][1]))
            else:
                bold_cells.append(("MISSING-BOLD", r, entries[0][1]))
        if len(entries) >= 2:
            c2 = entries[1][2]
            f2 = c2.font
            if f2.underline:
                underline_cells.append((r, entries[1][1]))
            else:
                underline_cells.append(("MISSING-UNDERLINE", r, entries[1][1]))

log("agent3 bold cells =", len([b for b in bold_cells if isinstance(b, tuple) and len(b) == 2]),
    "| MISSING/bad:", [b for b in bold_cells if not (isinstance(b, tuple) and len(b) == 2)])
log("agent3 underline cells =", len([u for u in underline_cells if isinstance(u, tuple) and len(u) == 2]),
    "| MISSING/bad:", [u for u in underline_cells if not (isinstance(u, tuple) and len(u) == 2)])

# overlap check: same cell both bold and underline
overlap = set(bold_cells) & set(underline_cells)
log("cells both bold AND underline:", len(overlap))

# ---- 3) folder-root deliverable: has any formatting? ----
wsF = load_ws(FINAL, S.SHEET1_NAME)
fmt_count = 0
for row in wsF.iter_rows():
    for c in row:
        f = c.font
        if f.bold or (f.underline and f.underline != "none" and f.underline is not None):
            fmt_count += 1
log("folder-root FINAL formatted cells (bold or underline):", fmt_count)

with open(os.path.join(WORK, "audit_agent3_format.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("\n".join(out))
