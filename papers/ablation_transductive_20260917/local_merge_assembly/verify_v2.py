# -*- coding: utf-8 -*-
"""Verify the v2 deliverable (with iris):
- counts NA / plusminus / mean-only
- bold & underline sets (441/441, no overlap)
- iris row: values + which cells got highlighted (rank rule check)
- diff vs the old deliverable / agent1 file: only iris ToPoGate cells changed
"""
import os
import re
import sys
from openpyxl import load_workbook

BASE = r"S:\Creation\写论文\武汉理工\论文们\TopoGate\papers\实验\0911调参以及全模型x数据集汇总"
V2 = os.path.join(BASE, "合并汇总_3源_56模型×147数据集_均值±std_含iris.xlsx")
OLD = os.path.join(BASE, "合并汇总_3源_56模型×147数据集_均值±std.xlsx")
AGENT1 = os.path.join(BASE, "__work__", "agent_1", "merged_result.xlsx")

wb = load_workbook(V2)
ws = wb["合并汇总"]
rows = list(ws.iter_rows())
header = [c.value for c in rows[0]]
ds_idx = header.index("dataset")

# --- counts ---
na = plus = mean_only = 0
bold = set()
under = set()
for r in range(1, len(rows)):
    for c in rows[r][1:]:
        v = c.value
        if v is None or str(v) == "NA":
            na += 1
        elif "±" in str(v):
            plus += 1
        elif re.match(r"^-?\d+\.\d{6}$", str(v)):
            mean_only += 1
        if c.font is not None and c.font.bold:
            bold.add((r, c.column))
        if c.font is not None and c.font.underline == "single":
            under.add((r, c.column))

# --- iris row ---
iris_r = None
for r in range(1, len(rows)):
    if str(rows[r][ds_idx].value).strip().lower() == "iris":
        iris_r = r
        break
iris_out = []
for c in rows[iris_r]:
    h = str(c.value)
    if h.startswith("ToPoGate_strict8_") or h in ("ARI", "NMI", "ACC") or h == "dataset":
        pass
tp_cols = [i for i, c in enumerate(rows[0]) if c.value and str(c.value).startswith("ToPoGate_strict8_")]
iris_out.append("iris ToPoGate cells: " + ", ".join("%s=%s" % (rows[0][i].value, rows[iris_r][i].value) for i in tp_cols))
iris_high = [(rows[0][c.column - 1].value, "bold" if (iris_r, c.column) in bold else ("underline" if (iris_r, c.column) in under else "")) for c in rows[iris_r][1:] if (iris_r, c.column) in bold or (iris_r, c.column) in under]
iris_out.append("iris row highlighted cells: " + str([h for h in iris_high if h[1]]))

# --- diff vs old deliverable (values only) ---
def load_values(path):
    w = load_workbook(path, read_only=True)
    s = w["合并汇总"]
    return [[c.value for c in row] for row in s.iter_rows()]

old = load_values(OLD)
new = load_values(V2)
diffs = []
if len(old) == len(new) and len(old[0]) == len(new[0]):
    for r in range(len(old)):
        for c in range(len(old[0])):
            a, b = old[r][c], new[r][c]
            if (a is None or str(a) == "nan") != (b is None or str(b) == "nan") or (a is not None and str(a) != str(b)):
                diffs.append((r, c, rows[0][c].value if r == 0 else old[r][0], a, b))
else:
    diffs.append(("SHAPE", len(old), len(new)))

out = [
    "V2 counts: NA=%d plusminus=%d mean-only=%d total=%d" % (na, plus, mean_only, na + plus + mean_only),
    "bold=%d underline=%d overlap=%d" % (len(bold), len(under), len(bold & under)),
    "diff cells vs old deliverable: %d" % len(diffs),
] + ["  %s" % str(d) for d in diffs[:20]] + iris_out
with open(os.path.join(BASE, "__work__", "verify_v2.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("\n".join(out))
