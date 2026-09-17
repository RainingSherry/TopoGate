# -*- coding: utf-8 -*-
"""
Post-format final verification of the official deliverable:
1) values unchanged (cell-text diff vs unformatted reference == 0, both sheets)
2) actual bold/underline sets == independently recomputed rank sets
3) formatting sets == agent3's formatted file formatting sets
"""
import os
import sys
import re
from openpyxl import load_workbook

WORK = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(WORK)
sys.path.insert(0, WORK)
import spec_canonical as S

FINAL = os.path.join(BASE, "合并汇总_3源_56模型×147数据集_均值±std.xlsx")
REF = os.path.join(WORK, "reference", S.OUTPUT_FILENAME)
AG3 = os.path.join(WORK, "agent_3", S.OUTPUT_FILENAME)

out = []
def log(*a):
    out.append(" ".join(str(x) for x in a))

# ---------- 1) values ----------
def valmat(path, sheet):
    wb = load_workbook(path, read_only=True, data_only=False)
    ws = wb[sheet]
    return [["" if c.value is None else str(c.value) for c in row] for row in ws.iter_rows()]

vdiff = 0
for sheet in [S.SHEET1_NAME, S.SHEET2_NAME]:
    a = valmat(FINAL, sheet)
    b = valmat(REF, sheet)
    if len(a) != len(b):
        vdiff += abs(len(a) - len(b))
        continue
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]:
                vdiff += 1
log("1) VALUES diff FINAL vs reference (both sheets):", vdiff)

# ---------- 2/3) formatting ----------
def fmt_sets(path):
    wb = load_workbook(path, read_only=False, data_only=False)
    ws = wb[S.SHEET1_NAME]
    rows = list(ws.iter_rows())
    groups = {"ARI": [], "NMI": [], "ACC": []}
    for c in rows[0][1:]:
        h = str(c.value)
        for met in groups:
            if h.endswith("_" + met):
                groups[met].append(c.column - 1)
                break
    bold, under = set(), set()
    for r in range(1, len(rows)):
        for met, cols in groups.items():
            entries = []
            for ci in cols:
                cell = rows[r][ci]
                v = cell.value
                if v is None or str(v) == "NA":
                    continue
                m = re.match(r"^(-?\d+\.\d{6})(?:±(-?\d+\.\d{6}))?$", str(v).strip())
                if not m:
                    raise ValueError((r, ci, v))
                entries.append((float(m.group(1)), ci, cell))
            entries.sort(key=lambda t: (-t[0], t[1]))
            if len(entries) >= 1 and entries[0][2].font.bold:
                bold.add((r, entries[0][1]))
            if len(entries) >= 2 and entries[1][2].font.underline:
                under.add((r, entries[1][1]))
    return bold, under

bF, uF = fmt_sets(FINAL)
bA, uA = fmt_sets(AG3)

log("2) FINAL bold=%d underline=%d expected %d/%d"
    % (len(bF), len(uF), 147 * 3, 147 * 3))
log("   overlap bold&underline in FINAL:", len(bF & uF))
log("3) FINAL vs agent3 formatting: bold equal=%s underline equal=%s"
    % (bF == bA, uF == uA))
if bF != bA:
    log("   bold diff examples:", list((bF - bA) | (bA - bF))[:5])
if uF != uA:
    log("   underline diff examples:", list((uF - uA) | (uA - uF))[:5])

with open(os.path.join(WORK, "final_verify2.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("\n".join(out))
