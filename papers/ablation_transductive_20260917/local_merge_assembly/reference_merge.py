# -*- coding: utf-8 -*-
"""
REFERENCE implementation (written by the orchestrator, independent of the agents).
Merges three benchmark xlsx into one: rows = datasets (147), per model 3 columns
(ARI/NMI/ACC), cell = "mean±std" (6 decimals) or mean-only or "NA".
"""
import os
import sys
import pandas as pd

WORK = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(WORK)
sys.path.insert(0, WORK)
import spec_canonical as S

IN_A = os.path.join(BASE, S.SRC_A_NAME)
IN_B = os.path.join(BASE, S.SRC_B_NAME)
IN_C = os.path.join(BASE, S.SRC_C_NAME)

A = pd.read_excel(IN_A, sheet_name="Benchmark")
B = pd.read_excel(IN_B)
C = pd.read_excel(IN_C)

SUFX_A = ["_ARI_mean", "_NMI_mean", "_ACC_mean", "_ARI_std", "_NMI_std", "_ACC_std"]
SUFX_B = ["_ARI_mean", "_ARI_std", "_NMI_mean", "_NMI_std", "_ACC_mean", "_ACC_std"]
SUFX_C = ["_ARI_mean", "_NMI_mean", "_ACC_mean", "_ARI_std", "_NMI_std", "_ACC_std"]
SUFX_ALL = ["_ARI_mean", "_NMI_mean", "_ACC_mean", "_ARI_std", "_NMI_std", "_ACC_std"]
METRICS = ["ARI", "NMI", "ACC"]


def model_order(df, suffixes):
    """models in header (first-appearance) order."""
    out = []
    for c in df.columns:
        if c == "dataset":
            continue
        for suf in suffixes:
            if str(c).endswith(suf):
                m = str(c)[: -len(suf)]
                if m not in out:
                    out.append(m)
                break
    return out


mA = model_order(A, SUFX_A)
mB = model_order(B, SUFX_B)
mC = model_order(C, SUFX_C)

# model universe: A order, then B (excluding dupes and scMAE), then C (excluding dupes)
models = list(mA)
for m in mB:
    if m not in models and m != "scMAE":
        models.append(m)
for m in mC:
    if m not in models:
        models.append(m)

# sources per model (priority A > B > C)
sources = {}
for m in models:
    src = []
    if m in mA:
        src.append("A")
    if m in mB and not (m == "scMAE"):
        src.append("B")
    if m in mC:
        src.append("C")
    sources[m] = src

dfmap = {"A": A, "B": B, "C": C}
sufmap = {"A": SUFX_A, "B": SUFX_B, "C": SUFX_C}

datasets = [str(d) for d in A["dataset"]]

# build merged matrix
rows = []
for ds in datasets:
    row = {"dataset": ds}
    for m in models:
        for met in METRICS:
            mean = None
            std = None
            for src in sources[m]:
                df = dfmap[src]
                key_mean = m + "_" + met + "_mean"
                key_std = m + "_" + met + "_std"
                if key_mean in df.columns:
                    v = df.loc[df["dataset"] == ds, key_mean]
                    if len(v) and pd.notna(v.iloc[0]):
                        mean = float(v.iloc[0])
                        vs = df.loc[df["dataset"] == ds, key_std]
                        if len(vs) and pd.notna(vs.iloc[0]):
                            std = float(vs.iloc[0])
                        break
            col = m + "_" + met
            if mean is None:
                row[col] = "NA"
            elif std is None:
                row[col] = "%.6f" % mean
            else:
                row[col] = "%.6f±%.6f" % (mean, std)
    rows.append(row)

cols = ["dataset"] + [m + "_" + met for m in models for met in METRICS]
merged = pd.DataFrame(rows, columns=cols)

# sheet 2: fixed notes + per-model table
per_model = []
num_cols = ["dataset"] + [m + "_ARI" for m in models]
for m in models:
    n_ds = 0
    n_cells = 0
    for ds in datasets:
        vals = [merged.loc[merged["dataset"] == ds, m + "_" + met].iloc[0] for met in METRICS]
        if any(v != "NA" for v in vals):
            n_ds += 1
            n_cells += sum(1 for v in vals if v != "NA")
    per_model.append([m, "+".join(sources[m]), n_ds, n_cells])

notes = [list(r) for r in S.NOTES_FIXED]
notes.append(["", ""])
notes.append(["模型清单", ""])
header_note = ["模型", "来源", "覆盖数据集数", "非NA单元格数(3指标合计)"]
notes_body = [header_note] + per_model

out_dir = os.path.join(WORK, "reference")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, S.OUTPUT_FILENAME)
with pd.ExcelWriter(out_path, engine="openpyxl") as w:
    merged.to_excel(w, sheet_name=S.SHEET1_NAME, index=False)
    pd.DataFrame(notes + notes_body).to_excel(w, sheet_name=S.SHEET2_NAME, index=False, header=False)

# report
n_na = int((merged.drop(columns=["dataset"]) == "NA").sum().sum())
n_plus = int(merged.drop(columns=["dataset"]).apply(lambda c: c.str.contains("±", na=False)).sum().sum())
n_mean_only = int(merged.drop(columns=["dataset"]).apply(
    lambda c: c.str.contains(r"^-?\d+\.\d{6}$", na=False)).sum().sum())
lines = [
    "models=%d datasets=%d cols=%d" % (len(models), len(datasets), len(cols)),
    "cell count=%d  NA=%d  plusminus=%d  mean-only=%d" % (len(datasets) * len(models) * 3, n_na, n_plus, n_mean_only),
    "models: %s" % models,
    "output: %s" % out_path,
]
with open(os.path.join(out_dir, "reference_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\n".join(lines))
