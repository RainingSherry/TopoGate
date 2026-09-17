# -*- coding: utf-8 -*-
"""Check cross-file overlaps: same-name models on shared datasets -> identical values?"""
import pandas as pd
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pA = os.path.join(base, 'TopoGate_CLUBench自带_Benchmark.xlsx')
pB = os.path.join(base, 'PlantNet_23模型×16数据集_ARI_NMI_ACC.xlsx')
pC = os.path.join(base, 'TopoGateBenchmark.xlsx')

A = pd.read_excel(pA, sheet_name='Benchmark')
C = pd.read_excel(pC, sheet_name='Benchmark')
B = pd.read_excel(pB)

def model_cols(df, model, order):
    """order: list of 6 suffixes in the file's native order"""
    return [model + s for s in order]

ORDER_A = ['_ARI_mean', '_NMI_mean', '_ACC_mean', '_ARI_std', '_NMI_std', '_ACC_std']
ORDER_B = ['_ARI_mean', '_ARI_std', '_NMI_mean', '_NMI_std', '_ACC_mean', '_ACC_std']
ORDER_C = ['_ARI_mean', '_NMI_mean', '_ACC_mean', '_ARI_std', '_NMI_std', '_ACC_std']

out = []
def log(*a):
    out.append(' '.join(str(x) for x in a))

# 1) A vs C: ToPoGate_strict8 on all 147 datasets
mA, mC = 'ToPoGate_strict8', 'ToPoGate_strict8'
ca, cc = model_cols(A, mA, ORDER_A), model_cols(C, mC, ORDER_C)
m = A[['dataset'] + ca].merge(C[['dataset'] + cc], on='dataset', suffixes=('_A', '_C'))
diff = 0
for s in ORDER_A:
    colA, colC = mA + s + '_A', mC + s + '_C'
    d = (~m[colA].isna()) & (~m[colC].isna()) & (m[colA] != m[colC])
    n = int(d.sum())
    diff += n
    if n:
        ex = m.loc[d, ['dataset', colA, colC]].head(3).to_string()
        log('A vs C model=%s suffix=%s diffs=%d example:\n%s' % (mA, s, n, ex))
log('A vs C ToPoGate_strict8 total differing overlapping cells:', diff)

# 2) A vs C other shared literal names
modelsA = sorted({c.rsplit('_', 2)[0] for c in A.columns if c != 'dataset'})
modelsC = sorted({c.rsplit('_', 2)[0] for c in C.columns if c != 'dataset'})
shared = sorted(set(modelsA) & set(modelsC))
log('shared literal model names A&C:', shared)
for m in shared:
    ca, cc = model_cols(A, m, ORDER_A), model_cols(C, m, ORDER_C)
    mm = A[['dataset'] + ca].merge(C[['dataset'] + cc], on='dataset', suffixes=('_A', '_C'))
    nd = 0
    for s in ORDER_A:
        colA, colC = m + s + '_A', m + s + '_C'
        nd += int((~mm[colA].isna() & ~mm[colC].isna() & (mm[colA] != mm[colC])).sum())
    log('  %-22s overlapping differing cells=%d' % (m, nd))

# 3) A vs B shared literal names on the 16 bio datasets
modelsB = sorted({c.rsplit('_', 2)[0] for c in B.columns if c != 'dataset'})
sharedAB = sorted(set(modelsA) & set(modelsB))
log('shared literal model names A&B:', sharedAB)
for m in sharedAB:
    ca = model_cols(A, m, ORDER_A)
    cb = model_cols(B, m, ORDER_B)
    mm = A[['dataset'] + ca].merge(B[['dataset'] + cb], on='dataset', suffixes=('_A', '_B'))
    nd = 0
    for sa, sb in zip(ORDER_A, ORDER_B):
        colA, colB = m + sa + '_A', m + sb + '_B'
        nd += int((~mm[colA].isna() & ~mm[colB].isna() & (mm[colA] != mm[colB])).sum())
    log('  %-22s overlapping differing cells=%d' % (m, nd))

# 4) B vs C shared literal names
sharedBC = sorted(set(modelsB) & set(modelsC))
log('shared literal model names B&C:', sharedBC)
for m in sharedBC:
    cb = model_cols(B, m, ORDER_B)
    cc = model_cols(C, m, ORDER_C)
    mm = B[['dataset'] + cb].merge(C[['dataset'] + cc], on='dataset', suffixes=('_B', '_C'))
    nd = 0
    for sb, sc in zip(ORDER_B, ORDER_C):
        colB, colC = m + sb + '_B', m + sc + '_C'
        nd += int((~mm[colB].isna() & ~mm[colC].isna() & (mm[colB] != mm[colC])).sum())
    log('  %-22s overlapping differing cells=%d' % (m, nd))

# 5) case/space variants across files: kmeans vs KMeans, PCA + KMeans vs PCA+KMeans, scdeepcluster vs scDeepCluster, scdcc vs scCDCG, idec vs IDEC, edesc vs EDESC
def get(df, model, suffix):
    col = model + suffix
    return df[['dataset', col]] if col in df.columns else None

pairs = [
    (A, 'kmeans', C, 'KMeans'),
    (B, 'PCA + KMeans', C, 'PCA+KMeans'),
    (B, 'scdeepcluster', C, 'scDeepCluster'),
    (B, 'scdcc', C, 'scCDCG'),
    (A, 'idec', C, 'IDEC'),
    (A, 'edesc', C, 'EDESC'),
    (B, 'scMAE', C, 'scMAE'),
    (B, 'scname', C, 'scNAME'),
]
for df1, m1, df2, m2 in pairs:
    g1 = get(df1, m1, '_ARI_mean'); g2 = get(df2, m2, '_ARI_mean')
    if g1 is None or g2 is None:
        log('pair %s(%s) vs %s(%s): missing cols' % (m1, 'A/B?', m2, 'C?'))
        continue
    mm = g1.merge(g2, on='dataset', suffixes=('_1', '_2'))
    cols = list(mm.columns)
    c1 = m1 + '_ARI_mean_1' if m1 + '_ARI_mean_1' in cols else (m1 + '_ARI_mean' if m1 + '_ARI_mean' in cols else None)
    c2 = m2 + '_ARI_mean_2' if m2 + '_ARI_mean_2' in cols else (m2 + '_ARI_mean' if m2 + '_ARI_mean' in cols else None)
    if c1 is None or c2 is None:
        log('  %-16s vs %-16s: cannot locate cols %s' % (m1, m2, cols))
        continue
    both = mm.dropna(subset=[c1, c2])
    nd = int((both[c1] != both[c2]).sum())
    log('  %-16s vs %-16s: common datasets=%d, ARI_mean differing=%d' % (m1, m2, len(both), nd))

report = os.path.join(base, '__work__', 'overlap_check.txt')
with open(report, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out))
print('done')
