# -*- coding: utf-8 -*-
"""Sanity-check reference output sample cells."""
import pandas as pd
import os
WORK = os.path.dirname(os.path.abspath(__file__))
ref = os.path.join(WORK, 'reference', 'merged_result.xlsx')
df = pd.read_excel(ref, sheet_name='合并汇总')
out = []
out.append('shape=%s' % (df.shape,))
for ds in ['20newsgroups', 'Pollen', 'worm_neuron_cell', 'Baron', 'wine']:
    r = df[df['dataset'] == ds]
    row = ['%s: ToPoGate_strict8_ARI=%r dec_ARI=%r scMAE_ARI=%r KMeans_ARI=%r RG-NeighborMix-scMAE_ARI=%r' % (
        ds,
        r['ToPoGate_strict8_ARI'].iloc[0],
        r['dec_ARI'].iloc[0],
        r['scMAE_ARI'].iloc[0],
        r['KMeans_ARI'].iloc[0],
        r['RG-NeighborMix-scMAE_ARI'].iloc[0])]
    out.extend(row)
out.append('all headers (%d): %s' % (len(df.columns), list(df.columns)[:8]))
out.append('last header: %s' % df.columns[-1])
with open(os.path.join(WORK, 'ref_sample.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out))
print('\n'.join(out))
