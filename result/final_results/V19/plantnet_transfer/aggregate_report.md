# V19 PlantNet-ARI fixed-parameter PCA200 evaluation

- Completed: 48/48
- Audit: passed
- Labels were excluded from preprocessing and fitting; benchmark labels supplied K and post-fit metrics only.
- The fixed configuration was selected with ARI in PlantNet, so this is benchmark-transfer evidence, not label-free tuning.

| Dataset | RG ARI mean+-std | matched scMAE ARI mean+-std | paired delta | old V19 RG | archived best |
|---|---:|---:|---:|---:|---:|
| mouse_retina__clubench_bridge | 0.9211+-0.0043 | 0.9254+-0.0110 | -0.0043 | 0.9322 | TopoGate 0.9235 |
| campbell__clubench_bridge | 0.2917+-0.1174 | 0.2860+-0.0265 | +0.0058 | 0.1958 | GCEALS 0.3139 |
| baron_human__clubench_bridge | 0.1790+-0.0212 | 0.1689+-0.0170 | +0.0102 | 0.2573 | NA |
| sms_spam_collection__shared_text | 0.8464+-0.0196 | 0.8511+-0.0141 | -0.0047 | 0.8343 | TopoGate 0.8606 |
| cnae9__shared_text | 0.3074+-0.0089 | 0.3146+-0.0452 | -0.0071 | 0.3786 | TopoGate 0.3303 |
| imdb__shared_text | 0.0295+-0.0201 | 0.0125+-0.0086 | +0.0171 | 0.0221 | NA |
| hate_speech__shared_text | 0.0195+-0.0097 | 0.0226+-0.0310 | -0.0031 | 0.0014 | NA |
| sentiment_labeld_sentences__shared_text | 0.0056+-0.0044 | 0.0028+-0.0034 | +0.0028 | 0.0020 | NA |

Overall ARI: RG 0.325049; matched scMAE 0.322975; paired delta +0.002074.
RG wins on 4/8 datasets by mean ARI.
