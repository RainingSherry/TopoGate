# V19 post-freeze comparison

- Final root: `result/V19/v19_rg_final_postfreeze_rel_both2_20260810`
- Archived baseline source: `result/baseline_comparison/summary.csv`
- V19 fit remains label-free; labels are used only for benchmark K and post-fit metrics.
- Selection status: proxy_supported final.
- Only `archived_sota_bridge_eligible` layers are joined to archived external baselines.
- Missing external rows remain missing; no zero imputation is performed.

## Dataset-level V19 means

| Dataset | Scope | Variant | ARI mean±std | NMI mean±std | ACC mean±std | n |
|---|---|---|---:|---:|---:|---:|
| baron_human__clubench_bridge | archived_sota_bridge_eligible | V19_rg_constant_gate | 0.2207±0.0415 | 0.4156±0.0408 | 0.4197±0.0399 | 3 |
| baron_human__clubench_bridge | archived_sota_bridge_eligible | V19_rg_default | 0.2611±0.1113 | 0.4486±0.1222 | 0.4335±0.1075 | 3 |
| baron_human__clubench_bridge | archived_sota_bridge_eligible | V19_rg_full | 0.2573±0.0449 | 0.4563±0.0522 | 0.4390±0.0367 | 3 |
| baron_human__clubench_bridge | archived_sota_bridge_eligible | V19_rg_nomix | 0.2387±0.0633 | 0.4354±0.0563 | 0.4397±0.0474 | 3 |
| baron_human__clubench_bridge | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.2170±0.0319 | 0.4042±0.0501 | 0.4119±0.0314 | 3 |
| baron_human__clubench_bridge | archived_sota_bridge_eligible | V19_scmae_only | 0.2387±0.0633 | 0.4354±0.0563 | 0.4397±0.0474 | 3 |
| baron_human__rg_native | internal_rg_native_only | V19_rg_constant_gate | 0.9009±0.0047 | 0.8691±0.0043 | 0.8844±0.0045 | 3 |
| baron_human__rg_native | internal_rg_native_only | V19_rg_default | 0.9031±0.0045 | 0.8699±0.0043 | 0.8877±0.0023 | 3 |
| baron_human__rg_native | internal_rg_native_only | V19_rg_full | 0.9031±0.0050 | 0.8702±0.0054 | 0.8871±0.0027 | 3 |
| baron_human__rg_native | internal_rg_native_only | V19_rg_nomix | 0.9007±0.0062 | 0.8688±0.0054 | 0.8854±0.0045 | 3 |
| baron_human__rg_native | internal_rg_native_only | V19_rg_reliability_off | 0.8998±0.0051 | 0.8668±0.0046 | 0.8839±0.0056 | 3 |
| baron_human__rg_native | internal_rg_native_only | V19_scmae_only | 0.9007±0.0062 | 0.8688±0.0054 | 0.8854±0.0045 | 3 |
| campbell__clubench_bridge | archived_sota_bridge_eligible | V19_rg_constant_gate | 0.2251±0.0365 | 0.3204±0.0054 | 0.3073±0.0168 | 3 |
| campbell__clubench_bridge | archived_sota_bridge_eligible | V19_rg_default | 0.1932±0.0256 | 0.3170±0.0130 | 0.2946±0.0118 | 3 |
| campbell__clubench_bridge | archived_sota_bridge_eligible | V19_rg_full | 0.1958±0.0292 | 0.3181±0.0009 | 0.2738±0.0215 | 3 |
| campbell__clubench_bridge | archived_sota_bridge_eligible | V19_rg_nomix | 0.2093±0.0119 | 0.3194±0.0027 | 0.2918±0.0104 | 3 |
| campbell__clubench_bridge | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.1846±0.0455 | 0.3193±0.0038 | 0.2760±0.0307 | 3 |
| campbell__clubench_bridge | archived_sota_bridge_eligible | V19_scmae_only | 0.2093±0.0119 | 0.3194±0.0027 | 0.2918±0.0104 | 3 |
| campbell__rg_native | internal_rg_native_only | V19_rg_constant_gate | 0.1917±0.0042 | 0.4649±0.0005 | 0.4049±0.0025 | 3 |
| campbell__rg_native | internal_rg_native_only | V19_rg_default | 0.1676±0.0319 | 0.4405±0.0302 | 0.3917±0.0233 | 3 |
| campbell__rg_native | internal_rg_native_only | V19_rg_full | 0.1687±0.0157 | 0.4332±0.0188 | 0.3842±0.0199 | 3 |
| campbell__rg_native | internal_rg_native_only | V19_rg_nomix | 0.1755±0.0120 | 0.4727±0.0221 | 0.4168±0.0159 | 3 |
| campbell__rg_native | internal_rg_native_only | V19_rg_reliability_off | 0.1588±0.0483 | 0.4330±0.0347 | 0.3796±0.0208 | 3 |
| campbell__rg_native | internal_rg_native_only | V19_scmae_only | 0.1755±0.0120 | 0.4727±0.0221 | 0.4168±0.0159 | 3 |
| cnae9__shared_text | archived_sota_bridge_eligible | V19_rg_constant_gate | 0.4417±0.0557 | 0.6481±0.0378 | 0.6460±0.0252 | 3 |
| cnae9__shared_text | archived_sota_bridge_eligible | V19_rg_default | 0.3630±0.0193 | 0.5992±0.0040 | 0.6065±0.0125 | 3 |
| cnae9__shared_text | archived_sota_bridge_eligible | V19_rg_full | 0.3786±0.0216 | 0.6107±0.0036 | 0.6108±0.0185 | 3 |
| cnae9__shared_text | archived_sota_bridge_eligible | V19_rg_nomix | 0.3648±0.0152 | 0.5867±0.0146 | 0.6068±0.0046 | 3 |
| cnae9__shared_text | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.3840±0.0338 | 0.6243±0.0147 | 0.6188±0.0195 | 3 |
| cnae9__shared_text | archived_sota_bridge_eligible | V19_scmae_only | 0.3648±0.0152 | 0.5867±0.0146 | 0.6068±0.0046 | 3 |
| hate_speech__shared_text | archived_sota_bridge_eligible | V19_rg_constant_gate | -0.0142±0.0098 | 0.0358±0.0095 | 0.4340±0.0535 | 3 |
| hate_speech__shared_text | archived_sota_bridge_eligible | V19_rg_default | 0.0041±0.0073 | 0.0250±0.0058 | 0.4909±0.0149 | 3 |
| hate_speech__shared_text | archived_sota_bridge_eligible | V19_rg_full | 0.0014±0.0085 | 0.0267±0.0026 | 0.4878±0.0142 | 3 |
| hate_speech__shared_text | archived_sota_bridge_eligible | V19_rg_nomix | -0.0126±0.0090 | 0.0498±0.0114 | 0.4705±0.0180 | 3 |
| hate_speech__shared_text | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.0070±0.0116 | 0.0267±0.0061 | 0.4956±0.0197 | 3 |
| hate_speech__shared_text | archived_sota_bridge_eligible | V19_scmae_only | -0.0126±0.0090 | 0.0498±0.0114 | 0.4705±0.0180 | 3 |
| imdb__shared_text | archived_sota_bridge_eligible | V19_rg_constant_gate | 0.0057±0.0094 | 0.0045±0.0070 | 0.5276±0.0335 | 3 |
| imdb__shared_text | archived_sota_bridge_eligible | V19_rg_default | 0.0196±0.0190 | 0.0152±0.0142 | 0.5655±0.0320 | 3 |
| imdb__shared_text | archived_sota_bridge_eligible | V19_rg_full | 0.0221±0.0148 | 0.0172±0.0112 | 0.5721±0.0242 | 3 |
| imdb__shared_text | archived_sota_bridge_eligible | V19_rg_nomix | 0.0289±0.0129 | 0.0216±0.0095 | 0.5840±0.0188 | 3 |
| imdb__shared_text | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.0234±0.0165 | 0.0181±0.0125 | 0.5741±0.0259 | 3 |
| imdb__shared_text | archived_sota_bridge_eligible | V19_scmae_only | 0.0289±0.0129 | 0.0216±0.0095 | 0.5840±0.0188 | 3 |
| mouse_retina__clubench_bridge | archived_sota_bridge_eligible | V19_rg_constant_gate | 0.9389±0.0058 | 0.8931±0.0064 | 0.9834±0.0014 | 3 |
| mouse_retina__clubench_bridge | archived_sota_bridge_eligible | V19_rg_default | 0.9356±0.0068 | 0.8866±0.0111 | 0.9825±0.0018 | 3 |
| mouse_retina__clubench_bridge | archived_sota_bridge_eligible | V19_rg_full | 0.9322±0.0046 | 0.8811±0.0090 | 0.9815±0.0014 | 3 |
| mouse_retina__clubench_bridge | archived_sota_bridge_eligible | V19_rg_nomix | 0.9310±0.0007 | 0.8830±0.0016 | 0.9814±0.0005 | 3 |
| mouse_retina__clubench_bridge | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.9239±0.0203 | 0.8701±0.0290 | 0.9792±0.0059 | 3 |
| mouse_retina__clubench_bridge | archived_sota_bridge_eligible | V19_scmae_only | 0.9310±0.0007 | 0.8830±0.0016 | 0.9814±0.0005 | 3 |
| mouse_retina__rg_native | internal_rg_native_only | V19_rg_constant_gate | 0.9451±0.0117 | 0.9072±0.0168 | 0.9849±0.0032 | 3 |
| mouse_retina__rg_native | internal_rg_native_only | V19_rg_default | 0.9405±0.0107 | 0.9023±0.0161 | 0.9837±0.0030 | 3 |
| mouse_retina__rg_native | internal_rg_native_only | V19_rg_full | 0.9420±0.0098 | 0.9046±0.0159 | 0.9841±0.0027 | 3 |
| mouse_retina__rg_native | internal_rg_native_only | V19_rg_nomix | 0.9552±0.0013 | 0.9233±0.0027 | 0.9876±0.0002 | 3 |
| mouse_retina__rg_native | internal_rg_native_only | V19_rg_reliability_off | 0.9420±0.0100 | 0.9023±0.0147 | 0.9840±0.0026 | 3 |
| mouse_retina__rg_native | internal_rg_native_only | V19_scmae_only | 0.9552±0.0013 | 0.9233±0.0027 | 0.9876±0.0002 | 3 |
| sentiment_labeld_sentences__shared_text | archived_sota_bridge_eligible | V19_rg_constant_gate | 0.0095±0.0064 | 0.0079±0.0049 | 0.5475±0.0167 | 3 |
| sentiment_labeld_sentences__shared_text | archived_sota_bridge_eligible | V19_rg_default | 0.0015±0.0010 | 0.0016±0.0007 | 0.5209±0.0054 | 3 |
| sentiment_labeld_sentences__shared_text | archived_sota_bridge_eligible | V19_rg_full | 0.0020±0.0011 | 0.0020±0.0008 | 0.5238±0.0061 | 3 |
| sentiment_labeld_sentences__shared_text | archived_sota_bridge_eligible | V19_rg_nomix | 0.0091±0.0027 | 0.0080±0.0019 | 0.5482±0.0074 | 3 |
| sentiment_labeld_sentences__shared_text | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.0018±0.0013 | 0.0018±0.0010 | 0.5224±0.0067 | 3 |
| sentiment_labeld_sentences__shared_text | archived_sota_bridge_eligible | V19_scmae_only | 0.0091±0.0027 | 0.0080±0.0019 | 0.5482±0.0074 | 3 |
| sms_spam_collection__shared_text | archived_sota_bridge_eligible | V19_rg_constant_gate | 0.8420±0.0240 | 0.7039±0.0398 | 0.9717±0.0048 | 3 |
| sms_spam_collection__shared_text | archived_sota_bridge_eligible | V19_rg_default | 0.8266±0.0110 | 0.6877±0.0250 | 0.9693±0.0025 | 3 |
| sms_spam_collection__shared_text | archived_sota_bridge_eligible | V19_rg_full | 0.8343±0.0053 | 0.6964±0.0057 | 0.9705±0.0007 | 3 |
| sms_spam_collection__shared_text | archived_sota_bridge_eligible | V19_rg_nomix | 0.8342±0.0112 | 0.6939±0.0225 | 0.9705±0.0025 | 3 |
| sms_spam_collection__shared_text | archived_sota_bridge_eligible | V19_rg_reliability_off | 0.8329±0.0159 | 0.6888±0.0261 | 0.9701±0.0032 | 3 |
| sms_spam_collection__shared_text | archived_sota_bridge_eligible | V19_scmae_only | 0.8342±0.0112 | 0.6939±0.0225 | 0.9705±0.0025 | 3 |

## External reference rows

See `comparison.csv` for the long-form table. External values retain their archived provenance and are not interpreted as fresh matched SOTA runs.
