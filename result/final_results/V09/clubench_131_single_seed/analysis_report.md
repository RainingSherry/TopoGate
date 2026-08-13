# CLUBench AHDPC vs HDPC vs V9: paired analysis
## Scope and protocol

This report uses only complete three-method records in `comparison_long.csv`. ARI is the primary comparison metric; ACC, NMI, AMI, RI and FMI are retained in `analysis_by_dataset.csv`. The run is single-seed (42), so the tables are engineering evidence and require multi-seed confirmation before a paper-level claim.

Complete triplets: **131**.

## Aggregate metrics

| Method | Mean ARI | Median ARI | Mean NMI | Mean ACC |
|---|---:|---:|---:|---:|
| AHDPC | 0.1830 | 0.0320 | 0.2401 | 0.5305 |
| HDPC | 0.1614 | 0.0104 | 0.2200 | 0.5165 |
| V9 | 0.3227 | 0.2484 | 0.3757 | 0.6059 |

## Paired V9 comparison by ARI

| Opponent | n | V9 wins | ties | V9 losses | Mean ΔARI | Median ΔARI |
|---|---:|---:|---:|---:|---:|---:|
| AHDPC | 131 | 105 | 2 | 24 | 0.1396 | 0.0573 |
| HDPC | 131 | 104 | 1 | 26 | 0.1613 | 0.0851 |

## V9 advantages over AHDPC

Descriptive threshold: ΔARI ≥ 0.10 (58 datasets); strong V9 outcome additionally requires V9 ARI ≥ 0.50 (26 datasets).

| Dataset | n | d | K | V9 ARI | AHDPC ARI | HDPC ARI | ΔARI | ΔNMI | ΔACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mouse_retina | 8352 | 6198 | 5 | 0.9304 | -0.0008 | -0.0008 | 0.9312 | 0.8762 | 0.1774 |
| enron | 9999 | 4096 | 2 | 0.8972 | -0.0000 | 0.0000 | 0.8972 | 0.8482 | 0.4713 |
| sms_spam_collection | 835 | 500 | 2 | 0.8521 | 0.0637 | -0.0020 | 0.7884 | 0.6612 | 0.1018 |
| predicting_pulsar_star | 9273 | 8 | 2 | 0.7576 | -0.0002 | -0.0002 | 0.7578 | 0.6067 | 0.0605 |
| htru2 | 9999 | 8 | 2 | 0.7148 | -0.0002 | -0.0002 | 0.7150 | 0.5768 | 0.0561 |
| breast_cancer_wisconsin_prognostic | 569 | 30 | 2 | 0.6882 | -0.0165 | -0.0208 | 0.7047 | 0.5494 | 0.3111 |
| spambase | 4601 | 57 | 2 | 0.5843 | -0.0002 | -0.0002 | 0.5844 | 0.4670 | 0.2767 |
| dry_bean | 9997 | 16 | 7 | 0.6998 | 0.1250 | 0.0074 | 0.5748 | 0.4891 | 0.5219 |
| CIFAR10_CLIP | 10000 | 512 | 10 | 0.5546 | 0.0000 | 0.0000 | 0.5546 | 0.6961 | 0.6322 |
| breast_cancer_wisconsin_original | 683 | 9 | 2 | 0.8526 | 0.3263 | -0.0361 | 0.5263 | 0.4162 | 0.1757 |
| ISOLET | 7797 | 617 | 26 | 0.5257 | 0.0321 | 0.0483 | 0.4936 | 0.5030 | 0.5255 |
| wine | 178 | 13 | 3 | 0.9122 | 0.4265 | 0.4412 | 0.4856 | 0.3263 | 0.3034 |
| wine_customer | 178 | 13 | 3 | 0.9122 | 0.4265 | 0.4412 | 0.4856 | 0.3263 | 0.3034 |
| ecoli | 327 | 7 | 5 | 0.7439 | 0.3054 | 0.2805 | 0.4385 | 0.2533 | 0.3242 |
| micro-mass | 360 | 1300 | 10 | 0.4857 | 0.0522 | 0.0518 | 0.4335 | 0.4252 | 0.3167 |
| zoo | 101 | 16 | 7 | 0.7933 | 0.4082 | 0.6338 | 0.3851 | 0.1436 | 0.2772 |
| wireless_indoor_localization | 2000 | 7 | 4 | 0.7688 | 0.3850 | 0.4238 | 0.3838 | 0.2136 | 0.2960 |
| Indian_pines | 8858 | 220 | 5 | 0.3130 | -0.0638 | -0.0455 | 0.3769 | 0.4039 | 0.2261 |
| fashion_mnist | 3000 | 784 | 10 | 0.3728 | 0.0001 | 0.0005 | 0.3727 | 0.5232 | 0.4423 |
| heart_attack_analysis_prediction_dataset | 303 | 13 | 2 | 0.3706 | 0.0028 | 0.0100 | 0.3678 | 0.2845 | 0.2541 |

## Negative datasets relative to AHDPC

Descriptive threshold: ΔARI ≤ −0.10 (7 datasets). The first table isolates cases where AHDPC itself is strong (ARI ≥ 0.50; 4 datasets); the second contains all substantial regressions.

### AHDPC-strong regressions

| Dataset | n | d | K | V9 ARI | AHDPC ARI | HDPC ARI | ΔARI | ΔNMI | ΔACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| banknote_authentication | 1372 | 4 | 2 | 0.0243 | 0.9624 | 0.6239 | -0.9381 | -0.9084 | -0.4111 |
| shuttle | 10000 | 9 | 2 | 0.1287 | 0.5891 | 0.3478 | -0.4604 | -0.4410 | -0.1980 |
| extyaleb | 319 | 30 | 5 | 0.3702 | 0.5281 | 0.4405 | -0.1579 | -0.0786 | 0.0219 |
| world12d | 150 | 12 | 5 | 0.6827 | 0.7923 | 0.7896 | -0.1096 | -0.0719 | -0.1400 |

### All substantial regressions

| Dataset | n | d | K | V9 ARI | AHDPC ARI | HDPC ARI | ΔARI | ΔNMI | ΔACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| banknote_authentication | 1372 | 4 | 2 | 0.0243 | 0.9624 | 0.6239 | -0.9381 | -0.9084 | -0.4111 |
| shuttle | 10000 | 9 | 2 | 0.1287 | 0.5891 | 0.3478 | -0.4604 | -0.4410 | -0.1980 |
| heart_disease | 297 | 13 | 5 | 0.1288 | 0.3325 | 0.2944 | -0.2037 | 0.0001 | -0.2121 |
| extyaleb | 319 | 30 | 5 | 0.3702 | 0.5281 | 0.4405 | -0.1579 | -0.0786 | 0.0219 |
| paris_housing_classification | 10000 | 17 | 2 | -0.0001 | 0.1199 | 0.1379 | -0.1200 | -0.0317 | -0.2731 |
| world12d | 150 | 12 | 5 | 0.6827 | 0.7923 | 0.7896 | -0.1096 | -0.0719 | -0.1400 |
| echocardiogram | 61 | 10 | 2 | 0.3521 | 0.4528 | 0.1471 | -0.1008 | -0.0526 | -0.0492 |

## Shared-difficulty datasets

All three methods have ARI ≤ 0.10 (43 datasets). These are not V9-specific failures and should not be counted as evidence against the gate.

| Dataset | n | d | K | V9 ARI | AHDPC ARI | HDPC ARI | ΔARI | ΔNMI | ΔACC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crowdsourced_mapping | 9997 | 28 | 4 | 0.0926 | 0.0353 | 0.0612 | 0.0573 | 0.1924 | -0.1485 |
| skillcraft1_master_table_dataset | 3303 | 18 | 6 | 0.0848 | 0.0064 | -0.0003 | 0.0784 | 0.1429 | 0.0778 |
| 20newsgroups | 9991 | 4096 | 20 | 0.0693 | 0.0000 | 0.0000 | 0.0693 | 0.1790 | 0.1434 |
| vehicle | 846 | 18 | 4 | 0.0658 | 0.0581 | 0.0177 | 0.0077 | -0.0286 | 0.0248 |
| wall-robot-navigation | 5456 | 24 | 4 | 0.0551 | 0.0357 | 0.0474 | 0.0195 | 0.0017 | 0.0528 |
| imdb | 3250 | 700 | 2 | 0.0477 | 0.0000 | 0.0000 | 0.0477 | 0.0352 | 0.1092 |
| PCam | 4000 | 27648 | 2 | 0.0465 | 0.0000 | 0.0000 | 0.0464 | 0.0656 | 0.1020 |
| microbes | 9995 | 24 | 10 | 0.0438 | 0.0005 | -0.0028 | 0.0433 | 0.0807 | 0.0135 |
| wine_quality | 4873 | 11 | 5 | 0.0421 | 0.0320 | 0.0266 | 0.0102 | 0.0160 | -0.0542 |
| cifar10 | 3250 | 1024 | 10 | 0.0403 | 0.0000 | 0.0000 | 0.0403 | 0.0733 | 0.1218 |
| insurance_company_benchmark | 5822 | 85 | 2 | 0.0379 | 0.0104 | 0.0104 | 0.0275 | 0.0157 | -0.2497 |
| rmftsa_sleepdata | 1024 | 2 | 4 | 0.0365 | 0.0383 | 0.0143 | -0.0017 | -0.0115 | -0.0254 |
| first-order-theorem-proving | 6118 | 51 | 6 | 0.0322 | 0.0131 | 0.0046 | 0.0190 | 0.0046 | -0.0051 |
| epileptic_seizure_recognition | 5750 | 178 | 5 | 0.0273 | 0.0000 | 0.0000 | 0.0273 | 0.1666 | 0.0369 |
| cardiovascular_study | 2927 | 15 | 2 | 0.0248 | 0.0006 | -0.0120 | 0.0243 | 0.0213 | 0.0697 |
| sentiment_labeld_sentences | 2748 | 200 | 2 | 0.0215 | -0.0000 | -0.0000 | 0.0215 | 0.0175 | 0.0695 |
| patient_treatment_classification | 4412 | 10 | 2 | 0.0196 | 0.0015 | 0.0014 | 0.0181 | 0.0171 | 0.0499 |
| siberian_weather_stats | 1407 | 11 | 7 | 0.0194 | 0.0100 | 0.0142 | 0.0094 | 0.0331 | -0.0505 |
| hate_speech | 3221 | 100 | 3 | 0.0187 | -0.0046 | -0.0150 | 0.0234 | 0.0529 | -0.3319 |
| magic_gamma_telescope | 9999 | 10 | 2 | 0.0176 | -0.0002 | -0.0002 | 0.0178 | 0.0054 | -0.0734 |
| blood_transfusion_service_center | 748 | 4 | 2 | 0.0168 | 0.0448 | 0.0636 | -0.0280 | -0.0524 | -0.0414 |
| seismic_bumps | 646 | 24 | 2 | 0.0168 | -0.0189 | -0.0189 | 0.0357 | 0.0298 | -0.0882 |
| harbermans_survival | 306 | 3 | 2 | 0.0161 | 0.0149 | 0.0078 | 0.0012 | 0.0085 | -0.0784 |
| customer_classification | 1000 | 11 | 4 | 0.0126 | 0.0046 | 0.0045 | 0.0081 | 0.0097 | 0.0280 |
| orbit_classification_for_prediction_nasa | 1722 | 11 | 3 | 0.0118 | 0.0063 | 0.0029 | 0.0055 | 0.0879 | 0.0029 |
| Drug Consumption | 1749 | 12 | 4 | 0.0094 | 0.0142 | 0.0143 | -0.0048 | -0.0015 | -0.0663 |
| statlog_german_credit | 1000 | 24 | 2 | 0.0094 | 0.0014 | 0.0450 | 0.0080 | 0.0017 | -0.1220 |
| tr45.wc | 676 | 8261 | 9 | 0.0069 | 0.0021 | -0.0023 | 0.0048 | 0.0888 | 0.0030 |
| mobile_price_classification | 2000 | 20 | 4 | 0.0068 | -0.0005 | -0.0004 | 0.0073 | 0.0092 | 0.0360 |
| diabetic_retinopathy_debrecen | 1151 | 19 | 2 | 0.0059 | 0.0025 | 0.0023 | 0.0034 | 0.0047 | 0.0104 |
| fraud_detection_bank | 9999 | 112 | 2 | 0.0049 | 0.0434 | -0.0504 | -0.0386 | 0.0610 | -0.1198 |
| fabert | 8237 | 800 | 7 | 0.0041 | 0.0001 | 0.0020 | 0.0040 | 0.0100 | -0.0426 |
| labeled_faces_in_the_wild | 2200 | 5828 | 2 | 0.0020 | 0.0000 | 0.0000 | 0.0020 | 0.0009 | 0.0245 |
| poker-hand | 10000 | 10 | 2 | -0.0000 | 0.0000 | 0.0001 | -0.0001 | 0.0001 | -0.0345 |
| tamilnadu-electricity | 10000 | 2 | 20 | -0.0001 | -0.0001 | 0.0003 | -0.0000 | -0.0006 | -0.0087 |
| street_view_house_numbers | 732 | 1024 | 10 | -0.0001 | 0.0013 | 0.0006 | -0.0014 | 0.0093 | -0.0464 |
| water_quality | 2011 | 9 | 2 | -0.0007 | 0.0060 | 0.0072 | -0.0067 | -0.0024 | -0.0985 |
| breast_cancer_coimbra | 116 | 9 | 2 | -0.0013 | 0.0057 | 0.0057 | -0.0070 | -0.0131 | -0.0172 |
| secom | 1567 | 590 | 2 | -0.0034 | 0.0102 | 0.0219 | -0.0136 | 0.0042 | -0.4180 |
| planning_relax | 182 | 12 | 2 | -0.0046 | -0.0046 | 0.0010 | 0.0000 | 0.0005 | 0.0055 |
| credit_risk_classification | 976 | 11 | 2 | -0.0184 | -0.0199 | -0.0197 | 0.0015 | 0.0005 | -0.0195 |
| steel-plates-fault | 1941 | 33 | 2 | -0.0399 | -0.0440 | -0.0442 | 0.0041 | -0.0124 | 0.0057 |
| parkinsons | 195 | 22 | 2 | -0.0965 | -0.0171 | -0.0121 | -0.0794 | -0.0821 | 0.0051 |

## Interpretation boundary

The deltas describe one fixed protocol and one seed. They identify where V9 helps or regresses relative to the frozen AHDPC/HDPC implementations; they do not establish significance, robustness, or a universally superior method.
