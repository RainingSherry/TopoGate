# CLUBench: AHDPC vs HDPC vs V9

## Protocol

- Input: CLUBench `load_data` column-wise z-score.
- `K = int(np.unique(y).size)` is used only for benchmark K and post-fit metrics.
- AHDPC/HDPC: fixed epsilon=1.0, `paper_semantic` normalization, `table_reproduction` adaptive-distance rule.
- V9: `learnable_gate_v9_adaptive`, seed=42, 80 epochs, already-standardized input with `scale_input=false`.

- Dataset records present: **131**; complete three-method datasets: **131**.

## Method-level aggregate metrics

| Method | Completed | Errors | Mean ARI | Median ARI | Mean NMI | Mean ACC |
|---|---:|---:|---:|---:|---:|---:|
| AHDPC | 131 | 0 | 0.1830 | 0.0320 | 0.2401 | 0.5305 |
| HDPC | 131 | 0 | 0.1614 | 0.0104 | 0.2200 | 0.5165 |
| V9 | 131 | 0 | 0.3227 | 0.2484 | 0.3757 | 0.6059 |

## V9 vs AHDPC

- Valid paired datasets: **131**; ARI wins/ties/losses: **105/2/24**; mean ΔARI: **0.1396**.

| Dataset | V9 ARI | Opponent ARI | ΔARI | ΔNMI | ΔACC |
|---|---:|---:|---:|---:|---:|
| Mouse_retina | 0.9304 | -0.0008 | 0.9312 | 0.8762 | 0.1774 |
| enron | 0.8972 | -0.0000 | 0.8972 | 0.8482 | 0.4713 |
| sms_spam_collection | 0.8521 | 0.0637 | 0.7884 | 0.6612 | 0.1018 |
| predicting_pulsar_star | 0.7576 | -0.0002 | 0.7578 | 0.6067 | 0.0605 |
| htru2 | 0.7148 | -0.0002 | 0.7150 | 0.5768 | 0.0561 |
| breast_cancer_wisconsin_prognostic | 0.6882 | -0.0165 | 0.7047 | 0.5494 | 0.3111 |
| spambase | 0.5843 | -0.0002 | 0.5844 | 0.4670 | 0.2767 |
| dry_bean | 0.6998 | 0.1250 | 0.5748 | 0.4891 | 0.5219 |
| CIFAR10_CLIP | 0.5546 | 0.0000 | 0.5546 | 0.6961 | 0.6322 |
| breast_cancer_wisconsin_original | 0.8526 | 0.3263 | 0.5263 | 0.4162 | 0.1757 |
| smoker_condition | 0.9299 | 0.9685 | -0.0386 | -0.0594 | -0.0099 |
| parkinsons | -0.0965 | -0.0171 | -0.0794 | -0.0821 | 0.0051 |
| image_segmentation | 0.4143 | 0.4973 | -0.0830 | -0.0128 | -0.0762 |
| echocardiogram | 0.3521 | 0.4528 | -0.1008 | -0.0526 | -0.0492 |
| world12d | 0.6827 | 0.7923 | -0.1096 | -0.0719 | -0.1400 |
| paris_housing_classification | -0.0001 | 0.1199 | -0.1200 | -0.0317 | -0.2731 |
| extyaleb | 0.3702 | 0.5281 | -0.1579 | -0.0786 | 0.0219 |
| heart_disease | 0.1288 | 0.3325 | -0.2037 | 0.0001 | -0.2121 |
| shuttle | 0.1287 | 0.5891 | -0.4604 | -0.4410 | -0.1980 |
| banknote_authentication | 0.0243 | 0.9624 | -0.9381 | -0.9084 | -0.4111 |

## V9 vs HDPC

- Valid paired datasets: **131**; ARI wins/ties/losses: **104/1/26**; mean ΔARI: **0.1613**.

| Dataset | V9 ARI | Opponent ARI | ΔARI | ΔNMI | ΔACC |
|---|---:|---:|---:|---:|---:|
| Mouse_retina | 0.9304 | -0.0008 | 0.9312 | 0.8762 | 0.1774 |
| enron | 0.8972 | 0.0000 | 0.8972 | 0.8482 | 0.4707 |
| breast_cancer_wisconsin_original | 0.8526 | -0.0361 | 0.8887 | 0.6234 | 0.4334 |
| sms_spam_collection | 0.8521 | -0.0020 | 0.8542 | 0.7167 | 0.1090 |
| predicting_pulsar_star | 0.7576 | -0.0002 | 0.7578 | 0.6067 | 0.0605 |
| htru2 | 0.7148 | -0.0002 | 0.7150 | 0.5768 | 0.0561 |
| breast_cancer_wisconsin_prognostic | 0.6882 | -0.0208 | 0.7090 | 0.5514 | 0.3199 |
| rice_dataset_cammeo_and_osmancik | 0.6894 | -0.0064 | 0.6958 | 0.5572 | 0.3575 |
| dry_bean | 0.6998 | 0.0074 | 0.6924 | 0.7136 | 0.5917 |
| spambase | 0.5843 | -0.0002 | 0.5844 | 0.4670 | 0.2767 |
| statlog_image_segmentation | 0.4715 | 0.5150 | -0.0435 | -0.0466 | -0.0095 |
| blood_transfusion_service_center | 0.0168 | 0.0636 | -0.0467 | -0.0547 | -0.0602 |
| image_segmentation | 0.4143 | 0.4716 | -0.0573 | -0.0203 | -0.0571 |
| extyaleb | 0.3702 | 0.4405 | -0.0703 | -0.0256 | 0.0658 |
| parkinsons | -0.0965 | -0.0121 | -0.0844 | -0.0854 | 0.0000 |
| world12d | 0.6827 | 0.7896 | -0.1068 | -0.0678 | -0.1333 |
| paris_housing_classification | -0.0001 | 0.1379 | -0.1380 | -0.0385 | -0.2873 |
| heart_disease | 0.1288 | 0.2944 | -0.1656 | -0.0089 | -0.1616 |
| shuttle | 0.1287 | 0.3478 | -0.2190 | -0.1259 | -0.1377 |
| banknote_authentication | 0.0243 | 0.6239 | -0.5996 | -0.5860 | -0.3156 |

## Per-dataset status

| Dataset | AHDPC | HDPC | V9 |
|---|---|---|---|
| 20newsgroups | completed | completed | completed |
| Baron Human | completed | completed | completed |
| CIFAR10_CLIP | completed | completed | completed |
| COIL20_CLIP | completed | completed | completed |
| Campbell | completed | completed | completed |
| Drug Consumption | completed | completed | completed |
| FashionMNIST_CLIP | completed | completed | completed |
| ISOLET | completed | completed | completed |
| Indian_pines | completed | completed | completed |
| JapaneseVowels | completed | completed | completed |
| MNIST_CLIP | completed | completed | completed |
| Mouse_retina | completed | completed | completed |
| PCam | completed | completed | completed |
| Waveform | completed | completed | completed |
| banknote_authentication | completed | completed | completed |
| birds_bones_and_living_habits | completed | completed | completed |
| blood_transfusion_service_center | completed | completed | completed |
| boston | completed | completed | completed |
| breast_cancer_coimbra | completed | completed | completed |
| breast_cancer_wisconsin_original | completed | completed | completed |
| breast_cancer_wisconsin_prognostic | completed | completed | completed |
| breast_tissue | completed | completed | completed |
| cardiovascular_study | completed | completed | completed |
| cifar10 | completed | completed | completed |
| classification_in_asteroseismology | completed | completed | completed |
| cnae9 | completed | completed | completed |
| coil20 | completed | completed | completed |
| credit_risk_classification | completed | completed | completed |
| crowdsourced_mapping | completed | completed | completed |
| customer_classification | completed | completed | completed |
| date_fruit | completed | completed | completed |
| dermatology | completed | completed | completed |
| diabetic_retinopathy_debrecen | completed | completed | completed |
| dilbert | completed | completed | completed |
| dry_bean | completed | completed | completed |
| durum_wheat_features | completed | completed | completed |
| echocardiogram | completed | completed | completed |
| ecoli | completed | completed | completed |
| enron | completed | completed | completed |
| epileptic_seizure_recognition | completed | completed | completed |
| extyaleb | completed | completed | completed |
| fabert | completed | completed | completed |
| fashion_mnist | completed | completed | completed |
| fbis.wc | completed | completed | completed |
| fetal_health_classification | completed | completed | completed |
| first-order-theorem-proving | completed | completed | completed |
| flickr_material_database | completed | completed | completed |
| fraud_detection_bank | completed | completed | completed |
| gas-drift | completed | completed | completed |
| gina_prior2 | completed | completed | completed |
| glass_identification | completed | completed | completed |
| har | completed | completed | completed |
| harbermans_survival | completed | completed | completed |
| hate_speech | completed | completed | completed |
| heart_attack_analysis_prediction_dataset | completed | completed | completed |
| heart_disease | completed | completed | completed |
| hepatitis | completed | completed | completed |
| htru2 | completed | completed | completed |
| human_stress_detection | completed | completed | completed |
| image_segmentation | completed | completed | completed |
| imdb | completed | completed | completed |
| insurance_company_benchmark | completed | completed | completed |
| ionosphere | completed | completed | completed |
| iris | completed | completed | completed |
| labeled_faces_in_the_wild | completed | completed | completed |
| letter_recognition | completed | completed | completed |
| magic_gamma_telescope | completed | completed | completed |
| mammographic_mass | completed | completed | completed |
| mfeat-factors | completed | completed | completed |
| mfeat-fourier | completed | completed | completed |
| mfeat-karhunen | completed | completed | completed |
| mfeat-morphological | completed | completed | completed |
| micro-mass | completed | completed | completed |
| microbes | completed | completed | completed |
| mnist64 | completed | completed | completed |
| mobile_price_classification | completed | completed | completed |
| music_genre_classification | completed | completed | completed |
| olivetti_faces | completed | completed | completed |
| optical_recognition_of_handwritten_digits | completed | completed | completed |
| orbit_classification_for_prediction_nasa | completed | completed | completed |
| paris_housing_classification | completed | completed | completed |
| parkinsons | completed | completed | completed |
| patient_treatment_classification | completed | completed | completed |
| pen_based_recognition_of_handwritten_digits | completed | completed | completed |
| ph_recognition | completed | completed | completed |
| pima_indians_diabetes_database | completed | completed | completed |
| pistachio | completed | completed | completed |
| planning_relax | completed | completed | completed |
| poker-hand | completed | completed | completed |
| predicting_pulsar_star | completed | completed | completed |
| pumpkin_seeds | completed | completed | completed |
| raisin | completed | completed | completed |
| reuters | completed | completed | completed |
| rice_dataset_cammeo_and_osmancik | completed | completed | completed |
| rice_seed_gonen_jasmine | completed | completed | completed |
| rmftsa_sleepdata | completed | completed | completed |
| satellite_image | completed | completed | completed |
| secom | completed | completed | completed |
| seeds | completed | completed | completed |
| seismic_bumps | completed | completed | completed |
| sentiment_labeld_sentences | completed | completed | completed |
| shuttle | completed | completed | completed |
| siberian_weather_stats | completed | completed | completed |
| skillcraft1_master_table_dataset | completed | completed | completed |
| smoker_condition | completed | completed | completed |
| sms_spam_collection | completed | completed | completed |
| spambase | completed | completed | completed |
| spectf_heart | completed | completed | completed |
| statlog_german_credit | completed | completed | completed |
| statlog_image_segmentation | completed | completed | completed |
| steel-plates-fault | completed | completed | completed |
| street_view_house_numbers | completed | completed | completed |
| student_grade | completed | completed | completed |
| synthetic_control | completed | completed | completed |
| tamilnadu-electricity | completed | completed | completed |
| tr45.wc | completed | completed | completed |
| turkish_music_emotion | completed | completed | completed |
| user_knowledge_modeling | completed | completed | completed |
| vehicle | completed | completed | completed |
| wall-robot-navigation | completed | completed | completed |
| water_quality | completed | completed | completed |
| weather | completed | completed | completed |
| website_phishing | completed | completed | completed |
| wine | completed | completed | completed |
| wine_customer | completed | completed | completed |
| wine_quality | completed | completed | completed |
| wireless_indoor_localization | completed | completed | completed |
| world12d | completed | completed | completed |
| wos | completed | completed | completed |
| yeast | completed | completed | completed |
| zoo | completed | completed | completed |
