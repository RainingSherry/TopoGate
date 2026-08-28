from scripts.adaptive_corruption_probe import protocol


def test_track_b_roles_and_corruption_library_are_frozen():
    protocol.validate_contract()
    assert protocol.PROJECT_ID == "adaptive_corruption_probe"
    assert protocol.BASE_COMMIT == "c80877cf904e41950315d37b95374825c33a7362"
    assert len(protocol.DEVELOPMENT_PANEL) == 6
    assert protocol.CORRUPTION_ARMS == (
        "C_clean_no_corruption",
        "C0_MatchedRandom",
        "C1_ValueOnly",
        "C2_SupportOnly",
        "C3_MixedMatched",
        "C4_StaticHard",
    )
    assert set(protocol.LEGAL_GPU_POOL).isdisjoint(protocol.FORBIDDEN_GPU_IDS)
    assert protocol.STRUCTURED_ARMS == (
        "C1_ValueOnly",
        "C2_SupportOnly",
        "C3_MixedMatched",
        "C4_StaticHard",
    )
    assert protocol.BACKBONE_CONFIG["input"] == "audited_S0_H0"
    assert protocol.BACKBONE_CONFIG["encoder_dims"] == ["d_eff", 64, 32]
    assert protocol.H0_SUPPORT_THRESHOLD_RATIO == 0.05
    assert protocol.CORRUPTION_RATE == 0.25
    assert protocol.SIMPLE_MIN_DATASET_COUNT == 2
    assert "pair_budget_rule" in protocol.BACKBONE_CONFIG


def test_track_b_fit_is_label_free_and_holdout_is_not_outcome_selected():
    config = protocol.resolved_config()
    assert config["labels_used_during_fit"] is False
    assert config["holdout_status"].startswith("not_yet_selected")
    assert config["positive_control"]["must_pass_before_real_null_decision"] is True
    assert config["cross_track_holdout_disjointness_required"] is True
    assert config["initial_stage"] == "B1"
    hierarchy = config["decision_hierarchy"]
    assert hierarchy["level_1_corruption_matters"]["primary_contrast"].startswith("ARI(C0")
    assert hierarchy["level_2_structured_beats_random"]["contrast"].startswith("Delta_random")
    assert hierarchy["level_3_adaptation_necessary"]["terminal_if_not_met"] == "random_corruption_sufficient"
