from scripts.sparse_corruption_principle_probe import protocol


def test_c0_contract_and_explicit_c2_authorization():
    protocol.validate_contract()
    protocol.validate_c2_authorization()
    assert protocol.PROJECT_ID == "sparse_corruption_principle_probe"
    assert protocol.DEVELOPMENT_PANEL == ("Mouse_retina", "Baron Human", "Campbell")
    assert protocol.PRINCIPLES == (
        "P0_Random",
        "P1_SupportPreserve",
        "P2_SupportTarget",
        "P3_FrequencyAware",
        "P4_ResidualHard",
        "P5_GeometryHard",
    )
    assert protocol.LEGAL_GPU_POOL == (1, 2, 3, 4, 5, 6)
    assert set(protocol.LEGAL_GPU_POOL).isdisjoint(protocol.FORBIDDEN_GPU_IDS)
    config = protocol.resolved_config()
    assert config["authorized_now"][-1] == "C2_54_run_matrix"
    assert config["locked_now"] == ["C3_holdout_runs", "adaptive_policy", "GAN", "learned_generator"]
    assert config["c2_matrix_authorized"] is True
    assert "dense H0" in config["support_interpretation_firewall"]
    assert config["label_firewall"]["forbidden_fit_inputs"] == ["y", "ARI", "NMI", "ACC", "cluster_purity"]
    assert config["formal_matrix"]["runs"] == 54
