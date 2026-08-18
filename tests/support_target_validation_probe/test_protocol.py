from scripts.support_target_validation_probe import protocol


def test_m0_m1_contract_is_independent_and_locked():
    protocol.validate_contract()
    assert protocol.PROJECT_ID == "support_target_validation_probe"
    assert protocol.M1_CONTROL == "P2_MM_SupportPreserve"
    assert protocol.DEVELOPMENT_PANEL == ("Mouse_retina", "Baron Human", "Campbell")
    assert protocol.PRIMARY_SEEDS == (42, 123, 7)
    assert protocol.FORBIDDEN_GPU_IDS == (0, 7)
    config = protocol.resolved_config()
    assert config["old_project_read_only"] == "sparse_corruption_principle_probe"
    assert set(config["locked_stages"]) >= {"M2_raw_x_bridge", "M3_holdout", "M4_full_backbone", "GAN"}
    assert config["labels"]["forbidden_fit_inputs"] == ["y", "ARI", "NMI", "ACC", "cluster_purity"]
