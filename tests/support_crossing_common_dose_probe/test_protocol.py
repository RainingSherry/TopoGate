from scripts.support_crossing_common_dose_probe import protocol


def test_d0_d1_contract_is_frozen_and_d2_locked():
    protocol.validate_contract()
    config = protocol.resolved_config()
    assert config["project_id"] == "support_crossing_common_dose_probe"
    assert config["d1_gate"]["minimum_common_positive_budget_row_fraction"] == 0.95
    assert config["d1_gate"]["dataset_total_relative_mismatch"] == 0.05
    assert config["d1_gate"]["median_row_relative_mismatch"] == 0.10
    assert config["d2_gpu_runs_started"] == 0
    assert config["locked_stages"] == ["D2_gpu_matrix", "raw_x_bridge", "holdout", "adaptive_policy", "GAN"]
    assert config["label_firewall"]["labels_loaded"] is False
