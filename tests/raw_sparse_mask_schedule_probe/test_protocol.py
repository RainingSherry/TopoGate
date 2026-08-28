from scripts.raw_sparse_mask_schedule_probe import protocol


def test_frozen_contract_is_valid():
    protocol.validate_contract()
    config = protocol.resolved_config()
    assert config["formal_matrix"]["main_runs"] == 90
    assert config["legal_gpu_pool"] == [1, 2, 3, 4, 5, 6]
    assert 0 not in config["legal_gpu_pool"]
    assert 7 not in config["legal_gpu_pool"]

