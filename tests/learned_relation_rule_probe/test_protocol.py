from scripts.learned_relation_rule_probe import protocol


def test_track_a_is_independent_and_gpu_contract_is_legal():
    protocol.validate_contract()
    assert protocol.PROJECT_ID == "learned_relation_rule_probe"
    assert protocol.BASE_COMMIT == "c80877cf904e41950315d37b95374825c33a7362"
    assert set(protocol.LEGAL_GPU_POOL).isdisjoint(protocol.FORBIDDEN_GPU_IDS)
    assert protocol.DEVELOPMENT_DATASETS == ("cnae9", "Campbell", "sms_spam_collection")
    assert len(protocol.SENTINEL_DATASETS) == 3


def test_track_a_configuration_never_allows_labels_in_fit():
    config = protocol.resolved_config()
    assert config["labels_used_during_fit"] is False
    assert config["diagnostic_supervision"]["deployable_method"] is False
    assert config["authorized_initial_stage"] == "A1"
