from scripts.relation_selection_probe.relation_features import (
    DATASETS,
    FEATURE_FAMILIES,
    PRIMARY_DATASETS,
    VIEW_DIM,
    VIEW_SEEDS,
)


def test_rs0_roles_and_feature_families_are_frozen():
    assert PRIMARY_DATASETS == ("cnae9", "Campbell", "sms_spam_collection")
    assert len(DATASETS) == 6
    assert len(VIEW_SEEDS) == 8
    assert VIEW_DIM == 96
    assert set(FEATURE_FAMILIES) == {"G", "T", "S", "G+T", "G+S", "T+S", "G+T+S"}
