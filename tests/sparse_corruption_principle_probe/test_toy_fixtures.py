from scripts.sparse_corruption_principle_probe.toy_fixtures import audit_world_definitions, make_world


def test_toy_worlds_have_the_declared_roles():
    result = audit_world_definitions()
    assert result["status"] == "completed_valid"
    assert result["labels_used_for_fixture_audit_only"] is True
    assert result["fit_labels_allowed"] is False
    assert all(result["checks"].values())
    assert make_world("S").x.shape == (96, 24)
    assert make_world("V").labels_for_audit.shape == (96,)
