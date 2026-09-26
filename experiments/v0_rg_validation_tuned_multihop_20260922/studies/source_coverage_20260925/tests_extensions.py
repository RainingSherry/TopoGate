"""Minimal invariant tests for probe_directions / extension_audit (CPU, synthetic, no labels).

Run: python -m pytest -q papers/source_coverage_20260925/tests_extensions.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_directions as probe  # noqa: E402
import extension_audit as audit  # noqa: E402

torch = pytest.importorskip("torch")


class TinyMAE(torch.nn.Module):
    """Same forward_mask contract as the scMAE AutoEncoder, with a dropout layer."""

    def __init__(self, genes=6, hidden=3):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(genes, hidden))
        self.mask_predictor = torch.nn.Linear(hidden, genes)
        self.decoder = torch.nn.Linear(hidden + genes, genes)

    def forward_mask(self, x):
        latent = self.encoder(x)
        logits = self.mask_predictor(latent)
        return latent, logits, self.decoder(torch.cat([latent, logits], dim=1))


def test_global_scale_is_translation_invariant_and_homogeneous():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 4))
    base = probe.global_embedding_scale(z)
    assert probe.global_embedding_scale(z + 7.0) == pytest.approx(base)
    assert probe.global_embedding_scale(3.0 * z) == pytest.approx(3.0 * base)
    with pytest.raises(ValueError):
        probe.global_embedding_scale(np.ones((5, 4)))


def test_normalized_response_is_invariant_to_latent_rescaling():
    torch.manual_seed(0)
    model = TinyMAE().eval()
    x = np.random.default_rng(1).standard_normal((20, 6)).astype(np.float32)
    d = np.random.default_rng(2).standard_normal((20, 6)).astype(np.float32)
    first = probe.probe_responses(model, x, d)["encoder_response"]
    scale_first = probe.global_embedding_scale(model.encoder(torch.as_tensor(x)).detach().numpy())
    with torch.no_grad():
        model.encoder[1].weight.mul_(4.0)
        model.encoder[1].bias.mul_(4.0)
    second = probe.probe_responses(model, x, d)["encoder_response"]
    scale_second = probe.global_embedding_scale(model.encoder(torch.as_tensor(x)).detach().numpy())
    np.testing.assert_allclose(first / scale_first, second / scale_second, rtol=1e-5, atol=1e-7)


def test_matched_random_has_source_norm_and_is_shared():
    source = np.random.default_rng(3).standard_normal((8, 10)).astype(np.float32)
    source[2] = 0.0
    unit = probe.unit_random_directions(8, 10, seed=5)
    matched, valid = probe.matched_random(unit, source)
    norms = np.linalg.norm(source, axis=1)
    np.testing.assert_allclose(np.linalg.norm(matched, axis=1)[valid], norms[valid], rtol=1e-5)
    assert not valid[2] and np.all(matched[2] == 0.0) and valid.sum() == 7
    again, _ = probe.matched_random(probe.unit_random_directions(8, 10, seed=5), source)
    np.testing.assert_array_equal(matched, again)  # identical vectors for H1 and multihop models


def test_zero_norm_and_missing_donor_are_excluded():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    anchors = np.array([0, 1, 2])
    donors = np.array([0, -1, 3])  # self donor -> zero norm, missing donor
    directions = probe.family_directions(X, anchors, donors)
    norms, valid = probe.direction_norms(directions)
    assert valid.tolist() == [False, False, True]
    assert norms[2] == pytest.approx(np.linalg.norm(X[3] - X[2]))


def test_lambda_zero_is_exact_and_eval_disables_dropout():
    torch.manual_seed(0)
    model = TinyMAE().train()
    x = np.random.default_rng(4).standard_normal((9, 6)).astype(np.float32)
    d = np.random.default_rng(5).standard_normal((9, 6)).astype(np.float32)
    out = probe.probe_responses(model, x, d, batch_size=4)  # sets eval internally
    assert not model.training
    assert np.all(out["encoder_response"][0] == 0.0) and np.all(out["decoder_anchor_mse_delta"][0] == 0.0)
    assert np.all(out["encoder_response"][3] > 0.0)
    repeat = probe.probe_responses(model, x, d, batch_size=9)
    np.testing.assert_allclose(out["encoder_response"], repeat["encoder_response"], rtol=1e-6, atol=1e-7)


def test_strict_restore_rejects_shape_missing_extra_and_nonfinite():
    model = TinyMAE()
    good = {k: v.clone() for k, v in TinyMAE().state_dict().items()}
    info = probe.strict_restore(model, good)
    assert info["n_tensors"] == len(good)
    wide = {k: v.clone() for k, v in TinyMAE(genes=7).state_dict().items()}
    with pytest.raises(probe.ProbeBlocked, match=r'"shape_mismatch": \[.*"encoder.1.weight"'):
        probe.strict_restore(TinyMAE(), wide)
    missing = dict(good); missing.pop("decoder.bias")
    with pytest.raises(probe.ProbeBlocked, match=r'"missing": \["decoder.bias"\]'):
        probe.strict_restore(TinyMAE(), missing)
    extra = dict(good); extra["extra.weight"] = torch.zeros(1)
    with pytest.raises(probe.ProbeBlocked, match=r'"unexpected": \["extra.weight"\]'):
        probe.strict_restore(TinyMAE(), extra)
    bad = dict(good); bad["decoder.bias"] = torch.full_like(good["decoder.bias"], float("nan"))
    with pytest.raises(probe.ProbeBlocked, match=r'"nonfinite": \["decoder.bias"\]'):
        probe.strict_restore(TinyMAE(), bad)
    target = TinyMAE()
    probe.strict_restore(target, good)
    assert all(torch.equal(target.state_dict()[k], good[k]) for k in good)


def test_anchor_sampling_is_seeded_bounded_and_unique():
    first = probe.sample_anchors(10_000, 512, seed=9)
    assert first.size == 512 and np.unique(first).size == 512
    np.testing.assert_array_equal(first, probe.sample_anchors(10_000, 512, seed=9))
    assert probe.sample_anchors(30, 512, seed=9).size == 30


def test_far_donors_respect_hop_restriction():
    nodes = [np.array([1, 2, 3])] * 4
    probs = [np.array([0.5, 0.25, 0.25])] * 4
    hops = [np.array([1, 2, 3])] * 4
    donors, drawn = probe.draw_plan_donors(nodes, probs, hops, np.array([0, 1, 2, 3]), seed=1, min_hop=2)
    assert set(donors.tolist()) <= {2, 3} and np.all(drawn >= 2)


def test_config_difference_and_profile_compare():
    assert probe.config_difference({"a": 1, "h": "H1"}, {"a": 1, "h": "H8_B1"}) == {"h": ["H1", "H8_B1"]}
    saved = {"max_hop": 8, "shell_mean_selected_count": {"1": 10.0, "2": 30.0}}
    assert probe.compare_profile(saved, dict(saved), ("max_hop", "shell_mean_selected_count")) == []
    changed = {"max_hop": 8, "shell_mean_selected_count": {"1": 10.0, "2": 31.0}}
    assert probe.compare_profile(saved, changed, ("max_hop", "shell_mean_selected_count")) == [
        "shell_mean_selected_count"]


def test_audit_refuses_object_arrays_and_hashes_ids(tmp_path):
    path = tmp_path / "Pollen.npz"
    np.savez(path, x=np.zeros((2, 2)), cell_ids=np.array(["c1", "c2"]),
             meta=np.array([{"k": 1}, None], dtype=object))
    info = audit.read_npz(path)
    assert info["object_arrays_refused"] == ["meta"]
    assert info["arrays"]["cell_ids"]["n_unique"] == 2
    assert audit.candidate_names("hrvatin_geo.profile.json") == ["hrvatin"]
    assert audit.candidate_names("Baron_Human.npz") == ["BaronHuman"]
    assert audit.compare_ids(["a", "b"], ["b", "c"])["n_shared"] == 1
