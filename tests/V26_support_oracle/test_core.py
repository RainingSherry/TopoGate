import numpy as np
import scipy.sparse as sp

from methods.TopoGate.V26_support_oracle import corruption, protocol
from methods.TopoGate.V26_support_oracle.data import value_only_profiles
from scripts.V26.run_matrix import can_admit


def test_protocol_has_exact_user_dataset_and_arm_contract():
    protocol.validate_protocol()
    assert len(protocol.DATASETS) == 11
    assert protocol.ARMS == ("CLEAN", "P0_RANDOM", "P1_SUPPORT_PRESERVE", "P2_SUPPORT_TARGET", "O_LABEL_ORACLE")


def test_p2_preserves_row_value_multiset_and_crosses_support():
    clean = np.array([[3.0, 0.0, 2.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    corrupted_value, audit = corruption.corrupt_batch(clean, np.array([0]), arm="P2_SUPPORT_TARGET", seed=42, epoch=0)
    assert np.array_equal(np.sort(clean[0]), np.sort(corrupted_value[0]))
    assert audit["support_crossing_total"] > 0


def test_oracle_targets_class_characteristic_support():
    matrix = sp.csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32))
    scores = corruption.build_simple_label_oracle(matrix, np.array([0, 0, 1, 1]))
    source, destination = scores.scores_for_row(0)
    assert source[0] > source[2]
    assert destination[2] > destination[0]


def test_value_only_profile_has_no_coordinate_information_or_zero_padding():
    matrix = sp.csr_matrix(np.array([[1, 0, 2, 0, 0, 0], [0, 2, 0, 1, 0, 0]], dtype=np.float32))
    profile = value_only_profiles(matrix, quantiles=8)
    assert np.array_equal(profile[0], profile[1])


def test_pack_first_admission_reserves_every_active_v26_job():
    row = {"gpu": 6, "free_mib": 10_000}
    active = {101: {"gpu": 6, "reservation_mib": 3_000.0}}
    baseline = {6: 10_000.0}
    assert can_admit(row, 2_000.0, active, baseline)
    active[102] = {"gpu": 6, "reservation_mib": 2_000.0}
    assert not can_admit(row, 2_000.0, active, baseline)


def test_multiclass_oracle_uses_coherent_weighted_other_profile():
    scores = corruption.OracleScores(
        class_probabilities=np.array([[1.0, 0.0], [0.2, 0.5], [0.3, 0.7]], dtype=np.float32),
        class_sizes=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
        metadata={},
    )
    source, destination = scores.scores_for_row(0)
    expected_other = np.array([(0.2 * 20 + 0.3 * 30) / 50, (0.5 * 20 + 0.7 * 30) / 50])
    assert np.allclose(source, np.array([1.0, 0.0]) - expected_other)
    assert np.allclose(destination, expected_other - np.array([1.0, 0.0]))
