#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for E3/E4 analysis CLI.

Tests:
  1. Hand-built synthetic examples with known split/merge rates
  2. Permutation invariance: relabeling cluster IDs should not change rates
  3. Singleton class handling (denominator zero)
  4. Community coassignment computation correctness
  5. End-to-end CLI execution on minimal synthetic data
"""

import tempfile
from pathlib import Path

import numpy as np

from compute_e3_e4_analysis import (
    compute_split_merge_rates_sparse,
    compute_community_coassignment,
    compute_e3_analysis,
    compute_e4_analysis,
    generate_metadata
)


# ============================================================================
#  Test 1: Hand-built synthetic examples
# ============================================================================
def test_perfect_clustering():
    """Perfect clustering: all same-class samples in same cluster."""
    labels_true = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 1, 1, 2, 2])
    
    split_rate, merge_rate, per_class_split, per_class_merge = compute_split_merge_rates_sparse(labels_true, predictions)
    
    assert split_rate == 0.0, "Perfect clustering should have zero split rate"
    assert merge_rate == 0.0, "Perfect clustering should have zero merge rate"
    assert all(v == 0.0 for v in per_class_split.values()), "All classes should have zero split"
    assert all(v == 0.0 for v in per_class_merge.values()), "All classes should have zero merge"


def test_complete_split():
    """Complete split: each sample in its own cluster."""
    labels_true = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0, 1, 2, 3, 4, 5])
    
    split_rate, merge_rate, per_class_split, per_class_merge = compute_split_merge_rates_sparse(labels_true, predictions)
    
    assert split_rate == 1.0, "Complete split should have split rate 1.0"
    assert merge_rate == 0.0, "No merges when all clusters are singletons"


def test_complete_merge():
    """Complete merge: all samples in one cluster."""
    labels_true = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 0, 0, 0, 0])
    
    split_rate, merge_rate, per_class_split, per_class_merge = compute_split_merge_rates_sparse(labels_true, predictions)
    
    assert split_rate == 0.0, "No splits when all in same cluster"
    assert merge_rate == 1.0, "Complete merge should have merge rate 1.0"


def test_known_split_merge():
    """Known example with specific split and merge rates."""
    # Class 0: samples 0,1,2 -> clusters 0,0,1 (1 split pair out of 3 same-class pairs)
    # Class 1: samples 3,4   -> clusters 1,1 (0 split pairs out of 1 same-class pair)
    # Total same-class pairs: 3 + 1 = 4
    # Total split pairs: 1
    # Split rate: 1/4 = 0.25
    
    # Cluster 0: samples 0,1 (both class 0) -> 0 merge pairs
    # Cluster 1: samples 2,3,4 (class 0, 1, 1) -> 1*2 = 2 merge pairs (sample 2 with 3,4)
    # Total different-class pairs: (5*4)/2 - 4 = 10 - 4 = 6
    # Total merge pairs: 2
    # Merge rate: 2/6 = 0.333...
    
    labels_true = np.array([0, 0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1, 1])
    
    split_rate, merge_rate, per_class_split, per_class_merge = compute_split_merge_rates_sparse(labels_true, predictions)
    
    assert abs(split_rate - 0.5) < 1e-6, f"Two split pairs / four same-class pairs: got {split_rate}"
    assert abs(merge_rate - (2/6)) < 1e-6, f"Expected merge rate {2/6:.4f}, got {merge_rate}"


# ============================================================================
#  Test 2: Permutation invariance
# ============================================================================
def test_permutation_invariance():
    """Relabeling cluster IDs should not change rates."""
    labels_true = np.array([0, 0, 1, 1, 2, 2, 2])
    predictions_a = np.array([0, 0, 1, 2, 1, 1, 1])
    
    # Permute cluster IDs: 0->2, 1->0, 2->1
    perm_map = {0: 2, 1: 0, 2: 1}
    predictions_b = np.array([perm_map[p] for p in predictions_a])
    
    split_a, merge_a, _, _ = compute_split_merge_rates_sparse(labels_true, predictions_a)
    split_b, merge_b, _, _ = compute_split_merge_rates_sparse(labels_true, predictions_b)
    
    assert abs(split_a - split_b) < 1e-9, "Split rate should be permutation-invariant"
    assert abs(merge_a - merge_b) < 1e-9, "Merge rate should be permutation-invariant"


# ============================================================================
#  Test 3: Singleton class handling
# ============================================================================
def test_singleton_class():
    """Class with only one sample should have zero split rate (denominator zero)."""
    labels_true = np.array([0, 1, 1, 2])  # class 0 is singleton, class 2 is singleton
    predictions = np.array([0, 0, 1, 2])
    
    split_rate, merge_rate, per_class_split, per_class_merge = compute_split_merge_rates_sparse(labels_true, predictions)
    
    assert np.isnan(per_class_split[0]), "Singleton splitting is undefined, excluded from macro mean"
    assert np.isnan(per_class_split[2])
    assert abs(per_class_merge[0] - 1/3) < 1e-12


# ============================================================================
#  Test 4: Community coassignment
# ============================================================================
def test_community_coassignment_perfect():
    """Same-class samples in different communities always assigned together."""
    labels_true = np.array([0, 0, 0, 0])  # all class 0
    community_ids = np.array([0, 0, 1, 1])  # two communities
    predictions = np.array([0, 0, 0, 0])  # all in same cluster
    
    coassign = compute_community_coassignment(labels_true, community_ids, predictions)
    
    # Should have one entry for (class=0, comm_a=0, comm_b=1)
    assert (0, 0, 1) in coassign, "Should compute coassignment for community pair"
    assert coassign[(0, 0, 1)] == 1.0, "Perfect coassignment should be 1.0"


def test_community_coassignment_zero():
    """Same-class samples in different communities never assigned together."""
    labels_true = np.array([0, 0, 0, 0])
    community_ids = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1])  # community 0 -> cluster 0, community 1 -> cluster 1
    
    coassign = compute_community_coassignment(labels_true, community_ids, predictions)
    
    assert (0, 0, 1) in coassign
    assert coassign[(0, 0, 1)] == 0.0, "Zero coassignment should be 0.0"


def test_community_coassignment_partial():
    """Partial coassignment."""
    labels_true = np.array([0, 0, 0, 0, 0, 0])
    community_ids = np.array([0, 0, 0, 1, 1, 1])
    # Community 0: samples 0,1,2 -> clusters 0,0,1
    # Community 1: samples 3,4,5 -> clusters 0,1,1
    # Pairs: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    # Same cluster: (0,3), (1,3) = 2 out of 9
    predictions = np.array([0, 0, 1, 0, 1, 1])
    
    coassign = compute_community_coassignment(labels_true, community_ids, predictions)
    
    assert (0, 0, 1) in coassign
    expected = 4 / 9
    assert abs(coassign[(0, 0, 1)] - expected) < 1e-6, f"Expected {expected:.4f}, got {coassign[(0, 0, 1)]}"


# ============================================================================
#  Test 5: End-to-end CLI execution
# ============================================================================
def test_e3_analysis_end_to_end():
    """End-to-end test of E3 analysis with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic inputs
        n = 20
        labels_true = np.array([0]*5 + [1]*5 + [2]*10)
        class_names = np.array(['ClassA', 'ClassB', 'ClassC'], dtype='<U20')
        tsne2 = np.random.randn(n, 2)
        
        layout_path = tmpdir / 'fixed_input_layout.npz'
        np.savez(layout_path, labels_true=labels_true, class_names=class_names, tsne2=tsne2)
        
        # Create frozen readouts
        seeds = np.array([7, 42])
        frozen_data = {'seeds': seeds}
        for seed in seeds:
            for arm in ['H1', 'multihop']:
                # 10 readouts, each with n predictions
                preds = np.random.randint(0, 3, size=(10, n))
                frozen_data[f'seed_{seed}_{arm}_predictions'] = preds
        
        frozen_path = tmpdir / 'frozen_readouts_10.npz'
        np.savez(frozen_path, **frozen_data)
        
        output_dir = tmpdir / 'output'
        
        # Run E3 analysis
        compute_e3_analysis('test_case', layout_path, frozen_path, output_dir)
        
        # Check outputs exist
        assert (output_dir / 'e3_pair_errors_per_readout.csv').exists()
        assert (output_dir / 'e3_pair_errors_per_seed.csv').exists()
        assert (output_dir / 'e3_pair_errors_aggregate.csv').exists()
        assert (output_dir / 'e3_class_macro_stats.csv').exists()
        assert (output_dir / 'e3_structure_map.png').exists()


def test_e4_analysis_end_to_end():
    """End-to-end test of E4 analysis with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic inputs
        n = 20
        labels_true = np.array([0]*10 + [1]*10)
        class_names = np.array(['ClassA', 'ClassB'], dtype='<U20')
        tsne2 = np.random.randn(n, 2)
        community_ids = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5)
        
        layout_path = tmpdir / 'fixed_input_layout.npz'
        np.savez(layout_path, labels_true=labels_true, class_names=class_names, tsne2=tsne2)
        
        community_path = tmpdir / 'community_ids.npy'
        np.save(community_path, community_ids)
        
        # Create frozen readouts
        seeds = np.array([7, 42])
        frozen_data = {'seeds': seeds}
        for seed in seeds:
            for arm in ['H1', 'multihop']:
                preds = np.random.randint(0, 3, size=(10, n))
                frozen_data[f'seed_{seed}_{arm}_predictions'] = preds
        
        frozen_path = tmpdir / 'frozen_readouts_10.npz'
        np.savez(frozen_path, **frozen_data)
        
        output_dir = tmpdir / 'output'
        
        # Run E4 analysis
        compute_e4_analysis('test_case', layout_path, frozen_path, community_path, output_dir)
        
        # Check outputs exist
        assert (output_dir / 'e4_community_pair_coassignment.csv').exists()
        assert (output_dir / 'e4_community_pair_selected_display.csv').exists()


def test_against_direct_pair_enumeration():
    rng=np.random.default_rng(19)
    for _ in range(30):
        y=rng.integers(0,3,9); p=rng.integers(0,4,9)
        s,m,_,_=compute_split_merge_rates_sparse(y,p)
        same=[p[i]!=p[j] for i in range(9) for j in range(i+1,9) if y[i]==y[j]]
        diff=[p[i]==p[j] for i in range(9) for j in range(i+1,9) if y[i]!=y[j]]
        assert np.isclose(s,np.mean(same))
        assert np.isclose(m,np.mean(diff))
        g=rng.integers(0,3,9)
        for (c,a,b),v in compute_community_coassignment(y,g,p).items():
            ix=np.flatnonzero((y==c)&(g==a));jx=np.flatnonzero((y==c)&(g==b))
            pairs=[p[i]==p[j] for i in ix for j in jx if a!=b or i<j]
            assert np.isclose(v,np.mean(pairs))


if __name__ == '__main__':
    np.random.seed(20260927)
    tests=[(n,f) for n,f in globals().copy().items() if n.startswith('test_') and callable(f)]
    for name,test in tests:
        test()
        print('PASS',name)
    print(f'{len(tests)} tests passed')

