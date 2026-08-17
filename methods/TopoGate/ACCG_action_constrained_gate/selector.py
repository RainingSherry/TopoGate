from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .feature_model import CrossFittedFeatureModel, FeatureFoldModel

try:
    from numba import njit
except ImportError:  # pragma: no cover - the repository environment includes numba
    def njit(*_args: Any, **_kwargs: Any):  # type: ignore[misc]
        def decorate(function: Any) -> Any:
            return function

        return decorate


@dataclass(frozen=True)
class SelectionResult:
    hard_mask: np.ndarray
    budgets: np.ndarray
    selected_counts: np.ndarray
    joint_delta: np.ndarray
    clean_energy: np.ndarray
    action_energy: np.ndarray
    constraint_infeasible: np.ndarray
    constraint_violated: np.ndarray
    fallback_counts: np.ndarray
    safe_selected_counts: np.ndarray
    hardness_sum: np.ndarray
    profile: dict[str, Any]


@njit(cache=False)
def _candidate_state(
    feature: int,
    z: np.ndarray,
    donor: np.ndarray,
    clean_residual: np.ndarray,
    current_residual: np.ndarray,
    footprint: np.ndarray,
    numerator: float,
    footprint_count: int,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    csc_data: np.ndarray,
    fp_indptr: np.ndarray,
    fp_indices: np.ndarray,
    candidate_marks: np.ndarray,
    affected_marks: np.ndarray,
    stamp: int,
) -> tuple[float, int, float]:
    start_fp = fp_indptr[feature]
    stop_fp = fp_indptr[feature + 1]
    new_count = 0
    for offset in range(start_fp, stop_fp):
        row = fp_indices[offset]
        candidate_marks[row] = stamp
        if not footprint[row]:
            new_count += 1
    candidate_count = footprint_count + new_count
    candidate_numerator = numerator
    change = donor[feature] - z[feature]
    start = csc_indptr[feature]
    stop = csc_indptr[feature + 1]
    for offset in range(start, stop):
        row = csc_indices[offset]
        affected_marks[row] = stamp
        old_value = current_residual[row]
        new_value = old_value + csc_data[offset] * change
        old_diff = old_value * old_value - clean_residual[row] * clean_residual[row]
        new_diff = new_value * new_value - clean_residual[row] * clean_residual[row]
        if footprint[row]:
            candidate_numerator += new_diff - old_diff
        elif candidate_marks[row] == stamp:
            candidate_numerator += new_diff
    for offset in range(start_fp, stop_fp):
        row = fp_indices[offset]
        if not footprint[row] and affected_marks[row] != stamp:
            old_value = current_residual[row]
            candidate_numerator += old_value * old_value - clean_residual[row] * clean_residual[row]
    if candidate_count <= 0:
        return candidate_numerator, candidate_count, 0.0
    return candidate_numerator, candidate_count, candidate_numerator / candidate_count


@njit(cache=False)
def _commit_candidate(
    feature: int,
    z: np.ndarray,
    donor: np.ndarray,
    current_residual: np.ndarray,
    footprint: np.ndarray,
    selected: np.ndarray,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    csc_data: np.ndarray,
    fp_indptr: np.ndarray,
    fp_indices: np.ndarray,
) -> None:
    change = donor[feature] - z[feature]
    for offset in range(csc_indptr[feature], csc_indptr[feature + 1]):
        current_residual[csc_indices[offset]] += csc_data[offset] * change
    for offset in range(fp_indptr[feature], fp_indptr[feature + 1]):
        footprint[fp_indices[offset]] = True
    selected[feature] = True


@njit(cache=False)
def _pair_state(
    first: int,
    second: int,
    z: np.ndarray,
    donor: np.ndarray,
    clean_residual: np.ndarray,
    current_residual: np.ndarray,
    footprint: np.ndarray,
    selected: np.ndarray,
    numerator: float,
    footprint_count: int,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    csc_data: np.ndarray,
    fp_indptr: np.ndarray,
    fp_indices: np.ndarray,
) -> tuple[float, int, float]:
    temporary_residual = current_residual.copy()
    temporary_footprint = footprint.copy()
    temporary_selected = selected.copy()
    candidate_marks = np.zeros(selected.size, dtype=np.int64)
    affected_marks = np.zeros(selected.size, dtype=np.int64)
    first_num, first_count, _first_delta = _candidate_state(
        first,
        z,
        donor,
        clean_residual,
        temporary_residual,
        temporary_footprint,
        numerator,
        footprint_count,
        csc_indptr,
        csc_indices,
        csc_data,
        fp_indptr,
        fp_indices,
        candidate_marks,
        affected_marks,
        1,
    )
    _commit_candidate(
        first,
        z,
        donor,
        temporary_residual,
        temporary_footprint,
        temporary_selected,
        csc_indptr,
        csc_indices,
        csc_data,
        fp_indptr,
        fp_indices,
    )
    second_num, second_count, second_delta = _candidate_state(
        second,
        z,
        donor,
        clean_residual,
        temporary_residual,
        temporary_footprint,
        first_num,
        first_count,
        csc_indptr,
        csc_indices,
        csc_data,
        fp_indptr,
        fp_indices,
        candidate_marks,
        affected_marks,
        2,
    )
    return second_num, second_count, second_delta


@njit(cache=False)
def _joint_select_row(
    scores: np.ndarray,
    eligible: np.ndarray,
    budget: int,
    z: np.ndarray,
    donor: np.ndarray,
    epsilon: float,
    clean_residual: np.ndarray,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    csc_data: np.ndarray,
    fp_indptr: np.ndarray,
    fp_indices: np.ndarray,
    greedy_passes: int,
    pair_lookahead: int,
    fallback_mode: int,
) -> tuple[np.ndarray, float, float, float, int, int, int]:
    n_features = scores.size
    selected = np.zeros(n_features, dtype=np.bool_)
    footprint = np.zeros(n_features, dtype=np.bool_)
    current_residual = clean_residual.copy()
    order = np.argsort(scores)[::-1]
    candidate_marks = np.zeros(n_features, dtype=np.int64)
    affected_marks = np.zeros(n_features, dtype=np.int64)
    stamp = 0
    numerator = 0.0
    footprint_count = 0
    selected_count = 0
    safe_count = 0
    fallback_count = 0
    for _pass in range(greedy_passes):
        before = selected_count
        for order_index in range(order.size):
            feature = order[order_index]
            if selected_count >= budget:
                break
            if not eligible[feature] or selected[feature]:
                continue
            stamp += 1
            candidate_numerator, candidate_count, candidate_delta = _candidate_state(
                feature,
                z,
                donor,
                clean_residual,
                current_residual,
                footprint,
                numerator,
                footprint_count,
                csc_indptr,
                csc_indices,
                csc_data,
                fp_indptr,
                fp_indices,
                candidate_marks,
                affected_marks,
                stamp,
            )
            if candidate_delta <= epsilon + 1e-12:
                _commit_candidate(
                    feature,
                    z,
                    donor,
                    current_residual,
                    footprint,
                    selected,
                    csc_indptr,
                    csc_indices,
                    csc_data,
                    fp_indptr,
                    fp_indices,
                )
                numerator = candidate_numerator
                footprint_count = candidate_count
                selected_count += 1
                safe_count += 1
        if selected_count >= budget or selected_count == before:
            break
    if selected_count + 1 < budget and pair_lookahead > 0:
        made_progress = True
        while selected_count + 1 < budget and made_progress:
            made_progress = False
            inspected = 0
            for order_index in range(order.size):
                first = order[order_index]
                if not eligible[first] or selected[first]:
                    continue
                inspected += 1
                if inspected > pair_lookahead:
                    break
                second = -1
                second_score = -np.inf
                for offset in range(fp_indptr[first], fp_indptr[first + 1]):
                    candidate = fp_indices[offset]
                    if candidate != first and eligible[candidate] and not selected[candidate] and scores[candidate] > second_score:
                        second = candidate
                        second_score = scores[candidate]
                if second < 0:
                    continue
                pair_num, pair_count, pair_delta = _pair_state(
                    first,
                    second,
                    z,
                    donor,
                    clean_residual,
                    current_residual,
                    footprint,
                    selected,
                    numerator,
                    footprint_count,
                    csc_indptr,
                    csc_indices,
                    csc_data,
                    fp_indptr,
                    fp_indices,
                )
                if pair_delta <= epsilon + 1e-12:
                    _commit_candidate(
                        first,
                        z,
                        donor,
                        current_residual,
                        footprint,
                        selected,
                        csc_indptr,
                        csc_indices,
                        csc_data,
                        fp_indptr,
                        fp_indices,
                    )
                    _commit_candidate(
                        second,
                        z,
                        donor,
                        current_residual,
                        footprint,
                        selected,
                        csc_indptr,
                        csc_indices,
                        csc_data,
                        fp_indptr,
                        fp_indices,
                    )
                    numerator = pair_num
                    footprint_count = pair_count
                    selected_count += 2
                    safe_count += 2
                    made_progress = True
                    break
    infeasible = selected_count < budget
    if infeasible and fallback_mode == 1:
        while selected_count < budget:
            best_feature = -1
            best_delta = np.inf
            best_score = -np.inf
            best_numerator = numerator
            best_count = footprint_count
            for feature in range(n_features):
                if not eligible[feature] or selected[feature]:
                    continue
                stamp += 1
                candidate_numerator, candidate_count, candidate_delta = _candidate_state(
                    feature,
                    z,
                    donor,
                    clean_residual,
                    current_residual,
                    footprint,
                    numerator,
                    footprint_count,
                    csc_indptr,
                    csc_indices,
                    csc_data,
                    fp_indptr,
                    fp_indices,
                    candidate_marks,
                    affected_marks,
                    stamp,
                )
                if candidate_delta < best_delta - 1e-12 or (
                    abs(candidate_delta - best_delta) <= 1e-12 and scores[feature] > best_score
                ):
                    best_feature = feature
                    best_delta = candidate_delta
                    best_score = scores[feature]
                    best_numerator = candidate_numerator
                    best_count = candidate_count
            if best_feature < 0:
                break
            _commit_candidate(
                best_feature,
                z,
                donor,
                current_residual,
                footprint,
                selected,
                csc_indptr,
                csc_indices,
                csc_data,
                fp_indptr,
                fp_indices,
            )
            numerator = best_numerator
            footprint_count = best_count
            selected_count += 1
            fallback_count += 1
    if footprint_count:
        clean_energy = 0.0
        action_energy = 0.0
        for feature in range(n_features):
            if footprint[feature]:
                clean_energy += clean_residual[feature] * clean_residual[feature]
                action_energy += current_residual[feature] * current_residual[feature]
        clean_energy /= footprint_count
        action_energy /= footprint_count
    else:
        clean_energy = 0.0
        action_energy = 0.0
    return selected, action_energy - clean_energy, clean_energy, action_energy, int(infeasible), fallback_count, safe_count


@njit(cache=False)
def _singleton_deltas(
    eligible: np.ndarray,
    z: np.ndarray,
    donor: np.ndarray,
    clean_residual: np.ndarray,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    csc_data: np.ndarray,
    fp_indptr: np.ndarray,
    fp_indices: np.ndarray,
) -> np.ndarray:
    n_features = eligible.size
    result = np.full(n_features, np.inf, dtype=np.float64)
    footprint = np.zeros(n_features, dtype=np.bool_)
    current = clean_residual.copy()
    candidate_marks = np.zeros(n_features, dtype=np.int64)
    affected_marks = np.zeros(n_features, dtype=np.int64)
    stamp = 0
    for feature in range(n_features):
        if not eligible[feature]:
            continue
        stamp += 1
        _num, _count, delta = _candidate_state(
            feature,
            z,
            donor,
            clean_residual,
            current,
            footprint,
            0.0,
            0,
            csc_indptr,
            csc_indices,
            csc_data,
            fp_indptr,
            fp_indices,
            candidate_marks,
            affected_marks,
            stamp,
        )
        result[feature] = delta
    return result


def _clean_residual(fold: FeatureFoldModel, z: np.ndarray) -> np.ndarray:
    return np.asarray(fold.residual_operator_csc.dot(np.asarray(z, dtype=np.float64)), dtype=np.float64)


def _coordinate_select_row(
    scores: np.ndarray,
    eligible: np.ndarray,
    budget: int,
    z: np.ndarray,
    donor: np.ndarray,
    epsilon: float,
    fold: FeatureFoldModel,
    fallback: str,
) -> tuple[np.ndarray, float, float, float, bool, int, int]:
    clean_residual = _clean_residual(fold, z)
    singleton = _singleton_deltas(
        eligible,
        z,
        donor,
        clean_residual,
        fold.residual_operator_csc.indptr,
        fold.residual_operator_csc.indices,
        fold.residual_operator_csc.data,
        fold.footprint_indptr,
        fold.footprint_indices,
    )
    order = np.argsort(scores)[::-1]
    safe = [int(feature) for feature in order if eligible[feature] and singleton[feature] <= epsilon]
    chosen = safe[:budget]
    infeasible = len(chosen) < budget
    fallback_count = 0
    if infeasible and fallback == "least_violation":
        remaining = [int(feature) for feature in np.argsort(singleton) if eligible[feature] and feature not in chosen]
        needed = budget - len(chosen)
        chosen.extend(remaining[:needed])
        fallback_count = min(needed, len(remaining))
    mask = np.zeros(scores.size, dtype=np.bool_)
    mask[np.asarray(chosen, dtype=np.int64)] = True
    delta, clean_energy, action_energy = fold.action_delta(z, donor, mask)
    return mask, delta, clean_energy, action_energy, infeasible, fallback_count, min(len(safe), budget)


def select_action(
    hardness_scores: np.ndarray,
    eligible: np.ndarray,
    z: np.ndarray,
    donor_z: np.ndarray,
    *,
    row_ids: np.ndarray,
    epsilon: np.ndarray | float,
    model: CrossFittedFeatureModel,
    mask_ratio: float,
    selector_mode: str,
    greedy_passes: int,
    pair_lookahead: int,
    fallback: str,
) -> SelectionResult:
    scores = np.asarray(hardness_scores, dtype=np.float64)
    eligible_values = np.asarray(eligible, dtype=np.bool_)
    transformed = np.asarray(z, dtype=np.float64)
    donors = np.asarray(donor_z, dtype=np.float64)
    rows = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or eligible_values.shape != scores.shape or transformed.shape != scores.shape or donors.shape != scores.shape:
        raise ValueError("selection arrays must have one matching [batch, feature] shape")
    if rows.size != scores.shape[0]:
        raise ValueError("row_ids must have one entry per batch row")
    if not 0.0 < float(mask_ratio) <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    eps = np.broadcast_to(np.asarray(epsilon, dtype=np.float64), (scores.shape[0],))
    hard = np.zeros(scores.shape, dtype=np.bool_)
    budgets = np.ceil(eligible_values.sum(axis=1) * float(mask_ratio)).astype(np.int64)
    budgets = np.minimum(budgets, eligible_values.sum(axis=1))
    joint_delta = np.zeros(scores.shape[0], dtype=np.float64)
    clean_energy = np.zeros(scores.shape[0], dtype=np.float64)
    action_energy = np.zeros(scores.shape[0], dtype=np.float64)
    infeasible = np.zeros(scores.shape[0], dtype=np.bool_)
    fallback_counts = np.zeros(scores.shape[0], dtype=np.int64)
    safe_counts = np.zeros(scores.shape[0], dtype=np.int64)
    for batch_row, row_id in enumerate(rows):
        fold = model.fold_for_row(int(row_id))
        budget = int(budgets[batch_row])
        if budget == 0:
            continue
        if selector_mode == "joint":
            clean_residual = _clean_residual(fold, transformed[batch_row])
            selected, delta, clean_value, action_value, is_infeasible, fallback_count, safe_count = _joint_select_row(
                scores[batch_row],
                eligible_values[batch_row],
                budget,
                transformed[batch_row],
                donors[batch_row],
                float(eps[batch_row]),
                clean_residual,
                fold.residual_operator_csc.indptr,
                fold.residual_operator_csc.indices,
                fold.residual_operator_csc.data,
                fold.footprint_indptr,
                fold.footprint_indices,
                int(greedy_passes),
                int(pair_lookahead),
                1 if fallback == "least_violation" else 0,
            )
        elif selector_mode == "coordinate":
            selected, delta, clean_value, action_value, is_infeasible, fallback_count, safe_count = _coordinate_select_row(
                scores[batch_row],
                eligible_values[batch_row],
                budget,
                transformed[batch_row],
                donors[batch_row],
                float(eps[batch_row]),
                fold,
                fallback,
            )
        else:
            raise ValueError(f"unsupported selector_mode: {selector_mode!r}")
        hard[batch_row] = selected
        joint_delta[batch_row] = delta
        clean_energy[batch_row] = clean_value
        action_energy[batch_row] = action_value
        infeasible[batch_row] = bool(is_infeasible)
        fallback_counts[batch_row] = int(fallback_count)
        safe_counts[batch_row] = int(safe_count)
    selected_counts = hard.sum(axis=1).astype(np.int64)
    violated = joint_delta > eps + 1e-9
    hardness_sum = np.sum(np.where(hard, scores, 0.0), axis=1)
    return SelectionResult(
        hard_mask=hard.astype(np.float32),
        budgets=budgets,
        selected_counts=selected_counts,
        joint_delta=joint_delta,
        clean_energy=clean_energy,
        action_energy=action_energy,
        constraint_infeasible=infeasible,
        constraint_violated=violated,
        fallback_counts=fallback_counts,
        safe_selected_counts=safe_counts,
        hardness_sum=hardness_sum,
        profile={
            "selector_mode": selector_mode,
            "fallback": fallback,
            "rows": int(scores.shape[0]),
            "exact_budget_rows": int(np.count_nonzero(selected_counts == budgets)),
            "infeasible_rows": int(np.count_nonzero(infeasible)),
            "violating_rows": int(np.count_nonzero(violated)),
        },
    )


def straight_through_mask(
    logits: torch.Tensor,
    eligible: torch.Tensor,
    gumbel: torch.Tensor,
    hard_mask: torch.Tensor,
    budgets: torch.Tensor,
    *,
    gumbel_scale: float,
    tau: float,
) -> torch.Tensor:
    """Use ACCG's constrained hard action with V21's smooth backward path."""

    if logits.shape != eligible.shape or logits.shape != gumbel.shape or logits.shape != hard_mask.shape:
        raise ValueError("straight-through tensors must have identical shapes")
    noisy = logits + float(gumbel_scale) * gumbel
    masked = noisy.masked_fill(~eligible.to(torch.bool), -torch.inf)
    max_budget = int(budgets.max().item()) if budgets.numel() else 0
    if max_budget == 0:
        return logits * 0.0
    top_values = torch.topk(masked, k=max_budget, dim=1, largest=True, sorted=True).values
    threshold = top_values.gather(1, (budgets - 1).clamp_min(0)[:, None])
    valid = budgets.gt(0)[:, None]
    soft = torch.sigmoid((noisy - threshold.detach()) / float(tau))
    soft = soft * eligible.to(logits.dtype) * valid.to(logits.dtype)
    return hard_mask.to(logits.dtype) + soft - soft.detach()


def exact_constrained_action(
    scores: np.ndarray,
    eligible: np.ndarray,
    budget: int,
    z: np.ndarray,
    donor_z: np.ndarray,
    *,
    epsilon: float,
    fold: FeatureFoldModel,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Brute-force oracle for small W5 instances and approximation-gap audits."""

    candidates = np.flatnonzero(np.asarray(eligible, dtype=np.bool_))
    if budget < 0 or budget > candidates.size:
        raise ValueError("budget is outside the eligible set")
    best_mask: np.ndarray | None = None
    best_hardness = -np.inf
    best_delta = np.inf
    feasible = False
    for combination in itertools.combinations(candidates.tolist(), int(budget)):
        mask = np.zeros(np.asarray(scores).size, dtype=np.bool_)
        mask[np.asarray(combination, dtype=np.int64)] = True
        delta, _clean, _action = fold.action_delta(z, donor_z, mask)
        hardness = float(np.sum(np.asarray(scores)[mask]))
        is_feasible = delta <= float(epsilon) + 1e-12
        if is_feasible and (not feasible or hardness > best_hardness):
            feasible = True
            best_mask = mask
            best_hardness = hardness
            best_delta = delta
        elif not feasible and (delta < best_delta - 1e-12 or (abs(delta - best_delta) <= 1e-12 and hardness > best_hardness)):
            best_mask = mask
            best_hardness = hardness
            best_delta = delta
    if best_mask is None:
        best_mask = np.zeros(np.asarray(scores).size, dtype=np.bool_)
    return best_mask, {
        "feasible": feasible,
        "hardness": float(best_hardness if np.isfinite(best_hardness) else 0.0),
        "joint_delta": float(best_delta if np.isfinite(best_delta) else 0.0),
        "combinations": int(math.comb(candidates.size, int(budget))) if budget <= candidates.size else 0,
    }
