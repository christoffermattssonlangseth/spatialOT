"""Optimal Transport transitions for RRMap AnnData objects.

This module computes Optimal Transport (OT) transitions between consecutive
course/timepoint labels in ``adata.obs[course_key]`` using a latent embedding
from ``adata.obsm``.

Implemented solvers (POT):
- Balanced entropic OT via ``ot.sinkhorn``
- Unbalanced entropic OT via ``ot.unbalanced.sinkhorn_unbalanced``

Modes:
- ``centroid`` (default): one point per ``(course, state)`` centroid.
- ``cell``: one point per cell, with optional per-course subsampling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
from anndata import AnnData
from scipy import sparse

TransitionDict = Dict[Tuple[Any, Any], pd.DataFrame]
PlanPayload = Dict[str, Any]
PlanDict = Dict[Tuple[Any, Any], PlanPayload]


def _as_numpy_2d(x: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert array-like/sparse input to a dense 2D NumPy array."""
    if sparse.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}.")
    return arr


def _resolve_embedding_key(adata: AnnData, embedding_key: Optional[str]) -> str:
    """Resolve embedding key with fallback priority X_scVI -> X_umap -> X_pca."""
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            msg = (
                f"Embedding key '{embedding_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
            warnings.warn(msg)
            raise KeyError(msg)
        return embedding_key

    for key in ("X_scVI", "X_umap", "X_pca"):
        if key in adata.obsm:
            return key

    msg = (
        "No embedding found for OT cost computation. Provide embedding_key or "
        "add one of: 'X_scVI', 'X_umap', 'X_pca' to adata.obsm."
    )
    warnings.warn(msg)
    raise KeyError(msg)


def _resolve_course_order(
    course_series: pd.Series,
    course_order: Optional[Sequence[Any]] = None,
) -> List[Any]:
    """Resolve course ordering.

    Rules:
    1. If explicit ``course_order`` is given: use it (filtered to present labels),
       then append any missing present labels sorted by string.
    2. Else, if ``course_series`` is ordered categorical: use categorical order.
    3. Else sort unique labels by string representation.
    """
    present = pd.Index(course_series.dropna().unique())
    present_set = set(present.tolist())

    if course_order is not None:
        explicit_raw = list(course_order)
        seen: set = set()
        explicit: List[Any] = []
        duplicates: List[Any] = []
        for value in explicit_raw:
            if value in seen:
                duplicates.append(value)
                continue
            seen.add(value)
            explicit.append(value)
        if duplicates:
            warnings.warn(
                "course_order contains duplicate labels; only first occurrences "
                f"will be used. Duplicates: {duplicates}"
            )

        explicit_set = set(explicit)
        extra = [x for x in explicit if x not in present_set]
        if extra:
            warnings.warn(
                f"course_order contains labels not present in adata: {extra}. "
                "They will be ignored."
            )

        ordered = [x for x in explicit if x in present_set]
        missing = [x for x in present.tolist() if x not in explicit_set]
        if missing:
            warnings.warn(
                "Some present course labels were missing in course_order and will "
                f"be appended sorted by string: {sorted(missing, key=lambda v: str(v))}"
            )
            ordered.extend(sorted(missing, key=lambda v: str(v)))
        return ordered

    if isinstance(course_series.dtype, pd.CategoricalDtype) and course_series.dtype.ordered:
        return [x for x in course_series.dtype.categories.tolist() if x in present_set]

    return sorted(present.tolist(), key=lambda v: str(v))


def _resolve_state_order(state_series: pd.Series) -> List[Any]:
    """Resolve stable global state order for output alignment."""
    present = pd.Index(state_series.dropna().unique())
    present_set = set(present.tolist())
    if isinstance(state_series.dtype, pd.CategoricalDtype):
        ordered = [x for x in state_series.dtype.categories.tolist() if x in present_set]
        # Include any values not in categories (rare but possible in some workflows).
        missing = [x for x in present.tolist() if x not in set(ordered)]
        if missing:
            ordered.extend(sorted(missing, key=lambda v: str(v)))
        return ordered
    return sorted(present.tolist(), key=lambda v: str(v))


def _prepare_cost_matrix(
    x_source: np.ndarray,
    x_target: np.ndarray,
    metric: str = "sqeuclidean",
    rescale_by_positive_median: bool = True,
) -> Tuple[np.ndarray, float]:
    """Compute non-negative OT cost matrix and optionally rescale by median(M[M>0])."""
    M = np.asarray(ot.dist(x_source, x_target, metric=metric), dtype=np.float64)
    np.maximum(M, 0.0, out=M)

    scale = 1.0
    if rescale_by_positive_median:
        positive = M[M > 0]
        if positive.size > 0:
            median = float(np.median(positive))
            if median > 0:
                M /= median
                scale = median
    return M, scale


def _make_weights(
    counts: np.ndarray,
    method: str,
    unbalanced_mass_mode: str,
) -> np.ndarray:
    """Construct OT marginals.

    For balanced OT, marginals are always normalized to sum to 1.
    For unbalanced OT:
    - ``normalized``: normalized to sum to 1.
    - ``raw``: raw counts (sum equals number of elements represented).
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")

    total = float(counts.sum())
    if total <= 0:
        raise ValueError("counts sum must be > 0.")

    if method == "balanced":
        return counts / total

    if unbalanced_mass_mode == "normalized":
        return counts / total
    if unbalanced_mass_mode == "raw":
        return counts

    raise ValueError(
        "unbalanced_mass_mode must be one of {'normalized', 'raw'}. "
        f"Got: {unbalanced_mass_mode}"
    )


def _compute_centroids(
    x: np.ndarray,
    states: np.ndarray,
    ordered_states: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Compute per-state centroids and counts in a given state order."""
    states = np.asarray(states)

    centroids: List[np.ndarray] = []
    counts: List[int] = []
    labels: List[Any] = []

    for state in ordered_states:
        mask = states == state
        n = int(mask.sum())
        if n == 0:
            continue
        labels.append(state)
        counts.append(n)
        centroids.append(x[mask].mean(axis=0))

    if not centroids:
        return np.zeros((0, x.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.float64), []

    return (
        np.vstack(centroids).astype(np.float64, copy=False),
        np.asarray(counts, dtype=np.float64),
        labels,
    )


def _run_ot_solver(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    method: str,
    reg: float,
    reg_m: float,
) -> np.ndarray:
    """Run POT OT solver and return non-negative dense transport plan."""
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}.")
    if method == "unbalanced" and reg_m <= 0:
        raise ValueError(f"reg_m must be > 0 for unbalanced OT, got {reg_m}.")

    if method == "balanced":
        plan = ot.sinkhorn(a, b, M, reg=reg)
    elif method == "unbalanced":
        plan = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg, reg_m=reg_m)
    else:
        raise ValueError("method must be one of {'balanced', 'unbalanced'}")

    plan = np.asarray(plan, dtype=np.float64)
    np.maximum(plan, 0.0, out=plan)

    if not np.all(np.isfinite(plan)):
        raise FloatingPointError("OT solver returned non-finite values in transport plan.")

    return plan


def _embed_matrix(
    matrix: np.ndarray,
    source_labels: Sequence[Any],
    target_labels: Sequence[Any],
    full_source_labels: Sequence[Any],
    full_target_labels: Sequence[Any],
) -> np.ndarray:
    """Embed matrix defined on source_labels x target_labels into full label axes."""
    out = np.zeros((len(full_source_labels), len(full_target_labels)), dtype=np.float64)
    src_pos = {label: i for i, label in enumerate(full_source_labels)}
    tgt_pos = {label: j for j, label in enumerate(full_target_labels)}

    src_idx = [src_pos[s] for s in source_labels]
    tgt_idx = [tgt_pos[t] for t in target_labels]
    out[np.ix_(src_idx, tgt_idx)] = matrix
    return out


def _aggregate_cell_plan_to_states(
    cell_plan: np.ndarray,
    source_states: np.ndarray,
    target_states: np.ndarray,
    source_order: Sequence[Any],
    target_order: Sequence[Any],
) -> np.ndarray:
    """Aggregate cell-cell transport mass to state-state transport mass."""
    src_codes = pd.Categorical(source_states, categories=list(source_order)).codes
    tgt_codes = pd.Categorical(target_states, categories=list(target_order)).codes

    if np.any(src_codes < 0):
        raise ValueError("Encountered source states not present in source_order.")
    if np.any(tgt_codes < 0):
        raise ValueError("Encountered target states not present in target_order.")

    src_membership = sparse.csr_matrix(
        (np.ones(src_codes.size, dtype=np.float64), (np.arange(src_codes.size), src_codes)),
        shape=(src_codes.size, len(source_order)),
    )
    tgt_membership = sparse.csr_matrix(
        (np.ones(tgt_codes.size, dtype=np.float64), (np.arange(tgt_codes.size), tgt_codes)),
        shape=(tgt_codes.size, len(target_order)),
    )

    aggregated = src_membership.T @ cell_plan @ tgt_membership
    if sparse.issparse(aggregated):
        aggregated = aggregated.toarray()
    return np.asarray(aggregated, dtype=np.float64)


def _normalize_transition(mass_matrix: np.ndarray) -> np.ndarray:
    """Normalize state-state mass matrix to sum to 1 (if total mass > 0)."""
    total = float(mass_matrix.sum())
    if total <= 0:
        return np.zeros_like(mass_matrix, dtype=np.float64)
    return mass_matrix / total


def _course_state_count_warning(
    course_label: Any,
    states: np.ndarray,
    min_states: int = 2,
    min_cells_per_state: int = 2,
) -> None:
    """Emit warning when a course has too few represented states."""
    counts = pd.Series(states).value_counts(dropna=True)
    n_states = int(counts.shape[0])
    if n_states < min_states:
        warnings.warn(
            f"Course '{course_label}' has only {n_states} state(s). "
            f"Expected at least {min_states} for informative transitions."
        )
        return

    enough = int((counts >= min_cells_per_state).sum())
    if enough < min_states:
        warnings.warn(
            f"Course '{course_label}' has only {enough} state(s) with at least "
            f"{min_cells_per_state} cells. Transitions may be unstable."
        )


def compute_ot_transitions(
    adata: AnnData,
    course_key: str = "course",
    state_key: str = "anno_L2",
    embedding_key: Optional[str] = None,
    method: str = "unbalanced",
    mode: str = "centroid",
    course_order: Optional[Sequence[Any]] = None,
    reg: float = 0.05,
    reg_m: float = 10.0,
    max_cells_per_course: int = 20000,
    random_state: int = 0,
    return_plans: bool = False,
    include_all_states: bool = False,
    unbalanced_mass_mode: str = "normalized",
    metric: str = "sqeuclidean",
    rescale_cost_by_median: bool = True,
) -> Union[TransitionDict, Tuple[TransitionDict, PlanDict]]:
    """Compute OT transitions between consecutive courses in an AnnData object.

    Parameters
    ----------
    adata
        AnnData containing cell-level observations and embedding.
    course_key
        ``adata.obs`` column with timepoint labels.
    state_key
        ``adata.obs`` column with cell state labels.
    embedding_key
        Key in ``adata.obsm`` used for OT costs. If ``None``, fallback order is
        ``X_scVI`` -> ``X_umap`` -> ``X_pca``.
    method
        OT variant: ``balanced`` or ``unbalanced``.
    mode
        ``centroid`` (default) or ``cell``.
        - ``centroid``: OT between per-(course, state) centroids.
        - ``cell``: OT between individual cells (can be heavy).
    course_order
        Optional explicit order of courses. If omitted, ordered categoricals are
        respected; otherwise labels are sorted by string.
    reg
        Entropic regularization strength for Sinkhorn.
    reg_m
        Mass regularization for unbalanced Sinkhorn.
    max_cells_per_course
        In ``cell`` mode, optional per-course subsampling cap.
    random_state
        Seed for subsampling.
    return_plans
        If ``True``, also return raw OT plan payloads for each course pair.
    include_all_states
        If ``True``, transition matrices are aligned to the global union of states.
        If ``False`` (default), absent states are dropped per course pair.
    unbalanced_mass_mode
        Controls marginals for unbalanced OT:
        - ``normalized``: marginals sum to 1.
        - ``raw``: raw counts, so total mass scales with represented cell counts.
          In this setting, row/column sums of the unbalanced plan can be compared
          directly to cell or centroid counts.
    metric
        Distance metric passed to ``ot.dist`` (default: ``sqeuclidean``).
    rescale_cost_by_median
        If ``True``, divide costs by ``median(M[M>0])`` for numerical stability.

    Returns
    -------
    transitions : dict
        Mapping ``(course_t, course_tp1) -> transition DataFrame`` where rows are
        source states and columns are target states.
    plans : dict, optional
        Returned only when ``return_plans=True``. Contains raw OT plan payloads.

    Notes
    -----
    - Balanced OT uses ``ot.sinkhorn(a, b, M, reg=reg)``.
    - Unbalanced OT uses
      ``ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg, reg_m=reg_m)``.
    - Transition matrices are normalized to sum to 1.
    """
    if course_key not in adata.obs:
        raise KeyError(f"{course_key!r} not found in adata.obs")
    if state_key not in adata.obs:
        raise KeyError(f"{state_key!r} not found in adata.obs")
    if method not in {"balanced", "unbalanced"}:
        raise ValueError("method must be one of {'balanced', 'unbalanced'}")
    if mode not in {"centroid", "cell"}:
        raise ValueError("mode must be one of {'centroid', 'cell'}")
    if unbalanced_mass_mode not in {"normalized", "raw"}:
        raise ValueError(
            "unbalanced_mass_mode must be one of {'normalized', 'raw'}"
        )
    if max_cells_per_course <= 0:
        raise ValueError("max_cells_per_course must be > 0")

    emb_key = _resolve_embedding_key(adata, embedding_key)
    embedding = _as_numpy_2d(adata.obsm[emb_key], dtype=np.float32)
    if embedding.shape[0] != adata.n_obs:
        raise ValueError(
            f"Embedding '{emb_key}' has {embedding.shape[0]} rows but adata has "
            f"{adata.n_obs} observations."
        )

    course_series = adata.obs[course_key]
    state_series = adata.obs[state_key]

    valid_mask = course_series.notna().to_numpy() & state_series.notna().to_numpy()
    if not np.all(valid_mask):
        n_drop = int((~valid_mask).sum())
        warnings.warn(
            f"Dropping {n_drop} cells with missing course/state labels for OT."
        )

    valid_courses = course_series.loc[valid_mask]
    valid_states = state_series.loc[valid_mask]

    ordered_courses = _resolve_course_order(valid_courses, course_order=course_order)
    if len(ordered_courses) < 2:
        warnings.warn(
            f"Need at least 2 courses to compute transitions; got {len(ordered_courses)}."
        )
        empty_transitions: TransitionDict = {}
        if return_plans:
            return empty_transitions, {}
        return empty_transitions

    global_state_order = _resolve_state_order(valid_states)

    course_values = course_series.to_numpy()
    state_values = state_series.to_numpy()

    rng = np.random.default_rng(random_state)
    course_data: Dict[Any, Dict[str, Any]] = {}

    for course in ordered_courses:
        idx_full = np.flatnonzero((course_values == course) & valid_mask)
        states_full = state_values[idx_full]
        _course_state_count_warning(course, states_full)

        idx_use = idx_full
        sampled = False
        if mode == "cell" and idx_full.size > max_cells_per_course:
            sampled = True
            idx_use = rng.choice(idx_full, size=max_cells_per_course, replace=False)
            idx_use = np.sort(idx_use)
            warnings.warn(
                f"Cell mode: subsampled course '{course}' from {idx_full.size} "
                f"to {idx_use.size} cells."
            )

        course_data[course] = {
            "indices_full": idx_full,
            "indices_used": idx_use,
            "n_full": int(idx_full.size),
            "n_used": int(idx_use.size),
            "sampled": sampled,
            "states_full": states_full,
            "states_used": state_values[idx_use],
        }

    transitions: TransitionDict = {}
    plans: PlanDict = {}

    # Required storage path for transition matrices.
    ot_uns = adata.uns.setdefault("ot_transitions", {})
    method_uns = ot_uns.setdefault(method, {})
    mode_uns = method_uns.setdefault(mode, {})
    # Overwrite this method/mode slot on each run to avoid stale pair keys.
    mode_uns.clear()
    metadata_uns = mode_uns.setdefault("__metadata__", {})
    metadata_uns["global"] = {
        "course_key": course_key,
        "state_key": state_key,
        "embedding_key": emb_key,
        "method": method,
        "mode": mode,
        "course_order": list(ordered_courses),
        "reg": float(reg),
        "reg_m": float(reg_m),
        "metric": metric,
        "max_cells_per_course": int(max_cells_per_course),
        "include_all_states": bool(include_all_states),
        "unbalanced_mass_mode": unbalanced_mass_mode,
        "rescale_cost_by_median": bool(rescale_cost_by_median),
        "random_state": int(random_state),
    }
    metadata_uns["course_stats"] = {
        str(course): {
            "n_cells_full": int(payload["n_full"]),
            "n_cells_used": int(payload["n_used"]),
            "sampled": bool(payload["sampled"]),
            "n_states_full": int(len(pd.Index(payload["states_full"]).unique())),
            "n_states_used": int(len(pd.Index(payload["states_used"]).unique())),
        }
        for course, payload in course_data.items()
    }

    for i in range(len(ordered_courses) - 1):
        c_t = ordered_courses[i]
        c_tp1 = ordered_courses[i + 1]
        pair = (c_t, c_tp1)
        pair_key = f"{c_t}->{c_tp1}"

        d_t = course_data[c_t]
        d_tp1 = course_data[c_tp1]

        if mode == "centroid":
            idx_t = d_t["indices_full"]
            idx_tp1 = d_tp1["indices_full"]
        else:
            idx_t = d_t["indices_used"]
            idx_tp1 = d_tp1["indices_used"]

        if idx_t.size == 0 or idx_tp1.size == 0:
            warnings.warn(
                f"Skipping pair {pair_key}: one side has no valid cells "
                f"(n_source={idx_t.size}, n_target={idx_tp1.size})."
            )
            continue

        x_t = embedding[idx_t]
        x_tp1 = embedding[idx_tp1]
        states_t = state_values[idx_t]
        states_tp1 = state_values[idx_tp1]

        if mode == "centroid":
            cent_t, counts_t, labels_t = _compute_centroids(
                x_t, states_t, ordered_states=global_state_order
            )
            cent_tp1, counts_tp1, labels_tp1 = _compute_centroids(
                x_tp1, states_tp1, ordered_states=global_state_order
            )

            if len(labels_t) < 2 or len(labels_tp1) < 2:
                warnings.warn(
                    f"Centroid mode for {pair_key} has very few states "
                    f"(n_source_states={len(labels_t)}, n_target_states={len(labels_tp1)})."
                )

            if cent_t.shape[0] == 0 or cent_tp1.shape[0] == 0:
                warnings.warn(
                    f"Skipping pair {pair_key}: no centroid states available after filtering."
                )
                continue

            M, cost_scale = _prepare_cost_matrix(
                cent_t,
                cent_tp1,
                metric=metric,
                rescale_by_positive_median=rescale_cost_by_median,
            )

            a = _make_weights(counts_t, method=method, unbalanced_mass_mode=unbalanced_mass_mode)
            b = _make_weights(
                counts_tp1, method=method, unbalanced_mass_mode=unbalanced_mass_mode
            )

            plan = _run_ot_solver(a, b, M, method=method, reg=reg, reg_m=reg_m)

            row_labels = list(global_state_order) if include_all_states else list(labels_t)
            col_labels = list(global_state_order) if include_all_states else list(labels_tp1)

            if include_all_states:
                state_mass = _embed_matrix(
                    plan,
                    source_labels=labels_t,
                    target_labels=labels_tp1,
                    full_source_labels=row_labels,
                    full_target_labels=col_labels,
                )
            else:
                state_mass = plan

            transition_df = pd.DataFrame(
                _normalize_transition(state_mass),
                index=pd.Index(row_labels, name=f"{state_key}_t"),
                columns=pd.Index(col_labels, name=f"{state_key}_tp1"),
            )

            if return_plans:
                plans[pair] = {
                    "plan": plan,
                    "cost_matrix": M,
                    "a": a,
                    "b": b,
                    "source_labels": list(labels_t),
                    "target_labels": list(labels_tp1),
                }

            metadata_uns[pair_key] = {
                "course_t": c_t,
                "course_tp1": c_tp1,
                "source_cells": int(idx_t.size),
                "target_cells": int(idx_tp1.size),
                "source_states": list(labels_t),
                "target_states": list(labels_tp1),
                "n_source_states": int(len(labels_t)),
                "n_target_states": int(len(labels_tp1)),
                "cost_median_scale": float(cost_scale),
                "plan_total_mass": float(plan.sum()),
                "state_mass_total": float(state_mass.sum()),
                "sampled_source": bool(d_t["sampled"]),
                "sampled_target": bool(d_tp1["sampled"]),
            }

        else:
            warnings.warn(
                "Running OT in cell mode. This can be memory-intensive for large courses."
            )

            n_t, n_tp1 = x_t.shape[0], x_tp1.shape[0]
            approx_gb = (n_t * n_tp1 * 8) / (1024**3)
            if approx_gb > 1.0:
                warnings.warn(
                    f"Estimated dense cost matrix size for {pair_key}: ~{approx_gb:.2f} GB. "
                    "Consider reducing max_cells_per_course."
                )

            M, cost_scale = _prepare_cost_matrix(
                x_t.astype(np.float32, copy=False),
                x_tp1.astype(np.float32, copy=False),
                metric=metric,
                rescale_by_positive_median=rescale_cost_by_median,
            )

            counts_t = np.ones(n_t, dtype=np.float64)
            counts_tp1 = np.ones(n_tp1, dtype=np.float64)
            a = _make_weights(counts_t, method=method, unbalanced_mass_mode=unbalanced_mass_mode)
            b = _make_weights(
                counts_tp1, method=method, unbalanced_mass_mode=unbalanced_mass_mode
            )

            plan = _run_ot_solver(a, b, M, method=method, reg=reg, reg_m=reg_m)

            present_src = set(pd.Index(states_t).unique().tolist())
            present_tgt = set(pd.Index(states_tp1).unique().tolist())
            if include_all_states:
                row_labels = list(global_state_order)
                col_labels = list(global_state_order)
            else:
                row_labels = [s for s in global_state_order if s in present_src]
                col_labels = [s for s in global_state_order if s in present_tgt]

            state_mass = _aggregate_cell_plan_to_states(
                plan,
                source_states=states_t,
                target_states=states_tp1,
                source_order=row_labels,
                target_order=col_labels,
            )

            transition_df = pd.DataFrame(
                _normalize_transition(state_mass),
                index=pd.Index(row_labels, name=f"{state_key}_t"),
                columns=pd.Index(col_labels, name=f"{state_key}_tp1"),
            )

            if return_plans:
                plans[pair] = {
                    "plan": plan,
                    "cost_matrix": M,
                    "a": a,
                    "b": b,
                    "source_indices": idx_t.copy(),
                    "target_indices": idx_tp1.copy(),
                    "source_states": states_t.copy(),
                    "target_states": states_tp1.copy(),
                }

            metadata_uns[pair_key] = {
                "course_t": c_t,
                "course_tp1": c_tp1,
                "source_cells": int(idx_t.size),
                "target_cells": int(idx_tp1.size),
                "source_states": list(row_labels),
                "target_states": list(col_labels),
                "n_source_states": int(len(row_labels)),
                "n_target_states": int(len(col_labels)),
                "cost_median_scale": float(cost_scale),
                "plan_total_mass": float(plan.sum()),
                "state_mass_total": float(state_mass.sum()),
                "sampled_source": bool(d_t["sampled"]),
                "sampled_target": bool(d_tp1["sampled"]),
            }

        transitions[pair] = transition_df
        mode_uns[pair_key] = transition_df

    if return_plans:
        return transitions, plans
    return transitions


def plot_transition_heatmap(
    transition_df: pd.DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a transition matrix heatmap using matplotlib only."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    data = np.asarray(transition_df.values, dtype=np.float64)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Transport mass")

    ax.set_xticks(np.arange(transition_df.shape[1]))
    ax.set_yticks(np.arange(transition_df.shape[0]))
    ax.set_xticklabels([str(x) for x in transition_df.columns], rotation=90)
    ax.set_yticklabels([str(x) for x in transition_df.index])
    ax.set_xlabel(str(transition_df.columns.name or "target state"))
    ax.set_ylabel(str(transition_df.index.name or "source state"))
    if title is not None:
        ax.set_title(title)

    return ax


def outgoing_entropy(
    transition_df: pd.DataFrame,
    base: float = 2.0,
) -> pd.Series:
    """Compute outgoing entropy per source state from row-normalized transitions."""
    if base <= 0 or base == 1:
        raise ValueError("base must be > 0 and != 1")

    x = np.asarray(transition_df.values, dtype=np.float64)
    row_sum = x.sum(axis=1, keepdims=True)

    p = np.zeros_like(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(x, row_sum, out=p, where=row_sum > 0)

    logp = np.zeros_like(p)
    mask = p > 0
    logp[mask] = np.log(p[mask]) / np.log(base)

    ent = -np.sum(p * logp, axis=1)
    ent[row_sum.ravel() <= 0] = np.nan

    return pd.Series(ent, index=transition_df.index, name="outgoing_entropy")


def top_k_destinations(
    transition_df: pd.DataFrame,
    k: int = 3,
    normalize_rows: bool = True,
) -> Dict[Any, List[Tuple[Any, float]]]:
    """Return top-k destination states for each source state."""
    if k <= 0:
        raise ValueError("k must be > 0")

    mat = np.asarray(transition_df.values, dtype=np.float64)
    if normalize_rows:
        row_sum = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)

    out: Dict[Any, List[Tuple[Any, float]]] = {}
    columns = list(transition_df.columns)

    for i, src in enumerate(transition_df.index):
        row = mat[i]
        if not np.any(row > 0):
            out[src] = []
            continue

        top_idx = np.argsort(row)[::-1]
        top_pairs: List[Tuple[Any, float]] = []
        for j in top_idx:
            val = float(row[j])
            if val <= 0:
                continue
            top_pairs.append((columns[j], val))
            if len(top_pairs) >= k:
                break
        out[src] = top_pairs

    return out


def unbalanced_mass_diagnostics(
    plan: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    source_labels: Optional[Sequence[Any]] = None,
    target_labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Mass diagnostics for unbalanced OT: compare row/col sums to input marginals.

    Returns a dict with:
    - ``source``: DataFrame with ``a``, ``row_sum``, ``difference``, ``ratio``.
    - ``target``: DataFrame with ``b``, ``col_sum``, ``difference``, ``ratio``.
    - ``summary``: scalar diagnostics including transported mass.
    """
    plan = np.asarray(plan, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if plan.ndim != 2:
        raise ValueError("plan must be a 2D matrix")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D arrays")
    if plan.shape != (a.size, b.size):
        raise ValueError(
            f"Shape mismatch: plan={plan.shape}, a={a.shape}, b={b.shape}"
        )

    row_sum = plan.sum(axis=1)
    col_sum = plan.sum(axis=0)

    if source_labels is None:
        source_labels = [f"source_{i}" for i in range(a.size)]
    if target_labels is None:
        target_labels = [f"target_{j}" for j in range(b.size)]

    source_labels = list(source_labels)
    target_labels = list(target_labels)
    if len(source_labels) != a.size or len(target_labels) != b.size:
        raise ValueError("source_labels/target_labels lengths must match plan dimensions")

    eps = 1e-12
    source_df = pd.DataFrame(
        {
            "a": a,
            "row_sum": row_sum,
            "difference": row_sum - a,
            "ratio": row_sum / (a + eps),
        },
        index=pd.Index(source_labels, name="source"),
    )

    target_df = pd.DataFrame(
        {
            "b": b,
            "col_sum": col_sum,
            "difference": col_sum - b,
            "ratio": col_sum / (b + eps),
        },
        index=pd.Index(target_labels, name="target"),
    )

    summary = {
        "total_a": float(a.sum()),
        "total_b": float(b.sum()),
        "transported_mass": float(plan.sum()),
        "source_l1_deviation": float(np.abs(row_sum - a).sum()),
        "target_l1_deviation": float(np.abs(col_sum - b).sum()),
    }

    return {"source": source_df, "target": target_df, "summary": summary}


def demo_unbalanced_centroid(
    adata: AnnData,
    course_key: str = "course",
    state_key: str = "anno_L2",
    random_state: int = 0,
) -> TransitionDict:
    """Short demo for RRMap: unbalanced centroid OT + summary print + one heatmap."""
    transitions = compute_ot_transitions(
        adata,
        course_key=course_key,
        state_key=state_key,
        method="unbalanced",
        mode="centroid",
        random_state=random_state,
    )

    for (c_t, c_tp1), t_df in transitions.items():
        print(f"\n{c_t} -> {c_tp1}")
        top = top_k_destinations(t_df, k=3, normalize_rows=True)
        for src_state, dst_list in top.items():
            if not dst_list:
                continue
            pretty = ", ".join([f"{dst}:{score:.3f}" for dst, score in dst_list])
            print(f"  {src_state}: {pretty}")

    if transitions:
        first_pair = next(iter(transitions.keys()))
        first_df = transitions[first_pair]
        plot_transition_heatmap(
            first_df,
            title=f"OT transition: {first_pair[0]} -> {first_pair[1]}",
        )
        plt.tight_layout()
        plt.show()

    return transitions


__all__ = [
    "compute_ot_transitions",
    "plot_transition_heatmap",
    "outgoing_entropy",
    "top_k_destinations",
    "unbalanced_mass_diagnostics",
    "demo_unbalanced_centroid",
]
