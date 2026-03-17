#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Default settings
# ==========================================================
USE_FIRST_COLUMN_AS_INDEX = True
TARGET_COLUMN = None

TARGET_TRANSFORM_METHOD = "logit"   # "none", "logit", "arcsine"
CLIP_ZERO_AND_HUNDRED = True
CORR_THRESHOLD = 0.85
REMOVE_NEAR_CONSTANT = True
NEAR_CONSTANT_STD_THRESHOLD = 1e-12

# Range of the number of additional descriptors
MIN_ADD = 1
MAX_ADD = 2

# Alpha candidates for LassoCV
ALPHAS = np.logspace(-4, 1, 80)
MAX_ITER = 100000
RANDOM_STATE = 42

# Multiprocessing settings
N_PROCESSES = 12
START_METHOD = "spawn"
CHUNKSIZE = 32
VERBOSE_EVERY = 1000

# Avoid nested BLAS parallelism
FORCE_SINGLE_THREAD_BLAS = True

# Output settings
SAVE_ALL_RESULTS = True
TOP_N_TO_SAVE = 500
FINAL_REFIT_TOP_N = 20


# ==========================================================
# Worker shared globals
# ==========================================================
_G_BASE_X: pd.DataFrame | None = None
_G_EXTRA_X: pd.DataFrame | None = None
_G_Y_TRANS: pd.Series | None = None
_G_Y_PCT_CLIPPED: pd.Series | None = None


# ==========================================================
# Basic functions
# ==========================================================
def read_csv_flexible(path: str | Path, use_first_column_as_index: bool = True) -> pd.DataFrame:
    if use_first_column_as_index:
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path)


def _safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).eps
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _safe_expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def parse_percentage_series(y_series: pd.Series) -> pd.Series:
    y = y_series.copy()
    if y.dtype == "O":
        y = y.astype(str).str.replace("%", "", regex=False).str.strip()
    y = pd.to_numeric(y, errors="coerce")
    return y


def prepare_target_percent(
    y_raw: pd.Series,
    target_transform_method: str = "logit",
    clip_zero_and_hundred: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    y_pct = parse_percentage_series(y_raw)

    if y_pct.isna().any():
        raise ValueError("The target contains values that cannot be converted to numeric.")
    if ((y_pct < 0) | (y_pct > 100)).any():
        raise ValueError("The target must be within the range 0 to 100.")

    positive_nonzero = y_pct[y_pct > 0]
    if len(positive_nonzero) == 0:
        raise ValueError("No positive non-zero values are available to determine the clipping width.")

    min_positive = float(positive_nonzero.min())
    epsilon_pct = min_positive / 2.0
    lower = epsilon_pct
    upper = 100.0 - epsilon_pct

    y_pct_clipped = y_pct.clip(lower=lower, upper=upper) if clip_zero_and_hundred else y_pct.copy()
    y_fraction = (y_pct_clipped / 100.0).astype(float)

    method = str(target_transform_method).lower()
    if method == "logit":
        y_trans = pd.Series(_safe_logit(y_fraction.to_numpy()), index=y_pct.index, name="y_transformed")
    elif method == "arcsine":
        y_trans = pd.Series(np.arcsin(np.sqrt(y_fraction.to_numpy())), index=y_pct.index, name="y_transformed")
    elif method == "none":
        y_trans = pd.Series(y_pct_clipped.copy(), index=y_pct.index, name="y_transformed")
    else:
        raise ValueError("TARGET_TRANSFORM_METHOD must be one of: 'none', 'logit', or 'arcsine'.")

    clip_info = {
        "target_transform_method": method,
        "clip_zero_and_hundred": bool(clip_zero_and_hundred),
        "min_positive_nonzero_percent": min_positive,
        "epsilon_percent": float(epsilon_pct),
        "lower_clip_percent": float(lower),
        "upper_clip_percent": float(upper),
        "num_zero_raw": int((y_pct == 0).sum()),
        "num_hundred_raw": int((y_pct == 100).sum()),
    }
    return y_pct, y_pct_clipped, y_trans, clip_info


def inverse_transform_to_percent(y_trans_array: np.ndarray, target_transform_method: str = "logit") -> np.ndarray:
    y_trans_array = np.asarray(y_trans_array, dtype=float)
    method = str(target_transform_method).lower()
    if method == "logit":
        return _safe_expit(y_trans_array) * 100.0
    if method == "arcsine":
        return (np.sin(y_trans_array) ** 2) * 100.0
    if method == "none":
        return y_trans_array
    raise ValueError("TARGET_TRANSFORM_METHOD must be one of: 'none', 'logit', or 'arcsine'.")


def remove_near_constant_features(X_df: pd.DataFrame, std_threshold: float = 1e-12):
    stds = X_df.std(axis=0, ddof=0)
    remove_cols = stds[stds <= std_threshold].index.tolist()
    X_out = X_df.drop(columns=remove_cols, errors="ignore")
    return X_out, remove_cols, stds


def remove_high_correlation_features(X_df: pd.DataFrame, threshold: float = 0.85):
    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    for col in upper.columns:
        if any(upper[col] >= threshold):
            to_drop.append(col)
    X_reduced = X_df.drop(columns=to_drop, errors="ignore")
    return X_reduced, to_drop, corr


def compute_metrics(y_true_pct: Sequence[float], y_pred_pct: Sequence[float]) -> dict:
    y_true_pct = np.asarray(y_true_pct, dtype=float)
    y_pred_pct = np.asarray(y_pred_pct, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_pct, y_pred_pct))),
        "MSE": float(mean_squared_error(y_true_pct, y_pred_pct)),
        "R2": float(r2_score(y_true_pct, y_pred_pct)),
    }


@dataclass
class SearchResult:
    n_added: int
    added_features: str
    loocv_rmse: float
    loocv_mse: float
    loocv_r2: float
    mean_alpha: float
    median_alpha: float
    n_features_after_filter_mean: float
    n_features_after_filter_min: int
    n_features_after_filter_max: int
    selection_frequency: str


# ==========================================================
# Data loading and alignment
# ==========================================================
def load_and_prepare_data(
    base_x_path: str | Path,
    extra_x_path: str | Path,
    y_path: str | Path,
    use_first_column_as_index: bool = True,
    target_column: str | None = None,
    target_transform_method: str = "logit",
    clip_zero_and_hundred: bool = True,
):
    base_x = read_csv_flexible(base_x_path, use_first_column_as_index)
    extra_x = read_csv_flexible(extra_x_path, use_first_column_as_index)
    y_df = read_csv_flexible(y_path, use_first_column_as_index)

    if target_column is None:
        if y_df.shape[1] != 1:
            raise ValueError("Please specify target_column because the output file contains multiple columns.")
        target_column = y_df.columns[0]

    if target_column not in y_df.columns:
        raise ValueError(f"TARGET_COLUMN={target_column} was not found in the output file.")

    common_index = base_x.index.intersection(extra_x.index).intersection(y_df.index)
    if len(common_index) == 0:
        raise ValueError("No common index values were found among base_x, extra_x, and y.")

    base_x = base_x.loc[common_index].copy().apply(pd.to_numeric, errors="coerce")
    extra_x = extra_x.loc[common_index].copy().apply(pd.to_numeric, errors="coerce")
    y_raw = y_df.loc[common_index, target_column].copy()

    if base_x.isna().any().any():
        raise ValueError("The base input file contains missing values or non-numeric values.")
    if extra_x.isna().any().any():
        raise ValueError("The extra descriptor file contains missing values or non-numeric values.")

    duplicate_cols = [c for c in extra_x.columns if c in base_x.columns]
    if duplicate_cols:
        extra_x = extra_x.drop(columns=duplicate_cols)

    y_pct, y_pct_clipped, y_trans, clip_info = prepare_target_percent(
        y_raw,
        target_transform_method=target_transform_method,
        clip_zero_and_hundred=clip_zero_and_hundred,
    )

    return base_x, extra_x, y_pct, y_pct_clipped, y_trans, clip_info, duplicate_cols, target_column


# ==========================================================
# Multiprocessing initialization
# ==========================================================
def _set_blas_single_thread_env() -> None:
    if FORCE_SINGLE_THREAD_BLAS:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def init_worker(
    base_x: pd.DataFrame,
    extra_x: pd.DataFrame,
    y_trans: pd.Series,
    y_pct_clipped: pd.Series,
) -> None:
    global _G_BASE_X, _G_EXTRA_X, _G_Y_TRANS, _G_Y_PCT_CLIPPED
    _set_blas_single_thread_env()
    _G_BASE_X = base_x
    _G_EXTRA_X = extra_x
    _G_Y_TRANS = y_trans
    _G_Y_PCT_CLIPPED = y_pct_clipped


# ==========================================================
# Evaluate one combination
# ==========================================================
def evaluate_one_combination(
    add_features: tuple[str, ...],
    base_x: pd.DataFrame,
    extra_x: pd.DataFrame,
    y_trans: pd.Series,
    y_pct_clipped: pd.Series,
    target_transform_method: str = "logit",
    corr_threshold: float = 0.85,
    remove_near_constant: bool = True,
    near_constant_std_threshold: float = 1e-12,
):
    X_all = pd.concat([base_x, extra_x[list(add_features)]], axis=1)
    loo = LeaveOneOut()

    obs_pct_list: list[float] = []
    pred_pct_list: list[float] = []
    alphas_used: list[float] = []
    n_features_after_filter_list: list[int] = []
    selected_feature_counter: dict[str, int] = {}

    for train_idx, test_idx in loo.split(X_all):
        X_train = X_all.iloc[train_idx].copy()
        X_test = X_all.iloc[test_idx].copy()
        y_train = y_trans.iloc[train_idx].copy()
        y_test_pct = float(y_pct_clipped.iloc[test_idx].values[0])

        if remove_near_constant:
            X_train, _, _ = remove_near_constant_features(
                X_train, std_threshold=near_constant_std_threshold
            )

        X_train, _, _ = remove_high_correlation_features(X_train, threshold=corr_threshold)

        if X_train.shape[1] == 0:
            return None

        X_test = X_test[X_train.columns].copy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LassoCV(
            alphas=ALPHAS,
            cv=LeaveOneOut(),
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X_train_scaled, y_train)

        y_pred_trans = float(np.asarray(model.predict(X_test_scaled)).reshape(-1)[0])
        y_pred_pct = float(
            inverse_transform_to_percent(
                np.array([y_pred_trans]),
                target_transform_method=target_transform_method,
            )[0]
        )

        obs_pct_list.append(y_test_pct)
        pred_pct_list.append(y_pred_pct)
        alphas_used.append(float(model.alpha_))
        n_features_after_filter_list.append(int(X_train.shape[1]))

        nz = np.asarray(model.coef_).reshape(-1) != 0
        for col in X_train.columns[nz]:
            selected_feature_counter[col] = selected_feature_counter.get(col, 0) + 1

    metrics = compute_metrics(obs_pct_list, pred_pct_list)
    selection_frequency = {
        k: int(v) for k, v in sorted(selected_feature_counter.items(), key=lambda x: (-x[1], x[0]))
    }

    return SearchResult(
        n_added=len(add_features),
        added_features=";".join(add_features),
        loocv_rmse=metrics["RMSE"],
        loocv_mse=metrics["MSE"],
        loocv_r2=metrics["R2"],
        mean_alpha=float(np.mean(alphas_used)),
        median_alpha=float(np.median(alphas_used)),
        n_features_after_filter_mean=float(np.mean(n_features_after_filter_list)),
        n_features_after_filter_min=int(np.min(n_features_after_filter_list)),
        n_features_after_filter_max=int(np.max(n_features_after_filter_list)),
        selection_frequency=json.dumps(selection_frequency, ensure_ascii=False),
    )


def worker_evaluate_combination(add_features: tuple[str, ...]):
    if _G_BASE_X is None or _G_EXTRA_X is None or _G_Y_TRANS is None or _G_Y_PCT_CLIPPED is None:
        raise RuntimeError("The worker process was not initialized properly.")
    return evaluate_one_combination(
        add_features=add_features,
        base_x=_G_BASE_X,
        extra_x=_G_EXTRA_X,
        y_trans=_G_Y_TRANS,
        y_pct_clipped=_G_Y_PCT_CLIPPED,
        target_transform_method=TARGET_TRANSFORM_METHOD,
        corr_threshold=CORR_THRESHOLD,
        remove_near_constant=REMOVE_NEAR_CONSTANT,
        near_constant_std_threshold=NEAR_CONSTANT_STD_THRESHOLD,
    )


# ==========================================================
# Exhaustive search
# ==========================================================
def build_candidate_combinations(extra_columns: Sequence[str], min_add: int, max_add: int) -> Iterator[tuple[str, ...]]:
    for r in range(min_add, max_add + 1):
        yield from itertools.combinations(extra_columns, r)


def count_candidate_combinations(n_extra: int, min_add: int, max_add: int) -> int:
    return int(sum(math.comb(n_extra, r) for r in range(min_add, max_add + 1)))


def exhaustive_search(base_x, extra_x, y_trans, y_pct_clipped, min_add: int, max_add: int, n_processes: int, chunksize: int, verbose: bool = True):
    _set_blas_single_thread_env()
    extra_columns = list(extra_x.columns)
    total_combs = count_candidate_combinations(len(extra_columns), min_add, max_add)

    if verbose:
        print(f"Number of candidate extra descriptors: {len(extra_columns)}")
        print(f"Total number of combinations to search: {total_combs}")
        print(f"Multiprocessing start method: {START_METHOD}")
        print(f"Number of processes: {n_processes}")
        print(f"Chunksize: {chunksize}")

    ctx = mp.get_context(START_METHOD)
    comb_iter = build_candidate_combinations(extra_columns, min_add, max_add)

    results: list[SearchResult] = []
    with ctx.Pool(
        processes=n_processes,
        initializer=init_worker,
        initargs=(base_x, extra_x, y_trans, y_pct_clipped),
    ) as pool:
        for idx, result in enumerate(pool.imap_unordered(worker_evaluate_combination, comb_iter, chunksize=chunksize), start=1):
            if result is not None:
                results.append(result)
            if verbose and VERBOSE_EVERY > 0 and (idx % VERBOSE_EVERY == 0 or idx == total_combs):
                print(f"Progress: {idx}/{total_combs} ({idx / total_combs:.1%})")

    df = pd.DataFrame([asdict(r) for r in results])
    df = df.sort_values(["loocv_rmse", "loocv_r2"], ascending=[True, False]).reset_index(drop=True)
    return df


# ==========================================================
# Full refit for top-ranked combinations
# ==========================================================
def fit_final_model_for_feature_set(
    feature_set: list[str],
    X_all: pd.DataFrame,
    y_trans: pd.Series,
    y_pct_clipped: pd.Series,
    target_transform_method: str = "logit",
    corr_threshold: float = 0.85,
    remove_near_constant: bool = True,
    near_constant_std_threshold: float = 1e-12,
):
    X_fit = X_all[feature_set].copy()

    if remove_near_constant:
        X_fit, removed_const, _ = remove_near_constant_features(X_fit, std_threshold=near_constant_std_threshold)
    else:
        removed_const = []

    X_fit, removed_corr, _ = remove_high_correlation_features(X_fit, threshold=corr_threshold)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)

    model = LassoCV(
        alphas=ALPHAS,
        cv=LeaveOneOut(),
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_scaled, y_trans)

    y_pred_trans = np.asarray(model.predict(X_scaled)).reshape(-1)
    y_pred_pct = inverse_transform_to_percent(y_pred_trans, target_transform_method)
    metrics = compute_metrics(y_pct_clipped.values, y_pred_pct)

    coef_df = pd.DataFrame({
        "feature": X_fit.columns,
        "coefficient": np.asarray(model.coef_).reshape(-1),
        "abs_coefficient": np.abs(np.asarray(model.coef_).reshape(-1)),
    }).sort_values("abs_coefficient", ascending=False)

    return {
        "n_input_features": len(feature_set),
        "n_features_after_filter": int(X_fit.shape[1]),
        "alpha": float(model.alpha_),
        "fullfit_rmse": metrics["RMSE"],
        "fullfit_r2": metrics["R2"],
        "removed_constant": removed_const,
        "removed_high_corr": removed_corr,
        "kept_features": X_fit.columns.tolist(),
        "coefficients": coef_df,
    }


# ==========================================================
# Main workflow
# ==========================================================
def run(
    base_x_path="data/input_lasso_nonzero.csv",
    y_path="data/output.csv",
    output_dir="4_results_lasso_exhaustive_search_multi",
    ranked1_output_path="data/input_LassoCV_added.csv",
    extra_x_path="data/input.csv",
    n_processes=N_PROCESSES,
    chunksize=CHUNKSIZE,
    use_first_column_as_index=USE_FIRST_COLUMN_AS_INDEX,
    target_column=TARGET_COLUMN,
    target_transform_method=TARGET_TRANSFORM_METHOD,
    clip_zero_and_hundred=CLIP_ZERO_AND_HUNDRED,
    corr_threshold=CORR_THRESHOLD,
    remove_near_constant=REMOVE_NEAR_CONSTANT,
    near_constant_std_threshold=NEAR_CONSTANT_STD_THRESHOLD,
    min_add=MIN_ADD,
    max_add=MAX_ADD,
    save_all_results=SAVE_ALL_RESULTS,
    top_n_to_save=TOP_N_TO_SAVE,
    final_refit_top_n=FINAL_REFIT_TOP_N,
    verbose=True,
):
    base_x_path = Path(base_x_path)
    extra_x_path = Path(extra_x_path)
    y_path = Path(y_path)
    output_dir = Path(output_dir)
    ranked1_output_path = Path(ranked1_output_path)

    if not base_x_path.exists():
        raise FileNotFoundError(f"The base input file was not found: {base_x_path}")
    if not extra_x_path.exists():
        raise FileNotFoundError(f"The extra descriptor file was not found: {extra_x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"The output file was not found: {y_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    (
        base_x,
        extra_x,
        y_pct,
        y_pct_clipped,
        y_trans,
        clip_info,
        duplicate_cols,
        resolved_target_column,
    ) = load_and_prepare_data(
        base_x_path=base_x_path,
        extra_x_path=extra_x_path,
        y_path=y_path,
        use_first_column_as_index=use_first_column_as_index,
        target_column=target_column,
        target_transform_method=target_transform_method,
        clip_zero_and_hundred=clip_zero_and_hundred,
    )

    meta = {
        "base_x_path": str(base_x_path),
        "extra_x_path": str(extra_x_path),
        "y_path": str(y_path),
        "ranked1_output_path": str(ranked1_output_path),
        "target_column": resolved_target_column,
        "n_samples": int(len(base_x)),
        "n_base_features": int(base_x.shape[1]),
        "n_extra_features": int(extra_x.shape[1]),
        "base_features": list(base_x.columns),
        "duplicate_columns_removed_from_extra": duplicate_cols,
        "min_add": int(min_add),
        "max_add": int(max_add),
        "corr_threshold": float(corr_threshold),
        "remove_near_constant": bool(remove_near_constant),
        "near_constant_std_threshold": float(near_constant_std_threshold),
        "alphas": [float(a) for a in ALPHAS],
        "target_transform": target_transform_method,
        "clip_info": clip_info,
        "n_processes": int(N_PROCESSES),
        "start_method": START_METHOD,
        "chunksize": int(CHUNKSIZE),
        "force_single_thread_blas": bool(FORCE_SINGLE_THREAD_BLAS),
        "search_space_size": count_candidate_combinations(extra_x.shape[1], min_add, max_add),
    }
    with open(output_dir / "search_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    results_df = exhaustive_search(
        base_x=base_x,
        extra_x=extra_x,
        y_trans=y_trans,
        y_pct_clipped=y_pct_clipped,
        min_add=min_add,
        max_add=max_add,
        n_processes=n_processes,
        chunksize=chunksize,
        verbose=verbose,
    )

    if save_all_results:
        results_df.to_csv(output_dir / "all_combinations_loocv_results.csv", index=False, encoding="utf-8-sig")

    top_df = results_df.head(top_n_to_save).copy()
    top_df.to_csv(output_dir / f"top_{top_n_to_save}_combinations.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    X_total = pd.concat([base_x, extra_x], axis=1)

    rank1_feature_set_saved = False

    for rank, row in top_df.head(final_refit_top_n).iterrows():
        add_features = row["added_features"].split(";") if row["added_features"] else []
        feature_set = list(base_x.columns) + add_features

        final_info = fit_final_model_for_feature_set(
            feature_set=feature_set,
            X_all=X_total,
            y_trans=y_trans,
            y_pct_clipped=y_pct_clipped,
            target_transform_method=target_transform_method,
            corr_threshold=corr_threshold,
            remove_near_constant=remove_near_constant,
            near_constant_std_threshold=near_constant_std_threshold,
        )

        coef_path = output_dir / f"rank_{rank+1:03d}_coefficients.csv"
        final_info["coefficients"].to_csv(coef_path, index=False, encoding="utf-8-sig")

        kept_feature_csv = output_dir / f"rank_{rank+1:03d}_kept_features_after_filter.csv"
        X_total[final_info["kept_features"]].to_csv(kept_feature_csv, encoding="utf-8-sig")

        summary_rows.append({
            "rank": rank + 1,
            "added_features": row["added_features"],
            "loocv_rmse": row["loocv_rmse"],
            "loocv_r2": row["loocv_r2"],
            "fullfit_rmse": final_info["fullfit_rmse"],
            "fullfit_r2": final_info["fullfit_r2"],
            "alpha": final_info["alpha"],
            "n_features_after_filter": final_info["n_features_after_filter"],
            "kept_features": ";".join(final_info["kept_features"]),
            "removed_constant": ";".join(final_info["removed_constant"]),
            "removed_high_corr": ";".join(final_info["removed_high_corr"]),
            "coefficient_file": coef_path.name,
            "kept_feature_file": kept_feature_csv.name,
        })

        if rank == 0:
            ranked1_output_path.parent.mkdir(parents=True, exist_ok=True)
            X_total[feature_set].to_csv(ranked1_output_path, encoding="utf-8-sig")

            ranked1_kept_path = output_dir / "rank_001_kept_features_after_filter.csv"
            X_total[final_info["kept_features"]].to_csv(ranked1_kept_path, encoding="utf-8-sig")
            rank1_feature_set_saved = True

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "top_models_final_refit_summary.csv", index=False, encoding="utf-8-sig")

    if verbose:
        print("")
        print("Exhaustive search completed.")
        print(f"Results saved to: {output_dir}")
        if rank1_feature_set_saved:
            print(f"Rank-1 feature set saved to: {ranked1_output_path}")

    return {
        "output_dir": str(output_dir),
        "ranked1_output_path": str(ranked1_output_path),
        "results_df": results_df,
        "top_df": top_df,
        "summary_df": summary_df,
    }


if __name__ == "__main__":
    mp.freeze_support()
    run(
        base_x_path="data/input_lasso_nonzero.csv",
        y_path="data/output.csv",
        output_dir="4_results_lasso_exhaustive_search_multi",
        ranked1_output_path="data/input_LassoCV_added.csv",
        extra_x_path="data/input.csv",
    )
