#!/usr/bin/env python
# coding: utf-8

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import logit
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Helper functions
# ============================================================
def read_csv_flexible(path, use_first_column_as_index=True):
    if use_first_column_as_index:
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path)


def parse_target_series(y_series, target_is_percent=True):
    y = y_series.copy()

    if y.dtype == "O":
        y = y.astype(str).str.replace("%", "", regex=False).str.strip()

    y = pd.to_numeric(y, errors="coerce")

    if y.isna().any():
        raise ValueError("The target column contains values that cannot be converted to numeric.")

    if target_is_percent and (((y < 0) | (y > 100)).any()):
        raise ValueError("The target values must be within the range 0 to 100.")

    return y


def prepare_target_for_correlation(
    y_raw,
    target_is_percent=True,
    apply_logit_transform=False,
    clip_zero_and_hundred=True,
):
    y_num = parse_target_series(y_raw, target_is_percent=target_is_percent)

    info = {
        "target_is_percent": bool(target_is_percent),
        "apply_logit_transform": bool(apply_logit_transform),
        "clip_zero_and_hundred": bool(clip_zero_and_hundred),
    }

    if target_is_percent and apply_logit_transform:
        positive_nonzero = y_num[y_num > 0]
        if len(positive_nonzero) == 0:
            raise ValueError("No positive non-zero values are available to determine the clipping width.")

        min_positive = positive_nonzero.min()
        epsilon_pct = min_positive / 2.0
        lower = epsilon_pct
        upper = 100.0 - epsilon_pct

        if clip_zero_and_hundred:
            y_clipped = y_num.clip(lower=lower, upper=upper)
        else:
            y_clipped = y_num.copy()

        y_corr = pd.Series(
            logit(y_clipped / 100.0),
            index=y_num.index,
            name="target_for_correlation",
        )

        info.update({
            "min_positive_nonzero_percent": float(min_positive),
            "epsilon_percent": float(epsilon_pct),
            "lower_clip_percent": float(lower),
            "upper_clip_percent": float(upper),
        })
        return y_num, y_clipped, y_corr, info

    y_clipped = y_num.copy()
    y_corr = pd.Series(y_clipped.copy(), index=y_num.index, name="target_for_correlation")
    return y_num, y_clipped, y_corr, info


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev

    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def safe_corr(func, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan, np.nan

    try:
        r, p = func(x, y)
    except Exception:
        r, p = np.nan, np.nan

    return r, p


def choose_metric_columns(priority="spearman"):
    if str(priority).lower() == "pearson":
        return "pearson_r", "abs_pearson_r", "pearson_p", "pearson_fdr_bh"
    return "spearman_r", "abs_spearman_r", "spearman_p", "spearman_fdr_bh"


def remove_identical_columns(df):
    """
    Remove columns whose values are exactly identical across all rows.
    When duplicated columns are found, the first column is kept and the later
    columns are removed.
    """
    keep_columns = []
    removed_columns = []
    signatures = {}

    for col in df.columns:
        signature = tuple(df[col].tolist())
        if signature in signatures:
            removed_columns.append(col)
        else:
            signatures[signature] = col
            keep_columns.append(col)

    return df[keep_columns].copy(), removed_columns


def select_top_features_with_descriptor_correlation_constraint(
    X_df,
    ranked_corr_df,
    top_k=None,
    descriptor_corr_threshold=0.85,
):
    """
    Select features in descending order of target correlation.
    A feature is accepted only if its absolute correlation with every
    previously selected feature is below the threshold.
    """
    ranked_features = ranked_corr_df["feature"].tolist()
    descriptor_corr = X_df.corr().abs()

    selected = []
    rejected = []

    for feat in ranked_features:
        if feat not in descriptor_corr.index:
            rejected.append({
                "feature": feat,
                "reason": "Feature is not present in the input table.",
            })
            continue

        keep_flag = True
        conflict_with = None
        conflict_corr = None

        for sel in selected:
            corr_val = descriptor_corr.loc[feat, sel]
            if pd.notna(corr_val) and corr_val >= descriptor_corr_threshold:
                keep_flag = False
                conflict_with = sel
                conflict_corr = corr_val
                break

        if keep_flag:
            selected.append(feat)
        else:
            rejected.append({
                "feature": feat,
                "reason": f"Highly correlated with already selected feature '{conflict_with}'.",
                "conflict_with": conflict_with,
                "descriptor_abs_corr": conflict_corr,
            })

        if top_k is not None and len(selected) >= top_k:
            break

    selected_df = ranked_corr_df[ranked_corr_df["feature"].isin(selected)].copy()
    selected_df["selection_order"] = selected_df["feature"].map({f: i + 1 for i, f in enumerate(selected)})
    selected_df = selected_df.sort_values("selection_order")

    rejected_df = pd.DataFrame(rejected)

    if len(selected) > 0:
        selected_descriptor_corr = descriptor_corr.loc[selected, selected]
    else:
        selected_descriptor_corr = pd.DataFrame()

    return selected, selected_df, rejected_df, selected_descriptor_corr


# ============================================================
# Main workflow
# ============================================================
def run(
    input_x_path,
    input_y_path,
    filtered_output_path="data/input_filtering.csv",
    use_first_column_as_index=True,
    target_column=None,
    target_is_percent=True,
    apply_logit_transform=False,
    clip_zero_and_hundred=True,
    univariate_priority="spearman",
    target_corr_threshold=0.05,
    fdr_threshold=0.05,
    use_fdr_filter=False,
    use_pvalue_filter=False,
    pvalue_threshold=0.05,
    top_k_features=None,
    use_sample_size_as_top_k=True,
    descriptor_corr_threshold=0.85,
    top_n_bar=20,
    top_n_scatter=9,
    output_dir=None,
    verbose=True,
):
    input_x_path = Path(input_x_path)
    input_y_path = Path(input_y_path)
    filtered_output_path = Path(filtered_output_path)

    if not input_x_path.exists():
        raise FileNotFoundError(f"Input X file was not found: {input_x_path}")
    if not input_y_path.exists():
        raise FileNotFoundError(f"Input Y file was not found: {input_y_path}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = Path(f"1_descriptor_target_filter_updated_{run_timestamp}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw_data"
    info_dir = output_dir / "info"
    plot_dir = output_dir / "plots"

    for d in [raw_dir, info_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {output_dir}")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    X = read_csv_flexible(input_x_path, use_first_column_as_index=use_first_column_as_index)
    y_df = read_csv_flexible(input_y_path, use_first_column_as_index=use_first_column_as_index)

    if target_column is None:
        if y_df.shape[1] != 1:
            raise ValueError("The output file contains multiple columns. Please specify target_column.")
        target_column = y_df.columns[0]

    if target_column not in y_df.columns:
        raise ValueError(f"'{target_column}' was not found in the output file.")

    common_index = X.index.intersection(y_df.index)
    if len(common_index) == 0:
        raise ValueError("No common index values were found between input X and input Y.")

    X = X.loc[common_index].copy()
    y_df = y_df.loc[common_index].copy()
    y_raw = y_df[target_column].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("The input X file contains missing values or non-numeric values.")

    # --------------------------------------------------
    # Remove identical descriptor columns
    # --------------------------------------------------
    X, removed_identical_columns = remove_identical_columns(X)

    X.to_csv(raw_dir / "input_loaded.csv", encoding="utf-8-sig")
    y_df.to_csv(raw_dir / "output_loaded.csv", encoding="utf-8-sig")
    pd.DataFrame({"removed_identical_feature": removed_identical_columns}).to_csv(
        info_dir / "removed_identical_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y_num, y_clipped, y_corr, target_info = prepare_target_for_correlation(
        y_raw,
        target_is_percent=target_is_percent,
        apply_logit_transform=apply_logit_transform,
        clip_zero_and_hundred=clip_zero_and_hundred,
    )

    target_df = pd.DataFrame({
        "target_raw_numeric": y_num,
        "target_clipped": y_clipped,
        "target_for_correlation": y_corr,
    })
    target_df.to_csv(info_dir / "target_prepared.csv", encoding="utf-8-sig")

    with open(info_dir / "target_preparation_info.json", "w", encoding="utf-8") as f:
        json.dump(target_info, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"X shape after identical-column removal: {X.shape}")
        print(f"Removed identical descriptor columns: {len(removed_identical_columns)}")
        print(f"y length: {len(y_corr)}")

    # --------------------------------------------------
    # Descriptor-target correlation analysis
    # --------------------------------------------------
    rows = []
    for col in X.columns:
        xvals = X[col].values
        yvals = y_corr.values

        pearson_r, pearson_p = safe_corr(pearsonr, xvals, yvals)
        spearman_r, spearman_p = safe_corr(spearmanr, xvals, yvals)

        rows.append({
            "feature": col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "abs_pearson_r": np.nan if pd.isna(pearson_r) else abs(pearson_r),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_spearman_r": np.nan if pd.isna(spearman_r) else abs(spearman_r),
        })

    corr_df = pd.DataFrame(rows)
    corr_df["pearson_fdr_bh"] = benjamini_hochberg(corr_df["pearson_p"].fillna(1.0).values)
    corr_df["spearman_fdr_bh"] = benjamini_hochberg(corr_df["spearman_p"].fillna(1.0).values)

    metric_col, abs_metric_col, p_col, fdr_col = choose_metric_columns(univariate_priority)

    corr_df = corr_df.sort_values(abs_metric_col, ascending=False)
    corr_df.to_csv(info_dir / "descriptor_target_correlations.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------
    # Candidate extraction and final feature selection
    # --------------------------------------------------
    candidate_mask = corr_df[abs_metric_col] >= target_corr_threshold

    if use_fdr_filter:
        candidate_mask = candidate_mask & (corr_df[fdr_col] < fdr_threshold)

    if use_pvalue_filter:
        candidate_mask = candidate_mask & (corr_df[p_col] < pvalue_threshold)

    candidate_df = corr_df[candidate_mask].copy().sort_values(abs_metric_col, ascending=False)
    candidate_df.to_csv(info_dir / "candidate_features_by_target_correlation.csv", index=False, encoding="utf-8-sig")

    if use_sample_size_as_top_k:
        resolved_top_k = len(X)
    else:
        resolved_top_k = top_k_features

    if resolved_top_k is not None:
        ranked_candidate_df = candidate_df.head(resolved_top_k).copy()
    else:
        ranked_candidate_df = candidate_df.copy()

    ranked_candidate_df.to_csv(info_dir / "ranked_candidate_features.csv", index=False, encoding="utf-8-sig")

    selected_feature_names, selected_df, rejected_df_by_descriptor_corr, selected_descriptor_corr = (
        select_top_features_with_descriptor_correlation_constraint(
            X_df=X,
            ranked_corr_df=ranked_candidate_df,
            top_k=resolved_top_k,
            descriptor_corr_threshold=descriptor_corr_threshold,
        )
    )

    removed_df = corr_df[~corr_df["feature"].isin(selected_feature_names)].copy()

    selected_df.to_csv(info_dir / "selected_features_final.csv", index=False, encoding="utf-8-sig")
    removed_df.to_csv(info_dir / "removed_features_final.csv", index=False, encoding="utf-8-sig")
    rejected_df_by_descriptor_corr.to_csv(
        info_dir / "rejected_by_descriptor_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"selected_feature": selected_feature_names}).to_csv(
        info_dir / "selected_feature_names.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"removed_feature": removed_df["feature"].tolist()}).to_csv(
        info_dir / "removed_feature_names.csv",
        index=False,
        encoding="utf-8-sig",
    )

    X_selected = X[selected_feature_names].copy()
    X_selected.to_csv(info_dir / "input_filtered_final.csv", encoding="utf-8-sig")

    filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
    X_selected.to_csv(filtered_output_path, encoding="utf-8-sig")

    if isinstance(selected_descriptor_corr, pd.DataFrame) and len(selected_descriptor_corr) > 0:
        selected_descriptor_corr.to_csv(
            info_dir / "selected_features_descriptor_correlation_matrix.csv",
            encoding="utf-8-sig",
        )

    filter_info = {
        "UNIVARIATE_PRIORITY": univariate_priority,
        "TARGET_CORR_THRESHOLD": target_corr_threshold,
        "USE_FDR_FILTER": use_fdr_filter,
        "FDR_THRESHOLD": fdr_threshold,
        "USE_PVALUE_FILTER": use_pvalue_filter,
        "PVALUE_THRESHOLD": pvalue_threshold,
        "TOP_K_FEATURES": top_k_features,
        "USE_SAMPLE_SIZE_AS_TOP_K": use_sample_size_as_top_k,
        "resolved_top_k": resolved_top_k,
        "DESCRIPTOR_CORR_THRESHOLD": descriptor_corr_threshold,
        "n_removed_identical_features": int(len(removed_identical_columns)),
        "n_candidate_features": int(len(candidate_df)),
        "n_ranked_candidate_features": int(len(ranked_candidate_df)),
        "n_selected_features": int(len(selected_df)),
        "n_removed_features": int(len(removed_df)),
    }

    with open(info_dir / "target_correlation_filter_info.json", "w", encoding="utf-8") as f:
        json.dump(filter_info, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"Number of candidate descriptors: {len(candidate_df)}")
        print(f"Number of ranked candidate descriptors: {len(ranked_candidate_df)}")
        print(f"Number of finally selected descriptors: {len(selected_df)}")
        print(f"Number of finally removed descriptors: {len(removed_df)}")
        print(f"Saved filtered input file: {filtered_output_path}")

    # --------------------------------------------------
    # Bar plots for top descriptors
    # --------------------------------------------------
    top_df = corr_df.head(top_n_bar).copy().iloc[::-1]

    plt.figure(figsize=(8, max(5, 0.35 * len(top_df))))
    plt.barh(top_df["feature"], top_df["spearman_r"])
    plt.xlabel("Spearman r")
    plt.ylabel("Feature")
    plt.title("Top descriptor-target correlations (Spearman)")
    plt.tight_layout()
    plt.savefig(plot_dir / "top_spearman_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, max(5, 0.35 * len(top_df))))
    plt.barh(top_df["feature"], top_df["pearson_r"])
    plt.xlabel("Pearson r")
    plt.ylabel("Feature")
    plt.title("Top descriptor-target correlations (Pearson)")
    plt.tight_layout()
    plt.savefig(plot_dir / "top_pearson_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print("Saved bar plots.")

    # --------------------------------------------------
    # Scatter plots for top descriptors
    # --------------------------------------------------
    top_scatter_features = corr_df.head(top_n_scatter)["feature"].tolist()

    n = len(top_scatter_features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, top_scatter_features):
        ax.scatter(X[feat], y_corr, alpha=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel("Target for correlation")
        row = corr_df[corr_df["feature"] == feat].iloc[0]
        ax.set_title(f"Spearman={row['spearman_r']:.2f}\nPearson={row['pearson_r']:.2f}")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(plot_dir / "top_descriptor_scatterplots.png", dpi=300, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"top_scatter_features": top_scatter_features}).to_csv(
        info_dir / "top_scatter_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if verbose:
        print("Saved scatter plots.")
        print("Analysis completed.")

    return {
        "output_dir": str(output_dir),
        "info_dir": str(info_dir),
        "plot_dir": str(plot_dir),
        "filtered_output_path": str(filtered_output_path),
        "removed_identical_columns": removed_identical_columns,
        "n_candidate_features": int(len(candidate_df)),
        "n_selected_features": int(len(selected_df)),
        "selected_feature_names": selected_feature_names,
    }


if __name__ == "__main__":
    run(
        input_x_path="data/input.csv",
        input_y_path="data/output.csv",
        filtered_output_path="data/input_filtering.csv",
    )
