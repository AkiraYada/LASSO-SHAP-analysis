#!/usr/bin/env python
# coding: utf-8

import sys
import json
import ast
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import logit, expit
from scipy.stats import spearmanr, pearsonr

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
import shap

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# --------------------------------------------------
# Default settings
# --------------------------------------------------
USE_FIRST_COLUMN_AS_INDEX = True
TARGET_COLUMN = None

TARGET_TRANSFORM_METHOD = "logit"   # "none", "logit", "arcsine"
CLIP_ZERO_AND_HUNDRED = True

CORR_THRESHOLD = 0.85
REMOVE_NEAR_CONSTANT = True
NEAR_CONSTANT_STD_THRESHOLD = 1e-12

MODELS_TO_RUN = [
    "Lasso",
    "ElasticNet",
    "Ridge",
    "PLS",
]

PARAM_GRIDS = {
    "Lasso": {
        "alpha": np.logspace(-4, 1, 80).tolist(),
    },
    "ElasticNet": {
        "alpha": np.logspace(-4, 1, 80).tolist(),
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
    },
    "Ridge": {
        "alpha": np.logspace(-4, 1, 80).tolist(),
    },
    "PLS": {
        # The final search range is adaptively restricted for small-data PLS.
        "n_components": [1, 2, 3, 4, 5, 6],
    },
}

GRIDSEARCH_SCORING = "neg_root_mean_squared_error"
N_JOBS = -1
VERBOSE = 0

N_BOOTSTRAP = 500
BOOTSTRAP_MODEL = "ElasticNet"

RUN_SHAP = True
SHAP_BACKGROUND_SIZE = 100
SHAP_KERNEL_NSAMPLES = 200
SHAP_MAX_DISPLAY = 15
SHAP_SCATTER_TOP_N = 6
SHAP_WATERFALL_SAMPLE_MODE = "extreme"   # "extreme", "all", "index"
SHAP_WATERFALL_MAX_SAMPLES = 19
SHAP_WATERFALL_SAMPLE_INDEX = None
SHAP_FORCE_TOP_N = 3
SHAP_FORCE_PLOT_MATPLOTLIB = False
SHAP_DEPENDENCE_TOP_N = 6

PAIRPLOT_FEATURES = None
PAIRPLOT_MAX_FEATURES = 6


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def save_json(path, data):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text_lines(path, lines):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def save_library_versions(info_dir):
    import sklearn
    version_info = {
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
    }
    save_json(Path(info_dir) / "library_versions.json", version_info)


def read_csv_flexible(path, use_first_column_as_index=True):
    if use_first_column_as_index:
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path)


def parse_percentage_series(y_series):
    y = y_series.copy()
    if y.dtype == "O":
        y = y.astype(str).str.replace("%", "", regex=False).str.strip()
    y = pd.to_numeric(y, errors="coerce")
    return y


def prepare_target_percent(y_raw, target_transform_method="logit", clip_zero_and_hundred=True):
    y_pct = parse_percentage_series(y_raw)

    if y_pct.isna().any():
        raise ValueError("The target contains values that cannot be converted to numeric.")
    if ((y_pct < 0) | (y_pct > 100)).any():
        raise ValueError("The target must be within the range 0 to 100.")

    positive_nonzero = y_pct[y_pct > 0]
    if len(positive_nonzero) == 0:
        raise ValueError("No positive non-zero values are available to determine the clipping width.")

    min_positive = positive_nonzero.min()
    epsilon_pct = min_positive / 2.0

    lower = epsilon_pct
    upper = 100.0 - epsilon_pct

    y_pct_clipped = y_pct.clip(lower=lower, upper=upper) if clip_zero_and_hundred else y_pct.copy()
    y_fraction = y_pct_clipped / 100.0
    method = str(target_transform_method).lower()

    if method == "logit":
        y_trans = pd.Series(logit(y_fraction), index=y_pct.index, name="y_transformed")
    elif method == "arcsine":
        y_trans = pd.Series(np.arcsin(np.sqrt(y_fraction)), index=y_pct.index, name="y_transformed")
    elif method == "none":
        y_trans = pd.Series(y_pct_clipped.copy(), index=y_pct.index, name="y_transformed")
    else:
        raise ValueError("TARGET_TRANSFORM_METHOD must be one of: 'none', 'logit', or 'arcsine'.")

    clip_info = {
        "target_transform_method": method,
        "clip_zero_and_hundred": bool(clip_zero_and_hundred),
        "min_positive_nonzero_percent": float(min_positive),
        "epsilon_percent": float(epsilon_pct),
        "lower_clip_percent": float(lower),
        "upper_clip_percent": float(upper),
        "num_zero_raw": int((y_pct == 0).sum()),
        "num_hundred_raw": int((y_pct == 100).sum()),
    }
    return y_pct, y_pct_clipped, y_trans, clip_info


def inverse_transform_to_percent(y_trans_array, target_transform_method="logit"):
    y_trans_array = np.asarray(y_trans_array)
    method = str(target_transform_method).lower()

    if method == "logit":
        return expit(y_trans_array) * 100.0
    elif method == "arcsine":
        return (np.sin(y_trans_array) ** 2) * 100.0
    elif method == "none":
        return y_trans_array
    else:
        raise ValueError("TARGET_TRANSFORM_METHOD must be one of: 'none', 'logit', or 'arcsine'.")


def remove_near_constant_features(X_df, std_threshold=1e-12):
    stds = X_df.std(axis=0, ddof=0)
    remove_cols = stds[stds <= std_threshold].index.tolist()
    X_out = X_df.drop(columns=remove_cols, errors="ignore")
    return X_out, remove_cols, stds


def remove_high_correlation_features(X_df, threshold=0.85):
    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    for col in upper.columns:
        if any(upper[col] >= threshold):
            to_drop.append(col)
    X_reduced = X_df.drop(columns=to_drop, errors="ignore")
    return X_reduced, to_drop, corr


def compute_metrics(y_true_pct, y_pred_pct):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_pct, y_pred_pct))),
        "MSE": float(mean_squared_error(y_true_pct, y_pred_pct)),
        "R2": float(r2_score(y_true_pct, y_pred_pct)),
    }


def yy_plot(y_true, y_pred, title, save_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Observed (%)")
    plt.ylabel("Predicted (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def metrics_barplot(model_summary_df, metric_name, save_path):
    df = model_summary_df.copy()
    ascending = False if metric_name == "R2" else True
    df = df.sort_values(metric_name, ascending=ascending)

    x = np.arange(len(df))
    vals = df[metric_name].values

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=45)
    ax.set_ylabel(metric_name)
    ax.set_title(f"LOOCV comparison ({metric_name})")

    span = vals.max() - vals.min() if len(vals) > 0 else 1
    offset = 0.01 * span if span > 0 else 0.01
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_model_dict(random_state=42):
    return {
        "Lasso": Lasso(max_iter=10000, random_state=random_state),
        "ElasticNet": ElasticNet(max_iter=10000, random_state=random_state),
        "Ridge": Ridge(random_state=random_state),
        "PLS": PLSRegression(),
    }


def corrdot(x, y, **kws):
    ax = plt.gca()
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        r = np.nan
    else:
        r = np.corrcoef(x, y)[0, 1]

    ax.set_axis_off()

    if np.isnan(r):
        ax.text(0.5, 0.5, "NaN", ha="center", va="center", fontsize=12)
        return

    size = abs(r) * 4000
    color = plt.cm.coolwarm((r + 1) / 2)

    ax.scatter([0.5], [0.5], s=size, color=color, alpha=0.7)
    ax.text(0.5, 0.5, f"{r:.2f}", ha="center", va="center", fontsize=10, color="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def make_descriptor_pairplot(df, save_path, max_features=6, feature_list=None, figsize_scale=2.2):
    plot_df = df.copy()
    plot_df = plot_df.select_dtypes(include=[np.number])

    if feature_list is not None:
        existing = [f for f in feature_list if f in plot_df.columns]
        plot_df = plot_df[existing].copy()
    else:
        plot_df = plot_df.iloc[:, :max_features].copy()

    if plot_df.shape[1] < 2:
        raise ValueError("At least two descriptors are required for the pair plot.")

    sns.set(style="white", font_scale=0.9)

    g = sns.PairGrid(plot_df, diag_sharey=False)
    g.map_lower(
        sns.regplot,
        scatter_kws={"s": 18, "alpha": 0.8},
        line_kws={"color": "black", "linewidth": 1},
    )
    g.map_diag(sns.histplot, kde=True)
    g.map_upper(corrdot)

    n = plot_df.shape[1]
    g.fig.set_size_inches(figsize_scale * n, figsize_scale * n)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def recommend_pls_component_grid(n_features, n_samples):
    """
    Build a conservative PLS component search range for small datasets.

    Practical rule:
    - Hard upper bound: min(n_features, n_samples - 1)
    - For small datasets, use a stricter cap to reduce overfitting.
      * n_samples <= 20: up to 4 components
      * n_samples <= 30: up to 6 components
      * otherwise: up to 10 components
    """
    hard_upper = min(n_features, n_samples - 1)

    if n_samples <= 20:
        soft_upper = 4
    elif n_samples <= 30:
        soft_upper = 6
    else:
        soft_upper = 10

    final_upper = min(hard_upper, soft_upper)

    if final_upper < 1:
        final_upper = 1

    return list(range(1, final_upper + 1))


def adapt_param_grid(model_name, param_grid, n_features, n_samples):
    grid = dict(param_grid)
    if model_name == "PLS":
        grid["n_components"] = recommend_pls_component_grid(
            n_features=n_features,
            n_samples=n_samples,
        )
    return grid


def build_inner_cv_splits(n_train):
    loo = LeaveOneOut()
    dummy = np.arange(n_train)
    return list(loo.split(dummy))


def select_waterfall_samples(pred_df, mode="extreme", max_samples=3, sample_index=None):
    if pred_df is None or len(pred_df) == 0:
        return []

    df = pred_df.copy()

    if mode == "index" and sample_index is not None:
        selected = df[df["Index"].astype(str) == str(sample_index)]["Index"].tolist()
        return selected[:max_samples]

    if mode == "all":
        return df["Index"].tolist()[:max_samples]

    df["abs_error"] = np.abs(df["Observed_percent"] - df["Predicted_percent"])

    candidates = []
    if len(df) > 0:
        candidates.append(df.sort_values("abs_error", ascending=False).iloc[0]["Index"])
        candidates.append(df.sort_values("Predicted_percent", ascending=False).iloc[0]["Index"])
        candidates.append(df.sort_values("Predicted_percent", ascending=True).iloc[0]["Index"])

    ordered_unique = []
    for idx in candidates + df.sort_values("abs_error", ascending=False)["Index"].tolist():
        if idx not in ordered_unique:
            ordered_unique.append(idx)

    return ordered_unique[:max_samples]


def ensure_shap_explanation(explainer, shap_values, X_values, feature_names):
    base_values = getattr(explainer, "expected_value", None)
    shap_values = np.asarray(shap_values)

    if np.isscalar(base_values) or base_values is None:
        if base_values is None:
            base_values = np.zeros(X_values.shape[0])
        else:
            base_values = np.repeat(float(base_values), X_values.shape[0])
    else:
        base_values = np.asarray(base_values).reshape(-1)
        if len(base_values) == 1:
            base_values = np.repeat(float(base_values[0]), X_values.shape[0])
        elif len(base_values) != X_values.shape[0]:
            base_values = np.repeat(float(np.mean(base_values)), X_values.shape[0])

    return shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=np.asarray(X_values),
        feature_names=feature_names,
    )


# --------------------------------------------------
# Main workflow
# --------------------------------------------------
def run(
    input_x_path="data/input_filtering.csv",
    input_y_path="data/output.csv",
    output_dir=None,
    lasso_nonzero_output_path="data/input_lasso_nonzero.csv",
    use_first_column_as_index=USE_FIRST_COLUMN_AS_INDEX,
    target_column=TARGET_COLUMN,
    target_transform_method=TARGET_TRANSFORM_METHOD,
    clip_zero_and_hundred=CLIP_ZERO_AND_HUNDRED,
    corr_threshold=CORR_THRESHOLD,
    remove_near_constant=REMOVE_NEAR_CONSTANT,
    near_constant_std_threshold=NEAR_CONSTANT_STD_THRESHOLD,
    models_to_run=None,
    bootstrap_model=BOOTSTRAP_MODEL,
    n_bootstrap=N_BOOTSTRAP,
    run_shap=RUN_SHAP,
    verbose=True,
):
    np.random.seed(RANDOM_STATE)

    input_x_path = Path(input_x_path)
    input_y_path = Path(input_y_path)
    lasso_nonzero_output_path = Path(lasso_nonzero_output_path)

    if output_dir is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"3_ML_small_data_analysis_{run_timestamp}")
    else:
        output_dir = Path(output_dir)

    if not input_x_path.exists():
        raise FileNotFoundError(f"Input X file was not found: {input_x_path}")
    if not input_y_path.exists():
        raise FileNotFoundError(f"Input Y file was not found: {input_y_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw_data"
    processed_dir = output_dir / "processed_data"
    eda_dir = output_dir / "eda"
    pred_dir = output_dir / "predictions"
    plot_dir = output_dir / "plots"
    info_dir = output_dir / "info"
    stab_dir = output_dir / "stability"
    shap_dir = output_dir / "shap"

    for d in [raw_dir, processed_dir, eda_dir, pred_dir, plot_dir, info_dir, stab_dir, shap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {output_dir}")

    model_dict = get_model_dict(RANDOM_STATE)
    if models_to_run is None:
        models_to_run = MODELS_TO_RUN.copy()
    models_to_run = [m for m in models_to_run if m in model_dict]

    if verbose:
        print("Models to run:", models_to_run)
        print("PLS component search uses a conservative small-data rule.")

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
        raise ValueError(f"TARGET_COLUMN={target_column} was not found in the output file.")

    common_index = X.index.intersection(y_df.index)
    if len(common_index) == 0:
        raise ValueError("No common index values were found between input.csv and output.csv.")

    X = X.loc[common_index].copy()
    y_df = y_df.loc[common_index].copy()
    y_raw = y_df[target_column].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("input.csv contains non-numeric values or missing values.")

    X.to_csv(raw_dir / "input_loaded.csv", encoding="utf-8-sig")
    y_df.to_csv(raw_dir / "output_loaded.csv", encoding="utf-8-sig")
    pd.DataFrame({"sample_id": X.index}).to_csv(info_dir / "sample_ids_used.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": X.columns}).to_csv(info_dir / "input_feature_names_original.csv", index=False, encoding="utf-8-sig")

    y_pct_raw, y_pct_clipped, y_trans, clip_info = prepare_target_percent(
        y_raw,
        target_transform_method=target_transform_method,
        clip_zero_and_hundred=clip_zero_and_hundred,
    )

    target_transform_df = pd.DataFrame({
        "y_raw_percent": y_pct_raw,
        "y_clipped_percent": y_pct_clipped,
        "y_transformed": y_trans,
    })
    target_transform_df.to_csv(processed_dir / "output_transformation.csv", encoding="utf-8-sig")
    save_json(info_dir / "target_transformation_info.json", clip_info)

    run_config = {
        "INPUT_X_PATH": str(input_x_path),
        "INPUT_Y_PATH": str(input_y_path),
        "OUTPUT_DIR": str(output_dir),
        "LASSO_NONZERO_OUTPUT_PATH": str(lasso_nonzero_output_path),
        "TARGET_COLUMN": target_column,
        "USE_FIRST_COLUMN_AS_INDEX": use_first_column_as_index,
        "TARGET_TRANSFORM_METHOD": target_transform_method,
        "CLIP_ZERO_AND_HUNDRED": clip_zero_and_hundred,
        "CORR_THRESHOLD": corr_threshold,
        "REMOVE_NEAR_CONSTANT": remove_near_constant,
        "NEAR_CONSTANT_STD_THRESHOLD": near_constant_std_threshold,
        "MODELS_TO_RUN": models_to_run,
        "GRIDSEARCH_SCORING": GRIDSEARCH_SCORING,
        "RANDOM_STATE": RANDOM_STATE,
        "N_BOOTSTRAP": n_bootstrap,
        "BOOTSTRAP_MODEL": bootstrap_model,
        "RUN_SHAP": run_shap,
        "PLS_COMPONENT_SELECTION_RULE": {
            "n_samples_le_20": "1 to min(4, n_features, n_samples - 1)",
            "n_samples_le_30": "1 to min(6, n_features, n_samples - 1)",
            "otherwise": "1 to min(10, n_features, n_samples - 1)",
        },
    }
    save_json(info_dir / "run_config.json", run_config)
    save_library_versions(info_dir)

    if verbose:
        print("X shape:", X.shape)
        print("y length:", len(y_trans))
        print("clip info:", clip_info)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    X_proc = X.copy()

    if remove_near_constant:
        X_proc, removed_constant_cols, feature_stds = remove_near_constant_features(
            X_proc,
            std_threshold=near_constant_std_threshold,
        )
    else:
        removed_constant_cols = []
        feature_stds = X_proc.std(axis=0, ddof=0)

    X_proc, removed_corr_cols, corr_matrix = remove_high_correlation_features(
        X_proc,
        threshold=corr_threshold,
    )

    save_text_lines(info_dir / "removed_near_constant_features.txt", removed_constant_cols)
    save_text_lines(info_dir / "removed_high_correlation_features.txt", removed_corr_cols)
    feature_stds.to_csv(info_dir / "feature_stds.csv", encoding="utf-8-sig")
    corr_matrix.to_csv(eda_dir / "descriptor_correlation_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame({"feature": X_proc.columns}).to_csv(info_dir / "features_after_filtering.csv", index=False, encoding="utf-8-sig")

    if verbose:
        print("Original X shape:", X.shape)
        print("Filtered X shape:", X_proc.shape)
        print("Removed near-constant:", len(removed_constant_cols))
        print("Removed high-correlation:", len(removed_corr_cols))

    # --------------------------------------------------
    # Univariate correlation
    # --------------------------------------------------
    univariate_rows = []
    for col in X_proc.columns:
        xvals = X_proc[col].values
        yvals = y_pct_clipped.values

        pearson_r, pearson_p = pearsonr(xvals, yvals)
        spearman_r, spearman_p = spearmanr(xvals, yvals)

        univariate_rows.append({
            "feature": col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "abs_pearson_r": abs(pearson_r),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_spearman_r": abs(spearman_r),
        })

    univariate_df = pd.DataFrame(univariate_rows).sort_values("abs_spearman_r", ascending=False)
    univariate_df.to_csv(eda_dir / "univariate_correlations.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------
    # Pair plot
    # --------------------------------------------------
    if PAIRPLOT_FEATURES is None:
        pairplot_features = univariate_df.head(PAIRPLOT_MAX_FEATURES)["feature"].tolist()
    else:
        pairplot_features = PAIRPLOT_FEATURES

    make_descriptor_pairplot(
        df=X_proc,
        save_path=plot_dir / "descriptor_pairplot.png",
        feature_list=pairplot_features,
    )

    pd.DataFrame({"pairplot_features": pairplot_features}).to_csv(
        info_dir / "descriptor_pairplot_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if verbose:
        print("Saved descriptor pair plot:", plot_dir / "descriptor_pairplot.png")
        print("Pair plot features:", pairplot_features)

    # --------------------------------------------------
    # Target distribution
    # --------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(y_pct_clipped.values, bins=min(10, max(5, len(y_pct_clipped) // 2)))
    plt.xlabel("Target (%)")
    plt.ylabel("Count")
    plt.title("Distribution of target (clipped % scale)")
    plt.tight_layout()
    plt.savefig(plot_dir / "target_distribution_percent.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(y_trans.values, bins=min(10, max(5, len(y_trans) // 2)))
    plt.xlabel("Transformed target")
    plt.ylabel("Count")
    plt.title("Distribution of transformed target")
    plt.tight_layout()
    plt.savefig(plot_dir / "target_distribution_transformed.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------
    # LOOCV model evaluation
    # --------------------------------------------------
    def loocv_model_evaluation(model_name, X_df, y_trans_local, y_pct_clipped_local):
        model = clone(model_dict[model_name])
        loo = LeaveOneOut()

        pred_rows = []
        removed_union = set()
        best_params_list = []
        selected_features_per_fold = []

        for fold_no, (train_idx, test_idx) in enumerate(loo.split(X_df), start=1):
            X_train = X_df.iloc[train_idx].copy()
            X_test = X_df.iloc[test_idx].copy()

            y_train_trans = y_trans_local.iloc[train_idx].copy()
            y_test_trans = y_trans_local.iloc[test_idx].copy()
            y_test_pct = y_pct_clipped_local.iloc[test_idx].copy()

            if remove_near_constant:
                X_train, removed_const_fold, _ = remove_near_constant_features(
                    X_train,
                    std_threshold=near_constant_std_threshold,
                )
                removed_union.update(removed_const_fold)

            X_train, removed_corr_fold, _ = remove_high_correlation_features(
                X_train,
                threshold=corr_threshold,
            )
            X_test = X_test[X_train.columns].copy()
            removed_union.update(removed_corr_fold)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            param_grid = adapt_param_grid(model_name, PARAM_GRIDS[model_name], X_train.shape[1], X_train.shape[0])
            inner_cv = build_inner_cv_splits(len(X_train))

            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=GRIDSEARCH_SCORING,
                cv=inner_cv,
                n_jobs=N_JOBS,
                verbose=VERBOSE,
                refit=True,
            )
            gs.fit(X_train_scaled, y_train_trans)

            best_model = gs.best_estimator_
            best_params_list.append(str(gs.best_params_))

            y_test_pred_trans = np.asarray(best_model.predict(X_test_scaled)).reshape(-1)
            y_test_pred_pct = inverse_transform_to_percent(
                y_test_pred_trans,
                target_transform_method=target_transform_method,
            )

            pred_rows.append({
                "Index": X_test.index[0],
                "Fold": fold_no,
                "Observed_percent": float(y_test_pct.values[0]),
                "Predicted_percent": float(y_test_pred_pct[0]),
                "Observed_transformed": float(y_test_trans.values[0]),
                "Predicted_transformed": float(y_test_pred_trans[0]),
            })

            selected_features_per_fold.append({
                "Fold": fold_no,
                "NumFeaturesUsed": X_train.shape[1],
                "FeaturesUsed": ";".join(X_train.columns.tolist()),
            })

        pred_df = pd.DataFrame(pred_rows).sort_values("Index")
        metrics = compute_metrics(pred_df["Observed_percent"], pred_df["Predicted_percent"])

        metrics_df = pd.DataFrame([{
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MSE": metrics["MSE"],
            "R2": metrics["R2"],
        }])

        pred_df.to_csv(pred_dir / f"{model_name}_LOOCV_predictions.csv", index=False, encoding="utf-8-sig")
        metrics_df.to_csv(info_dir / f"{model_name}_LOOCV_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"BestParams": best_params_list}).to_csv(
            info_dir / f"{model_name}_LOOCV_best_params.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(selected_features_per_fold).to_csv(
            info_dir / f"{model_name}_LOOCV_features_per_fold.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_text_lines(info_dir / f"{model_name}_LOOCV_removed_features_union.txt", sorted(removed_union))

        yy_plot(
            pred_df["Observed_percent"],
            pred_df["Predicted_percent"],
            title=f"{model_name} LOOCV y-y plot",
            save_path=plot_dir / f"{model_name}_LOOCV_yy.png",
        )

        return metrics_df, pred_df

    model_summary_rows = []
    for model_name in models_to_run:
        if verbose:
            print("=" * 70)
            print(f"Running LOOCV for: {model_name}")
        metrics_df, pred_df = loocv_model_evaluation(
            model_name=model_name,
            X_df=X_proc,
            y_trans_local=y_trans,
            y_pct_clipped_local=y_pct_clipped,
        )
        model_summary_rows.append(metrics_df.iloc[0].to_dict())

    model_summary_df = pd.DataFrame(model_summary_rows).sort_values("RMSE", ascending=True)
    model_summary_df.to_csv(info_dir / "model_comparison_loocv.csv", index=False, encoding="utf-8-sig")

    metrics_barplot(model_summary_df, "RMSE", plot_dir / "comparison_loocv_RMSE.png")
    metrics_barplot(model_summary_df, "MSE", plot_dir / "comparison_loocv_MSE.png")
    metrics_barplot(model_summary_df, "R2", plot_dir / "comparison_loocv_R2.png")

    best_model_name = model_summary_df.iloc[0]["Model"]
    save_text_lines(info_dir / "best_model.txt", [best_model_name])

    if verbose:
        print("Best model:", best_model_name)

    # --------------------------------------------------
    # Final fit
    # --------------------------------------------------
    def get_representative_best_params(model_name):
        df_local = pd.read_csv(info_dir / f"{model_name}_LOOCV_best_params.csv")
        params_str = df_local["BestParams"].dropna().astype(str).mode().iloc[0]
        return ast.literal_eval(params_str)

    def fit_final_model(model_name, X_df, y_trans_local, y_pct_clipped_local):
        model = clone(model_dict[model_name])
        rep_params = get_representative_best_params(model_name)
        rep_grid = adapt_param_grid(model_name, PARAM_GRIDS[model_name], X_df.shape[1], X_df.shape[0])
        valid_params = {k: v for k, v in rep_params.items() if k in rep_grid}
        model.set_params(**valid_params)

        X_fit = X_df.copy()

        if remove_near_constant:
            X_fit, removed_const, _ = remove_near_constant_features(
                X_fit,
                std_threshold=near_constant_std_threshold,
            )
        else:
            removed_const = []

        X_fit, removed_corr, corr_matrix_final = remove_high_correlation_features(
            X_fit,
            threshold=corr_threshold,
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_fit)

        model.fit(X_scaled, y_trans_local)
        y_pred_trans = np.asarray(model.predict(X_scaled)).reshape(-1)
        y_pred_pct = inverse_transform_to_percent(
            y_pred_trans,
            target_transform_method=target_transform_method,
        )

        metrics = compute_metrics(y_pct_clipped_local, y_pred_pct)
        metrics_df = pd.DataFrame([{
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MSE": metrics["MSE"],
            "R2": metrics["R2"],
        }])

        pred_df = pd.DataFrame({
            "Index": X_fit.index,
            "Observed_percent": y_pct_clipped_local.values,
            "Predicted_percent": y_pred_pct,
            "Observed_transformed": y_trans_local.values,
            "Predicted_transformed": y_pred_trans,
        })

        pred_df.to_csv(pred_dir / f"{model_name}_fullfit_predictions.csv", index=False, encoding="utf-8-sig")
        metrics_df.to_csv(info_dir / f"{model_name}_fullfit_metrics.csv", index=False, encoding="utf-8-sig")
        corr_matrix_final.to_csv(info_dir / f"{model_name}_final_correlation_matrix.csv", encoding="utf-8-sig")
        pd.DataFrame({"feature": X_fit.columns}).to_csv(
            info_dir / f"{model_name}_final_features_used.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_text_lines(info_dir / f"{model_name}_final_removed_constant_features.txt", removed_const)
        save_text_lines(info_dir / f"{model_name}_final_removed_high_corr_features.txt", removed_corr)

        scaler_params = pd.DataFrame({
            "feature": X_fit.columns,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        })
        scaler_params.to_csv(info_dir / f"{model_name}_final_scaler_parameters.csv", index=False, encoding="utf-8-sig")

        yy_plot(
            y_pct_clipped_local,
            y_pred_pct,
            title=f"{model_name} full fit y-y plot",
            save_path=plot_dir / f"{model_name}_fullfit_yy.png",
        )

        return {
            "model_name": model_name,
            "model": model,
            "X_fit": X_fit,
            "X_scaled": X_scaled,
            "scaler": scaler,
            "pred_df": pred_df,
            "metrics_df": metrics_df,
            "params": valid_params,
            "corr_matrix_final": corr_matrix_final,
            "removed_const": removed_const,
            "removed_corr": removed_corr,
        }

    def extract_model_contributions(model_name, model, feature_names):
        if model_name in ["Lasso", "ElasticNet", "Ridge", "PLS"]:
            coef = np.asarray(model.coef_).reshape(-1)
            result_df = pd.DataFrame({
                "feature": feature_names,
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
                "sign": np.sign(coef),
            }).sort_values("abs_coefficient", ascending=False)
            result_df.to_csv(info_dir / f"{model_name}_coefficients.csv", index=False, encoding="utf-8-sig")

            plt.figure(figsize=(10, 6))
            top_df = result_df.head(30).iloc[::-1]
            plt.barh(top_df["feature"], top_df["coefficient"])
            plt.xlabel("Coefficient")
            plt.ylabel("Feature")
            plt.title(f"{model_name}: coefficients (Top 30)")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{model_name}_coefficients_top30.png", dpi=300, bbox_inches="tight")
            plt.close()
            return result_df

        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).reshape(-1)
            result_df = pd.DataFrame({
                "feature": feature_names,
                "importance": imp,
            }).sort_values("importance", ascending=False)
            result_df.to_csv(info_dir / f"{model_name}_feature_importance.csv", index=False, encoding="utf-8-sig")

            plt.figure(figsize=(10, 6))
            top_df = result_df.head(30).iloc[::-1]
            plt.barh(top_df["feature"], top_df["importance"])
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.title(f"{model_name}: feature importance (Top 30)")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{model_name}_feature_importance_top30.png", dpi=300, bbox_inches="tight")
            plt.close()
            return result_df

        return None

    final_model_results = {}
    fullfit_summary_rows = []
    contribution_results = {}

    for model_name in models_to_run:
        if verbose:
            print("=" * 70)
            print(f"Running full fit for: {model_name}")

        result = fit_final_model(
            model_name,
            X_proc,
            y_trans,
            y_pct_clipped,
        )
        final_model_results[model_name] = result
        fullfit_summary_rows.append(result["metrics_df"].iloc[0].to_dict())

        contribution_df = extract_model_contributions(
            model_name,
            result["model"],
            result["X_fit"].columns,
        )
        contribution_results[model_name] = contribution_df

        if model_name == "Lasso" and contribution_df is not None:
            nonzero_df = contribution_df[np.abs(contribution_df["coefficient"]) > 1e-15].copy()
            zero_df = contribution_df[np.abs(contribution_df["coefficient"]) <= 1e-15].copy()

            nonzero_features = [f for f in nonzero_df["feature"].tolist() if f in X.columns]
            X_lasso_nonzero = X[nonzero_features].copy()

            lasso_nonzero_output_path.parent.mkdir(parents=True, exist_ok=True)
            X_lasso_nonzero.to_csv(lasso_nonzero_output_path, encoding="utf-8-sig")

            zero_df.to_csv(info_dir / "Lasso_zero_coefficient_features.csv", index=False, encoding="utf-8-sig")
            nonzero_df.to_csv(info_dir / "Lasso_nonzero_coefficient_features.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame({"feature": nonzero_features}).to_csv(
                info_dir / "Lasso_nonzero_features_for_output.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if verbose:
                print(f"Saved Lasso non-zero feature input file: {lasso_nonzero_output_path}")
                print(f"Number of Lasso non-zero coefficients: {len(nonzero_features)}")

    fullfit_summary_df = pd.DataFrame(fullfit_summary_rows).sort_values("RMSE", ascending=True)
    fullfit_summary_df.to_csv(info_dir / "model_comparison_fullfit.csv", index=False, encoding="utf-8-sig")

    best_result = final_model_results[best_model_name]
    best_model_fitted = best_result["model"]
    X_final = best_result["X_fit"]
    X_final_scaled = best_result["X_scaled"]
    final_pred_df = best_result["pred_df"]
    final_metrics_df = best_result["metrics_df"]
    final_params = best_result["params"]
    contribution_df = contribution_results[best_model_name]

    if verbose:
        print("Representative parameters of the best LOOCV model:", final_params)

    # --------------------------------------------------
    # Bootstrap stability
    # --------------------------------------------------
    def bootstrap_stability_analysis(model_name, X_df, y_trans_local, n_bootstrap_local=200):
        if model_name not in ["Lasso", "ElasticNet", "Ridge", "PLS"]:
            return None, None

        model = clone(model_dict[model_name])
        rep_params = get_representative_best_params(model_name)
        param_grid = adapt_param_grid(model_name, PARAM_GRIDS[model_name], X_df.shape[1], X_df.shape[0])
        valid_params = {k: v for k, v in rep_params.items() if k in param_grid}
        model.set_params(**valid_params)

        n = len(X_df)
        coef_rows = []

        for b in range(1, n_bootstrap_local + 1):
            sample_idx = np.random.choice(np.arange(n), size=n, replace=True)
            X_boot = X_df.iloc[sample_idx].copy()
            y_boot = y_trans_local.iloc[sample_idx].copy()

            if remove_near_constant:
                X_boot, _, _ = remove_near_constant_features(
                    X_boot,
                    std_threshold=near_constant_std_threshold,
                )

            X_boot, _, _ = remove_high_correlation_features(
                X_boot,
                threshold=corr_threshold,
            )

            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot)

            fitted = clone(model)
            fitted.fit(X_boot_scaled, y_boot)

            coef_map = {}
            vals = np.asarray(fitted.coef_).reshape(-1)
            for feat, val in zip(X_boot.columns.tolist(), vals):
                coef_map[feat] = float(val)

            row = {"bootstrap_id": b}
            row.update(coef_map)
            coef_rows.append(row)

        coef_df = pd.DataFrame(coef_rows).fillna(0.0)
        coef_df.to_csv(stab_dir / f"{model_name}_bootstrap_coefficients_raw.csv", index=False, encoding="utf-8-sig")

        feature_cols = [c for c in coef_df.columns if c != "bootstrap_id"]
        summary_rows = []
        for f in feature_cols:
            vals = coef_df[f].values
            summary_rows.append({
                "feature": f,
                "selection_frequency": float(np.mean(np.abs(vals) > 1e-15)),
                "coef_mean": float(np.mean(vals)),
                "coef_std": float(np.std(vals)),
                "mean_abs_coef": float(np.mean(np.abs(vals))),
                "positive_frequency": float(np.mean(vals > 0)),
                "negative_frequency": float(np.mean(vals < 0)),
                "sign_consistency": float(max(np.mean(vals > 0), np.mean(vals < 0))),
            })

        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["selection_frequency", "mean_abs_coef"],
            ascending=[False, False],
        )
        summary_df.to_csv(stab_dir / f"{model_name}_bootstrap_stability_summary.csv", index=False, encoding="utf-8-sig")

        top_df = summary_df.head(30).iloc[::-1]

        plt.figure(figsize=(10, 8))
        plt.barh(top_df["feature"], top_df["selection_frequency"])
        plt.xlabel("Selection frequency")
        plt.ylabel("Feature")
        plt.title(f"{model_name}: bootstrap selection frequency (Top 30)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{model_name}_bootstrap_selection_frequency_top30.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.barh(top_df["feature"], top_df["coef_mean"])
        plt.xlabel("Mean coefficient")
        plt.ylabel("Feature")
        plt.title(f"{model_name}: bootstrap mean coefficient (Top 30)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{model_name}_bootstrap_mean_coefficient_top30.png", dpi=300, bbox_inches="tight")
        plt.close()

        return coef_df, summary_df

    bootstrap_raw_df = None
    bootstrap_summary_df = None
    if bootstrap_model in model_dict:
        bootstrap_raw_df, bootstrap_summary_df = bootstrap_stability_analysis(
            bootstrap_model,
            X_proc,
            y_trans,
            n_bootstrap_local=n_bootstrap,
        )

    # --------------------------------------------------
    # SHAP analysis
    # --------------------------------------------------
    def run_shap_analysis(model_name, model, X_df, X_scaled_df, pred_df=None):
        feature_names = list(X_df.columns)

        try:
            if model_name in ["Lasso", "ElasticNet", "Ridge"]:
                explainer = shap.LinearExplainer(model, X_scaled_df, feature_perturbation="interventional")
                shap_values = explainer.shap_values(X_scaled_df)
            else:
                background = shap.sample(
                    X_scaled_df,
                    min(SHAP_BACKGROUND_SIZE, len(X_scaled_df)),
                    random_state=RANDOM_STATE,
                )
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_scaled_df, nsamples=SHAP_KERNEL_NSAMPLES)

            shap_values = np.asarray(shap_values)
            shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_df.index)
            shap_df.to_csv(shap_dir / f"{model_name}_shap_values.csv", encoding="utf-8-sig")

            mean_abs_shap = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            mean_abs_shap.to_csv(shap_dir / f"{model_name}_mean_abs_shap.csv", index=False, encoding="utf-8-sig")

            explanation = ensure_shap_explanation(
                explainer=explainer,
                shap_values=shap_values,
                X_values=X_scaled_df.values,
                feature_names=feature_names,
            )

            plt.figure()
            shap.summary_plot(
                shap_values,
                X_scaled_df,
                feature_names=feature_names,
                max_display=min(SHAP_MAX_DISPLAY, len(feature_names)),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(shap_dir / f"{model_name}_shap_summary.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            shap.summary_plot(
                shap_values,
                X_scaled_df,
                feature_names=feature_names,
                plot_type="bar",
                max_display=min(SHAP_MAX_DISPLAY, len(feature_names)),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(shap_dir / f"{model_name}_shap_bar.png", dpi=300, bbox_inches="tight")
            plt.close()

            scatter_features = mean_abs_shap["feature"].head(min(SHAP_SCATTER_TOP_N, len(mean_abs_shap))).tolist()
            pd.DataFrame({"scatter_feature": scatter_features}).to_csv(
                shap_dir / f"{model_name}_shap_scatter_features.csv",
                index=False,
                encoding="utf-8-sig",
            )

            for feat in scatter_features:
                feat_idx = feature_names.index(feat)
                plt.figure()
                shap.plots.scatter(explanation[:, feat_idx], show=False)
                plt.tight_layout()
                safe_feat = str(feat).replace("/", "_").replace("\\", "_").replace(":", "_")
                plt.savefig(
                    shap_dir / f"{model_name}_shap_scatter_{feat_idx+1:02d}_{safe_feat}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            dependence_features = mean_abs_shap["feature"].head(min(SHAP_DEPENDENCE_TOP_N, len(mean_abs_shap))).tolist()
            pd.DataFrame({"dependence_feature": dependence_features}).to_csv(
                shap_dir / f"{model_name}_shap_dependence_features.csv",
                index=False,
                encoding="utf-8-sig",
            )

            for feat in dependence_features:
                feat_idx = feature_names.index(feat)
                plt.figure()
                shap.dependence_plot(
                    feat_idx,
                    shap_values,
                    X_scaled_df,
                    feature_names=feature_names,
                    interaction_index="auto",
                    show=False,
                )
                plt.tight_layout()
                safe_feat = str(feat).replace("/", "_").replace("\\", "_").replace(":", "_")
                plt.savefig(
                    shap_dir / f"{model_name}_shap_dependence_{feat_idx+1:02d}_{safe_feat}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            selected_indices = select_waterfall_samples(
                pred_df=pred_df,
                mode=SHAP_WATERFALL_SAMPLE_MODE,
                max_samples=SHAP_WATERFALL_MAX_SAMPLES,
                sample_index=SHAP_WATERFALL_SAMPLE_INDEX,
            )

            waterfall_meta = []
            for idx_value in selected_indices:
                if idx_value not in X_df.index:
                    continue

                row_pos = X_df.index.get_loc(idx_value)
                row_exp = explanation[row_pos]

                plt.figure()
                shap.plots.waterfall(
                    row_exp,
                    max_display=min(SHAP_MAX_DISPLAY, len(feature_names)),
                    show=False,
                )
                plt.tight_layout()
                safe_idx = str(idx_value).replace("/", "_").replace("\\", "_").replace(":", "_")
                plt.savefig(
                    shap_dir / f"{model_name}_shap_waterfall_{safe_idx}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                meta_row = {"Index": idx_value, "row_position": int(row_pos)}
                if pred_df is not None and "Index" in pred_df.columns:
                    matched = pred_df[pred_df["Index"] == idx_value]
                    if len(matched) > 0:
                        rec = matched.iloc[0]
                        meta_row.update({
                            "Observed_percent": rec.get("Observed_percent", np.nan),
                            "Predicted_percent": rec.get("Predicted_percent", np.nan),
                            "abs_error": abs(rec.get("Observed_percent", np.nan) - rec.get("Predicted_percent", np.nan)),
                        })
                waterfall_meta.append(meta_row)

            if waterfall_meta:
                pd.DataFrame(waterfall_meta).to_csv(
                    shap_dir / f"{model_name}_shap_waterfall_samples.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            force_meta = []
            force_indices = selected_indices[:max(1, int(SHAP_FORCE_TOP_N))]
            for idx_value in force_indices:
                if idx_value not in X_df.index:
                    continue

                row_pos = X_df.index.get_loc(idx_value)
                try:
                    force_plot = shap.force_plot(
                        explanation.base_values[row_pos],
                        explanation.values[row_pos],
                        X_scaled_df.iloc[row_pos, :],
                        feature_names=feature_names,
                        matplotlib=SHAP_FORCE_PLOT_MATPLOTLIB,
                        show=False,
                    )
                    safe_idx = str(idx_value).replace("/", "_").replace("\\", "_").replace(":", "_")

                    if SHAP_FORCE_PLOT_MATPLOTLIB:
                        plt.tight_layout()
                        saved_path = shap_dir / f"{model_name}_shap_force_{safe_idx}.png"
                        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
                        plt.close()
                    else:
                        saved_path = shap_dir / f"{model_name}_shap_force_{safe_idx}.html"
                        shap.save_html(str(saved_path), force_plot)

                    meta_row = {
                        "Index": idx_value,
                        "row_position": int(row_pos),
                        "saved_file": saved_path.name,
                    }
                    if pred_df is not None and "Index" in pred_df.columns:
                        matched = pred_df[pred_df["Index"] == idx_value]
                        if len(matched) > 0:
                            rec = matched.iloc[0]
                            meta_row.update({
                                "Observed_percent": rec.get("Observed_percent", np.nan),
                                "Predicted_percent": rec.get("Predicted_percent", np.nan),
                                "abs_error": abs(rec.get("Observed_percent", np.nan) - rec.get("Predicted_percent", np.nan)),
                            })
                    force_meta.append(meta_row)
                except Exception as force_e:
                    print(f"Failed to save the force plot for sample {idx_value}: {force_e}")

            try:
                global_force_plot = shap.force_plot(
                    explanation.base_values,
                    explanation.values,
                    X_scaled_df,
                    feature_names=feature_names,
                    matplotlib=False,
                    show=False,
                )
                shap.save_html(str(shap_dir / f"{model_name}_shap_force_global.html"), global_force_plot)
            except Exception as global_force_e:
                print(f"Failed to save the global force plot: {global_force_e}")

            if force_meta:
                pd.DataFrame(force_meta).to_csv(
                    shap_dir / f"{model_name}_shap_force_samples.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            return mean_abs_shap

        except Exception as e:
            print("An exception occurred during SHAP analysis:", e)
            return None

    shap_summary_df = None
    if run_shap:
        X_final_scaled_df = pd.DataFrame(X_final_scaled, columns=X_final.columns, index=X_final.index)
        shap_summary_df = run_shap_analysis(
            best_model_name,
            best_model_fitted,
            X_final,
            X_final_scaled_df,
            pred_df=final_pred_df,
        )

    summary_txt = [
        f"Best LOOCV model: {best_model_name}",
        f"Bootstrap model: {bootstrap_model}",
        f"Full models built for: {', '.join(models_to_run)}",
        "",
        "Basic interpretation policy:",
        "- Evaluate predictive performance with LOOCV.",
        "- Build full-fit models for all specified algorithms.",
        "- Save a Lasso-based reduced input table using only non-zero coefficient descriptors.",
        "- Use SHAP as supporting evidence for the best LOOCV model.",
    ]
    save_text_lines(info_dir / "analysis_summary_notes.txt", summary_txt)

    if verbose:
        print("All analyses have been completed.")
        print(f"Results saved to: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "best_model": best_model_name,
        "model_comparison": model_summary_df,
        "fullfit_comparison": fullfit_summary_df,
        "final_metrics_best_model": final_metrics_df,
        "contribution_df_best_model": contribution_df,
        "bootstrap_summary_df": bootstrap_summary_df,
        "shap_summary_df": shap_summary_df,
        "lasso_nonzero_output_path": str(lasso_nonzero_output_path),
    }


if __name__ == "__main__":
    run(
        input_x_path="data/input_filtering.csv",
        input_y_path="data/output.csv",
        output_dir="3_ML_small_data_analysis",
        lasso_nonzero_output_path="data/input_lasso_nonzero.csv",
    )
