#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

# Suppress RDKit internal logs
RDLogger.DisableLog("rdApp.*")


# ============================================================
# Descriptor definitions
# ============================================================
ALL_DESC_LIST = sorted(Descriptors.descList, key=lambda x: x[0])
ALL_DESC_DICT = {name: func for name, func in ALL_DESC_LIST}

SELECTED_DESCRIPTORS = [
    "MaxAbsPartialCharge",
    "MaxPartialCharge",
    "MinPartialCharge",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "NumAromaticRings",
    "RingCount",
    "NumAromaticHeterocycles",
    "NumHeteroatoms",
    "NumHAcceptors",
    "NumHDonors",
    "LabuteASA",
    "TPSA",
    "NumRotatableBonds",
]

DEFAULT_CORRELATION_THRESHOLD = 0.995


# ============================================================
# Logging
# ============================================================
def log(message, verbose=False):
    if verbose:
        print(message)


# ============================================================
# Basic utilities
# ============================================================
def get_default_paths(input_file=None, calculation_dir=None, final_output=None):
    if input_file is None:
        input_file = Path("data/RDKit_descriptors") / "SMILES_list.csv"
    else:
        input_file = Path(input_file)

    base_dir = input_file.parent

    if calculation_dir is None:
        calculation_dir = base_dir / "calculation"
    else:
        calculation_dir = Path(calculation_dir)

    if final_output is None:
        final_output = base_dir / "RDkit_selected_descriptors.csv"
    else:
        final_output = Path(final_output)

    return input_file, calculation_dir, final_output


def calc_all_descriptors_from_smiles(smiles):
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None

    if mol is None:
        return None

    values = {}
    for name, func in ALL_DESC_DICT.items():
        try:
            values[name] = func(mol)
        except Exception:
            values[name] = None

    return values


def build_descriptor_table(df, id_col, smiles_col, source_label):
    records = []
    invalid_rows = []

    for _, row in df.iterrows():
        compound_id = row[id_col]
        smiles = row[smiles_col]

        desc = calc_all_descriptors_from_smiles(smiles)
        if desc is None:
            invalid_rows.append(
                {
                    "compound_id": compound_id,
                    "smiles": smiles,
                    "source_column": source_label,
                }
            )
            continue

        record = {id_col: compound_id}
        record.update(desc)
        records.append(record)

    if records:
        result_df = pd.DataFrame(records)
    else:
        result_df = pd.DataFrame(columns=[id_col] + list(ALL_DESC_DICT.keys()))

    result_df = result_df.replace([float("inf"), float("-inf")], pd.NA)
    return result_df, invalid_rows


def remove_constant_columns(df, protected_columns=None):
    if protected_columns is None:
        protected_columns = []

    keep_columns = []
    removed_columns = []

    for col in df.columns:
        if col in protected_columns:
            keep_columns.append(col)
            continue

        if df[col].nunique(dropna=False) <= 1:
            removed_columns.append(col)
        else:
            keep_columns.append(col)

    return df[keep_columns].copy(), removed_columns


def pick_selected_descriptors(df, id_col):
    peoe_vsa_cols = [col for col in df.columns if col.startswith("PEOE_VSA")]
    available_selected = [col for col in SELECTED_DESCRIPTORS if col in df.columns]

    final_columns = [id_col] + available_selected + peoe_vsa_cols
    final_columns = list(dict.fromkeys(final_columns))

    return df[final_columns].copy()


def drop_redundant_h_columns(df_rdkit, df_h, id_col, corr_threshold=DEFAULT_CORRELATION_THRESHOLD):
    shared_columns = [
        col for col in df_rdkit.columns
        if col != id_col and col in df_h.columns
    ]

    dropped_info = []
    keep_h_columns = [id_col]

    for col in shared_columns:
        left = df_rdkit[col]
        right = df_h[col]

        comparison = pd.concat([left, right], axis=1).dropna()
        if comparison.empty:
            keep_h_columns.append(col)
            continue

        are_identical = comparison.iloc[:, 0].equals(comparison.iloc[:, 1])
        if are_identical:
            dropped_info.append(
                {
                    "descriptor": col,
                    "reason": "identical",
                    "correlation": 1.0,
                }
            )
            continue

        if len(comparison) < 2:
            keep_h_columns.append(col)
            continue

        corr = comparison.iloc[:, 0].corr(comparison.iloc[:, 1])
        if pd.notna(corr) and abs(corr) >= corr_threshold:
            dropped_info.append(
                {
                    "descriptor": col,
                    "reason": "high_correlation",
                    "correlation": float(corr),
                }
            )
        else:
            keep_h_columns.append(col)

    unique_h_columns = [
        col for col in df_h.columns
        if col not in shared_columns and col != id_col
    ]
    keep_h_columns.extend(unique_h_columns)

    result_df = df_h[keep_h_columns].copy()
    dropped_df = pd.DataFrame(dropped_info)
    return result_df, dropped_df


def add_prefix_except_id(df, id_col, prefix="H_"):
    renamed = {}
    for col in df.columns:
        if col != id_col:
            renamed[col] = f"{prefix}{col}"
    return df.rename(columns=renamed)


def save_removed_columns_txt(file_path, removed_columns, protected_columns=None):
    if protected_columns is None:
        protected_columns = []

    file_path = Path(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        for col in removed_columns:
            if col not in protected_columns:
                f.write(f"{col}\n")


def save_invalid_smiles_log(file_path, invalid_rows):
    file_path = Path(file_path)
    if invalid_rows:
        pd.DataFrame(invalid_rows).to_csv(file_path, index=False)
    else:
        pd.DataFrame(columns=["compound_id", "smiles", "source_column"]).to_csv(file_path, index=False)


# ============================================================
# Main workflow
# ============================================================
def run(
    input_file=None,
    calculation_dir=None,
    final_output=None,
    corr_threshold=DEFAULT_CORRELATION_THRESHOLD,
    verbose=False,
    return_results=False,
):
    input_file, calculation_dir, final_output = get_default_paths(
        input_file=input_file,
        calculation_dir=calculation_dir,
        final_output=final_output,
    )

    calculation_dir.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file was not found: {input_file}")

    log("Start: RDKit descriptor calculation", verbose)
    log(f"Input file      : {input_file}", verbose)
    log(f"Calculation dir : {calculation_dir}", verbose)
    log(f"Final output    : {final_output}", verbose)

    df = pd.read_csv(input_file)
    if df.shape[1] < 3:
        raise ValueError("The input CSV must contain at least three columns: ID, SMILES, and H_SMILES.")

    id_col = df.columns[0]
    smiles_col = df.columns[1]
    h_smiles_col = df.columns[2]

    log(f"ID column       : {id_col}", verbose)
    log(f"SMILES column   : {smiles_col}", verbose)
    log(f"H_SMILES column : {h_smiles_col}", verbose)
    log(f"Descriptor count: {len(ALL_DESC_DICT)}", verbose)

    df_rdkit_raw, invalid_rdkit = build_descriptor_table(df, id_col, smiles_col, smiles_col)
    df_h_raw, invalid_h = build_descriptor_table(df, id_col, h_smiles_col, h_smiles_col)

    rdkit_csv = calculation_dir / "RDKit.csv"
    h_rdkit_csv = calculation_dir / "H_RDKit.csv"
    df_rdkit_raw.to_csv(rdkit_csv, index=False)
    df_h_raw.to_csv(h_rdkit_csv, index=False)

    log(f"Saved raw descriptor table: {rdkit_csv}", verbose)
    log(f"Saved raw descriptor table: {h_rdkit_csv}", verbose)

    df_rdkit_clean, removed_rdkit = remove_constant_columns(df_rdkit_raw, protected_columns=[id_col])
    df_h_clean, removed_h = remove_constant_columns(df_h_raw, protected_columns=[id_col])

    save_removed_columns_txt(
        calculation_dir / "RDKit_removed_descriptors.txt",
        removed_rdkit,
        protected_columns=[id_col],
    )
    save_removed_columns_txt(
        calculation_dir / "H_RDKit_removed_descriptors.txt",
        removed_h,
        protected_columns=[id_col],
    )

    df_rdkit_selected = pick_selected_descriptors(df_rdkit_clean, id_col)
    df_h_selected = pick_selected_descriptors(df_h_clean, id_col)

    df_h_filtered, dropped_h_info = drop_redundant_h_columns(
        df_rdkit_selected,
        df_h_selected,
        id_col,
        corr_threshold=corr_threshold,
    )

    redundant_file = calculation_dir / "H_RDKit_redundant_with_RDKit.csv"
    if dropped_h_info.empty:
        pd.DataFrame(columns=["descriptor", "reason", "correlation"]).to_csv(redundant_file, index=False)
    else:
        dropped_h_info.to_csv(redundant_file, index=False)

    df_h_prefixed = add_prefix_except_id(df_h_filtered, id_col, prefix="H_")
    merged_df = pd.merge(df_rdkit_selected, df_h_prefixed, on=id_col, how="inner")

    merged_df.to_csv(final_output, index=False)

    invalid_log_file = calculation_dir / "invalid_smiles_log.csv"
    save_invalid_smiles_log(invalid_log_file, invalid_rdkit + invalid_h)

    if verbose:
        print(f"Saved final merged table: {final_output}")
        print("Finished")
        print(f"Final merged shape: {merged_df.shape}")
    else:
        print(f"Saved: {final_output}")

    if return_results:
        return {
            "input_file": str(input_file),
            "calculation_dir": str(calculation_dir),
            "final_output": str(final_output),
            "id_col": id_col,
            "smiles_col": smiles_col,
            "h_smiles_col": h_smiles_col,
            "rdkit_raw_shape": df_rdkit_raw.shape,
            "h_raw_shape": df_h_raw.shape,
            "rdkit_selected_shape": df_rdkit_selected.shape,
            "h_filtered_shape": df_h_filtered.shape,
            "merged_shape": merged_df.shape,
            "removed_rdkit_columns": removed_rdkit,
            "removed_h_columns": removed_h,
            "dropped_h_info": dropped_h_info,
        }

    return None


if __name__ == "__main__":
    run()
