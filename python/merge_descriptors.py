#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import sys


def run(
    dft_file,
    rdkit_file,
    output_file,
    verbose=False
):
    dft_file = Path(dft_file)
    rdkit_file = Path(rdkit_file)
    output_file = Path(output_file)

    if not dft_file.exists():
        raise FileNotFoundError(f"DFT file not found: {dft_file}")
    if not rdkit_file.exists():
        raise FileNotFoundError(f"RDKit file not found: {rdkit_file}")

    if verbose:
        print("Reading descriptor tables")

    df_dft = pd.read_csv(dft_file)
    df_rdkit = pd.read_csv(rdkit_file)

    if verbose:
        print("DFT descriptors shape :", df_dft.shape)
        print("RDKit descriptors shape:", df_rdkit.shape)

    id_col = df_dft.columns[0]

    if verbose:
        print(f"Using ID column: {id_col}")

    # Merge tables
    df_merged = pd.merge(df_dft, df_rdkit, on=id_col, how="inner")

    # Always show final merged size
    print(f"Final merged shape: {df_merged.shape}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_merged.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python merge_descriptors.py <DFT_csv> <RDKit_csv> <output_csv>")
        sys.exit(1)

    run(
        dft_file=sys.argv[1],
        rdkit_file=sys.argv[2],
        output_file=sys.argv[3],
        verbose=True
    )
