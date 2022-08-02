#!/usr/bin/env python3


from typing import Union

import pandas as pd
from loguru import logger
from sklearn import preprocessing

from .misc.datatools import load_csv


def standardize_gene_data(
    fname: Union[str, pd.DataFrame], standardize: bool = False
) -> pd.DataFrame:
    """
    This applies:
        - transposes the original csv matrix so that each row becomes a list of
        gene features
        - changes 'gene' column name to 'Sample' (phenotype/metadata csv
        has 'Sample' as a column name)

    Args:
        fname: `str` or `pd.DataFrame`
            Input original gene csv or dataframe

    Returns:
        `pd.DataFrame`
    """
    logger.info("Standardizing gene data into proper format.")
    df = fname
    if isinstance(fname, str):
        df = pd.read_csv(fname, chunksize=10000, iterator=False)
        df = pd.concat(df, ignore_index=False)

    df = df.T.reset_index()
    df.rename(columns=df.iloc[0], inplace=True)
    df.drop(df.index[0], inplace=True)

    samples = []
    # for speed, linear search in a slice
    if "gene" in df.columns[:100]:
        df.rename(columns={"gene": "Sample"}, inplace=True)
        samples = list(df.pop("Sample"))
        df.reset_index(drop=True, inplace=True)

    df = (
        pd.DataFrame(
            preprocessing.StandardScaler().fit_transform(df.to_numpy()), columns=df.columns
        )
        if standardize
        else df
    ).astype(float)
    df.insert(0, "Sample", samples, True)
    return df.reset_index(drop=True)


def merge_gene_phenotype(
    data_gene: Union[str, pd.DataFrame],
    data_phenotype: Union[str, pd.DataFrame],
    on: str = "Sample",
) -> pd.DataFrame:
    """
    Merge gene and metadata/phenotype dataframes into single
    based on a common column `on`.
    """
    logger.info("Merging gene-phenotype dataframes...")
    data_gene = load_csv(data_gene)
    data_phenotype = load_csv(data_phenotype)
    return pd.merge(data_phenotype, data_gene, on=on, how="outer").reset_index(drop=True)


def main():
    pass


if __name__ == "__main__":
    main()
