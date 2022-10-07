#!/usr/bin/env python3


from functools import cached_property
from typing import List, Optional, Type, Union

import pandas as pd
from loguru import logger
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from .misc.datatools import load_csv


def standardize_gene_data(
    fname: Union[str, pd.DataFrame],
    scaler: Optional[Type[StandardScaler]] = None,
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

        `scaler`: ```sklearn.preprocessing.StandardScaler```
            Input data scaling object. If provided, we first try to transform data
            using this one. Else, it will fit and then transform on the input data.

    Returns:
        `pd.DataFrame`
    """
    logger.info("Standardizing gene data into proper format.")
    df = fname
    if isinstance(df, str):
        df = pd.read_csv(df, chunksize=10000, iterator=False)
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

    try:
        df = (
            pd.DataFrame(scaler.transform(df.to_numpy()), columns=df.columns) if scaler else df
        ).astype(float)
    except NotFittedError:
        logger.warning("Fitting the incoming `scaler`.")
        df = (
            pd.DataFrame(scaler.fit_transform(df.to_numpy()), columns=df.columns) if scaler else df
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


class DataLoader:
    """ """

    def __init__(self, csv_gene: str, csv_phenotype: str, scaler: Optional[StandardScaler] = None):
        self.csv_gene = csv_gene
        self.csv_phenotype = csv_phenotype
        self.scaler = scaler
        self.samples = []
        self.df_genes = None
        self.df_merged = None
        self.df_attrs = None

    @property
    def gene_data(self) -> pd.DataFrame:
        logger.info(f"Loading gene only data...")
        df_genes = self.df_genes
        if df_genes is None:
            df_genes = standardize_gene_data(self.csv_gene, scaler=self.scaler)

        if "Sample" in df_genes and len(self.samples) < 1:
            self.samples = df_genes.pop("Sample")
        self.df_genes = df_genes
        logger.info(f"Gene data loaded | {df_genes.shape}")
        return df_genes

    @property
    def phenotype_attrs_data(self) -> pd.DataFrame:
        logger.info("Loading phenotype/attrs only data...")
        df_attrs = self.df_attrs
        if df_attrs is None:
            df_attrs = load_csv(self.csv_phenotype)

        self.df_attrs = df_attrs
        logger.info(f"Phenotype attrs data loaded | {df_attrs.shape}")
        return df_attrs

    def get_merged_data(
        self,
        cols_genes: Optional[List[str]] = None,
        on: str = "Sample",
    ) -> pd.DataFrame:
        logger.info("Combing gene+phenotype data...")

        # early cache check to speed up the returns
        if self.df_merged is not None and isinstance(self.df_merged, pd.DataFrame):
            return self.df_merged

        df_genes = self.gene_data
        samples = self.samples
        assert df_genes is not None and len(samples) > 0, ValueError(
            "Cannot load data. Make sure gene data and samples are valid!"
        )

        df_attrs = self.phenotype_attrs_data
        assert df_attrs is not None, ValueError(
            "Cannot load. Make sure phenotype attrs data is valid!"
        )

        cols_genes = cols_genes or self.cols_genes_all
        if self.df_merged is None:
            logger.debug(f"Using {len(cols_genes)} genes!")
            self.df_merged = merge_gene_phenotype(
                pd.concat([self.samples, self.df_genes[cols_genes]], axis=1),
                df_attrs,
                on,
            )
        logger.info(f"Merged data | {self.df_merged.shape}")
        return self.df_merged

    @property
    def cols_genes_all(self) -> List[str]:
        return self.df_genes.columns.to_list()

    @cached_property
    def cols_genes_sampled(self) -> List[str]:
        assert self.df_merged is not None, ValueError(
            "df_merged data is yet not formed. Are you sure you have done DataLoader.get_merged_data(...) first?"
        )
        cols_attrs = set(self.df_attrs.columns)

        # done to maintain order
        # otherwise set difference would have done the job efficiently
        res = []
        for _col in self.df_merged.columns:
            if _col not in cols_attrs and "sample" not in _col.lower():
                res.append(_col)
        return res


def main():
    pass


if __name__ == "__main__":
    main()
