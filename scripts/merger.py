#!/usr/bin/env python3

import sys
import time

import click
from bps_numerical.preprocessing import merge_gene_phenotype, standardize_gene_data
from loguru import logger

sys.path.append("./")
sys.path.append("../code/")
sys.path.append("./code/")


@click.command()
@click.option("--gene_csv", help="CSV with gene only information.")
@click.option("--phenotype_csv", help="CSV with only phenotype information.")
@click.option("--output_csv", help="Output CSV path to dump the merged csv into.")
@click.option(
    "--merge_column", default="Sample", help="Data id column common to both gene and phenotype."
)
def merge_and_dump(
    gene_csv: str,
    phenotype_csv: str,
    merge_column: str = "Sample",
    output_csv: str = "tmp/merged.csv",
):
    start = time.time()
    df_gene = standardize_gene_data(gene_csv)
    df_merged = merge_gene_phenotype(
        data_gene=df_gene, data_phenotype=phenotype_csv, on=merge_column
    )
    df_merged.to_csv(output_csv, index=False)
    logger.info(f"Took {time.time()-start} seconds to merge!")


def main():
    # gene_csv = (
    #     "/Users/nishparadox/dev/uah/nasa-impact/gene-experiments/data/OneDrive_1_3-21-2022/gen.csv"
    # )
    # phenotype_csv = (
    #     "/Users/nishparadox/dev/uah/nasa-impact/gene-experiments/data/OneDrive_1_3-21-2022/meta.csv"
    # )

    # df_gene = standardize_gene_data(gene_csv)

    # df_merged = merge_gene_phenotype(data_gene=df_gene, data_phenotype=phenotype_csv, on="Sample")
    # print(df_merged.shape)
    # print(df_merged.head())

    merge_and_dump()


if __name__ == "__main__":
    main()
