import pandas as pd
import numpy as np
from logger import CustomLogger
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import plotly.express as px


def check_missing_data(df: pd.DataFrame) -> None:
    print(df.isna().sum())


def df_description(df: pd.DataFrame) -> None:
    print(df.describe())


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if (
        "x" in df.columns
        and "y" in df.columns
        and "z" in df.columns
        and "price" in df.columns
    ):
        return df[(df.x * df.y * df.z != 0) & (df.price > 0)]

    else:
        raise Exception(f"Error: Invalid Fields")


def df_scatter_matrix(df: pd.DataFrame):
    scatter_matrix(df.select_dtypes(include=["number"]), figsize=(14, 10))
    plt.show()


def df_hist(df: pd.DataFrame):
    df.hist(bins=100, figsize=(14, 10))
    plt.show()


def plot_diamonds_price_by(df: pd.DataFrame, cut_column: str):
    px.violin(
        df, x=cut_column, y="price", color=cut_column, title=f"Price by {cut_column}"
    ).show()
