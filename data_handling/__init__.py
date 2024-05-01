""" data handling package"""
import pandas as pd
from pandas import DataFrame


def load_data(path: str, fields: list) -> DataFrame:
    """Function to load data from ESA Challenge Dataset

    Args:
        path (str): location of the file
        fields (list): list of fields of interest

    Returns:
        _type_: Dataframe
    """
    df = pd.read_csv(path)

    remove_fields = []
    for column in list(df.columns):
        if column not in fields:
            remove_fields.append(column)

    df = df.drop(columns=remove_fields, axis=1)

    return df


def convert_pc_from_log_to_dec(df: DataFrame) -> DataFrame:
    """Function to convert log 10 risks to decimals

    Args:
        df (DataFrame): input dataframe with risk in log10

    Returns:
        DataFrame: output dataframe with risk in decimals
    """
    df["pc"] = df["risk"].apply(lambda x: 10**x)
    df.drop("risk", axis="columns", inplace=True)
    return df
