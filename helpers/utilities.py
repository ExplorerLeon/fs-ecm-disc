"""Contains helper functions
"""
import argparse
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# *================================= helpers =================================*
# *--------------------------------- general ---------------------------------*


def get_snapshot_date(args) -> str:
    if not args["snapshot_date"]:
        today = date.today()
        format_date = str(today.strftime("%Y-%m-%d"))
        args["snapshot_date"] = format_date
        LOGGER.info(f"No snapshot date provide using: {format_date}")

    return args


def get_weights(df, col, name="weight"):
    if len(df[col]) > 1:
        total = df[col].sum()
    else:
        total = 0
    if total == 0:
        df[name] = 0
    elif len(df) > 1 and total > 0:
        df[name] = df[col] / total

    return df


# *--------------------------------- Logger ----------------------------------*
def logging_wrap(logger, log_path):
    """Creates a file handler and console handler to add to a logger object"""
    # Need to replace this with a config file...

    # logging.basicConfig(level=logging.INFO, format = console_logging_format)
    # logging.info('Start reading database')

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    file_logging_format = (
        "[%(asctime)10s] - [%(levelname)-10s] - [%(" "name)-8s] --- %(message)s"
    )
    formatter = logging.Formatter(file_logging_format)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    console_logging_format = "[%(name)-40s] --- %(message)s"
    formatter = logging.Formatter(console_logging_format)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# *-------------------------------- argparse ---------------------------------*
def parse_arguments(args) -> Dict:
    """Parses command line arguments
    :return: parsed arguments as a dictionary
    """
    parser = argparse.ArgumentParser("Perform an action using the EC Tools code base")

    parser.add_argument(
        "mode", choices=["frp", "surety", "cwi"], default="frp", type=str.lower
    )

    subparsers = parser.add_subparsers(
        title="actions", description="possible actions using this code base"
    )

    parser_create = subparsers.add_parser(
        "create", help="create the input required by the EC tool"
    )

    parser_create.set_defaults(which="create")

    parser_create.add_argument("json_path", type=str, help="path to the input json")

    parser_create.add_argument(
        "-s",
        "--snapshot_date",
        metavar="\b",
        type=str,
        help="snapshot date to use in the DW query; format: yyyy-mm-dd",
    )

    parser_create.add_argument(
        "-o",
        "--output_path",
        metavar="\b",
        type=str,
        help="path to save created input file; default is "
        "project_abs_root/upload/Input/",
    )

    parser_create.add_argument(
        "-f",
        "--filename",
        metavar="\b",
        type=str,
        default="EC_Tool_Input.xlsx",
        help="name of the output file; default is EC_Tool_Input.xlsx",
    )

    parser_write = subparsers.add_parser(
        "write",
        help="writes the output of the EC tool required by PBI " "to the specified DB",
    )

    parser_write.set_defaults(which="write")

    parser_write.add_argument("json_path", type=str, help="path to the input json")

    parser_write.add_argument(
        "folder_path",
        type=str,
        help="path to the where the output to " "write is located",
    )

    parser_write.add_argument(
        "-s",
        "--snapshot_date",
        metavar="\b",
        type=str,
        help="snapshot date to use when writing the output; format: " "yyyy-mm-dd",
    )

    parser_simulate = subparsers.add_parser(
        "simulate", help="runs a simulation using the simulation engine"
    )
    parser_simulate.set_defaults(which="simulate")

    parser_simulate.add_argument(
        "runs", type=int, help="Number of runs for the simulation engine"
    )
    parser_simulate.add_argument(
        "file_path", type=str, help="path to input required by the " "simulation engine"
    )

    parser_upload = subparsers.add_parser(
        "upload",
        help="upload the output of the EC tool, including log "
        "files, to the specified Data Lake (Azure V1)",
    )

    parser_upload.set_defaults(which="upload")

    parser_upload.add_argument(
        "folder_path",
        type=str,
        help="path to folder with the files to "
        "upload; folder should have an input "
        "and output folder respectively",
    )

    return vars(parser.parse_args(args))


# *================================== main ===================================*
# *---------------------------------- Input ----------------------------------*


def create_combined_dict(
    df_dict, dfs_to_add, sheet_names, location
) -> Dict[str, pd.DataFrame]:
    """
    :return: combined df dict
    """
    try:
        final_sheet_order = list(df_dict.keys())
        for df, sheet_name in zip(dfs_to_add, sheet_names):
            final_sheet_order.insert(location, sheet_name)

            df_dict[sheet_name] = df

            final_dict = {key: df_dict[key] for key in final_sheet_order}
        LOGGER.info("create_combined_dict() success: Dictionaries combined")

        return final_dict
    except Exception as e:
        LOGGER.error(f"create_combined_dict() error: {e}")


def save_combined_file(df_dict, filename: str, output_path: str) -> None:
    """
    Combines the df dict into a single excel file and saves the result.
    The sheet names are dict keys

    :param df_dict: dictionary of DataFrames to combine
    :param filename: output filename
    :param output_path: output path, default is absolute project
    root/upload/Input
    :return: None
    """
    try:
        # if not output_path:
        #     project_root = Path(os.getcwd())
        #     output_path = project_root.joinpath("upload", "Input")
        # # output_path.mkdir(parents=True, exist_ok=True)
        if not output_path:
            output_path = '/tmp/'

        output_file_path = os.path.join(output_path, filename)

        xlsx_writer = pd.ExcelWriter(output_file_path, engine="openpyxl")

        for key, value in df_dict.items():
            value.to_excel(xlsx_writer, sheet_name=key, index=False)

        xlsx_writer._save()
        LOGGER.info("save_combined_file() success: Combined file saved.")
    except Exception as e:
        LOGGER.error(f"save_combined_file() error: {e}")


def calculate_claim_frequency(pd, period, status, defect_rate, propensity):
    if status == "Completed" or status == "Terminated":
        claim_freq = (
            np.minimum(pd * period, 1) * defect_rate * propensity * (period / 6)
        )
    elif status == "Issued":
        claim_freq = (
            np.minimum(6 * pd, 1) * defect_rate * propensity
            + np.minimum(np.maximum(period - 6, 0) * pd, 1) * propensity
        )
    else:
        claim_freq = 0

    return claim_freq


def calculate_remaining_period(actual, estimated, valuation, status):
    if status == "Completed" or status == "Terminated":
        try:
            period = np.maximum(6 - ((valuation.date() - actual.date()).days / 365), 0)
        except TypeError:
            period = 0
    elif status == "Issued":
        try:
            period = 6 + ((estimated.date() - valuation.date()).days / 365)
        except TypeError:
            period = 0
    else:
        period = 0

    return period


def calculate_1y_pd(claim_ind, sum_incurred, pd):  # TODO: confirm logic
    if sum_incurred > 0:
        pd_1y = 0
    elif claim_ind == 1:
        pd_1y = 1
    else:
        pd_1y = pd

    return pd_1y


def calculate_days_as_years(df, start_col, end_col):
    """
    This function calculates the difference between two columns in days,
    divides by 365, and handles potential errors.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        start_col (str): The name of the column containing the start date.
        end_col (str): The name of the column containing the end date.

    Returns:
        pandas.Series: A new Series containing the number of years between
                       start and end dates (or 0 if there's an error).
    """
    try:
        # Calculate difference in days and divide by 365 for years
        days_diff = (df[end_col] - df[start_col]).dt.days / 365
    except:
        # Handle any errors during date calculation
        days_diff = pd.Series([0] * len(df), dtype=float)
    return days_diff


def safe_divide_columns(df, numerator_col, denominator_col, result_col_name="result"):
    """
    This function performs division between columns in a DataFrame and handles potential division by zero errors.

    Args:
        df (pandas.DataFrame): The DataFrame containing the columns.
        numerator_col (str): The name of the column containing the numerator values.
        denominator_col (str): The name of the column containing the denominator values.
        result_col_name (str, optional): The name of the column to store the division results. Defaults to "result".

    Returns:
        pandas.DataFrame: The DataFrame with a new column containing the division results.
    """
    try:
        df[result_col_name] = df[numerator_col] / df[denominator_col]
    except ZeroDivisionError:
        df[result_col_name] = 0  # Fill with 0 on division by zero
    return df


def datedif(series1, series2):
    """
    This function calculates the difference in days between two pandas Series containing dates.

    Args:
        series1: The first pandas Series containing dates.
        series2: The second pandas Series containing dates.

    Returns:
        A new pandas Series containing the difference in days between the two input Series.
    """
    # Ensure both Series are datetime format
    series1 = pd.to_datetime(series1)
    series2 = pd.to_datetime(series2)

    # Calculate the difference in days and divide by days in a year (365.25 for accuracy)
    year_diff = (series2 - series1) / timedelta(days=365)

    # Round to nearest integer year
    return year_diff


def get_surety_notional_retained(notional, company_name):
    """
    This function calculates the notional value which is to be retained according
    the current Surety reinsurance structure (1 Jul 23 - 30 Jun 24).

    Args:
        notional: Exposure of the company.
        company_name: Name of the company.

    Returns:
        notional_retained: The amount of exposure retained by AssetInsure.
    """
    if (
        company_name == "CIMIC Group Limited"
        and notional >= 100_000_000
        and notional <= 400_000_000
    ):
        notional_retained = 0.075 * notional

    elif notional <= 50_000_000:
        notional_retained = 0.25 * notional

    elif notional <= 250_000_000:
        notional_retained = 0.125 * notional

    return notional_retained


def get_surety_notional_ceded(notional, company_name):
    """
    This function calculates the notional value which is to be ceded according
    the current Surety reinsurance structure (1 Jul 23 - 30 Jun 24).

    Args:
        notional: Exposure of the company.
        company_name: Name of the company.

    Returns:
        notional_ceded: The amount of exposure ceded by AssetInsure.
    """
    if (
        company_name == "CIMIC Group Limited"
        and notional >= 100_000_000
        and notional <= 400_000_000
    ):
        notional_ceded = 0.925 * notional

    elif notional <= 50_000_000:
        notional_ceded = 0.75 * notional

    elif notional <= 250_000_000:
        notional_ceded = 0.875 * notional

    return notional_ceded


# *------------------------------- Simulation --------------------------------*
def save_output(output_path, list_of_dfs, filenames):
    project_root = Path(os.getcwd())

    if not output_path:
        output_path = project_root.joinpath("upload", "Output")

    output_path.mkdir(parents=True, exist_ok=True)

    for df, filename in zip(list_of_dfs, filenames):
        df.to_csv(output_path.joinpath(filename), index=False)


def excel_to_dict(file_path):
    # Read the Excel file into a dictionary of DataFrames
    xls = pd.ExcelFile(file_path)
    sheet_dict = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

    return sheet_dict


# *===========================================================================*
