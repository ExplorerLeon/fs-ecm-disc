import json
import logging
import os
import sys
import time
from pathlib import Path
import pandas as pd
import alban_api.client.FRC_pricing_tool_client as FRC_api
import alban_api.client.configp as configp
from helpers.utilities import get_weights

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


# *================================= create ==================================*


# *================================ simulate =================================*
def update_config(api, runs):
    config_dict = configp.load_config_parameters()
    config_dict["num_scenarios"] = runs
    api.update_config(config_dict)


def prep_simulation_input(input_path):
    # Check input
    input_data, checkflag = FRC_api.import_tables_from_Excel(input_path)

    if not checkflag:
        LOGGER.error("The input sheets are not in the correct format")
        raise Exception("Incorrect input sheet format")

    _, checkflag = FRC_api.clean_import_tables(input_data)

    # this is also done when the data is uploaded to the simulation server,
    # but this helps prevent unnecessary calls
    # if not checkflag:
    # LOGGER.error("The input failed Alban's input consistency checks")
    # raise Exception("Inputs failed Alban's input consistency ")

    return input_data


def multiple_simulation_attempts(input_data, runs):
    max_tries = 5
    counter = 0
    result_check = None

    # try the api for a maximum of max_tries
    while (result_check is None) and (counter < max_tries):
        LOGGER.info(f"Attempt {counter}...")
        result_dict, result_check = simulation_session(input_data, runs, result_check)
        counter += 1

    if result_check is None:
        LOGGER.error(f"The API had no return after {max_tries} attempts. " f"Byeeeeeee")
        sys.exit(1)

    return result_dict


def simulation_session(input_data, runs, result_check):
    # create api session
    api = FRC_api.FRC_api_session()
    update_config(api, runs)

    res = api.upload_import_data(input_data)
    res_dict = json.loads(res.text)

    # this wiill fail if the response text changes
    if res.status_code == 200:
        LOGGER.info(
            f"Data uploaded successfully, with status: " f"{res_dict['status'].lower()}"
        )
    else:
        LOGGER.error(f"Unsuccessful, with status: " f"{res_dict['status'].lower()}")

    res = api.run_all()
    result_dict = json.loads(res.text)

    if result_dict["status"] == "running":
        LOGGER.info("Model running...")

    res = api.get_status()
    result_dict = json.loads(res.text)

    # api session must either return results complete or close, terminating
    # the loop
    while result_dict["info"]["status"].lower() != "results complete":
        time.sleep(30)
        res = api.get_status()
        if res:
            result_dict = json.loads(res.text)
            if result_dict["info"]["status"].lower() == "results complete":
                result_check = "Success!"
                LOGGER.info(f"Model run successfully")
        elif res is None:
            break

    if result_check == "Success!":
        result_dict = json.loads(res.text)

        num_scenarios = result_dict["config"]["num_scenarios"]

        if num_scenarios != runs:
            LOGGER.error(
                f"Config was not updated successfully,"
                f" the model was  run {num_scenarios} times"
            )

        res = api.download_result_data()
        result_dict = json.loads(res.text)

        LOGGER.info(f"Results downloaded with status: {result_dict['status']}")

    return result_dict, result_check


def get_simulation_df_dict(res_dict, df_keys):
    # converts dict of jsons to dict of dfs
    results_df_dict = FRC_api.convert_json_to_df_dict(json.dumps(res_dict))
    df_list = {df_key: results_df_dict[df_key] for df_key in df_keys}

    return df_list


def get_per_cp_df(metrics_data, summary, gross_var, net_var):
    path = Path("./upload/Output/")

    # Assumes that the relevant sheet will always be second
    perCP_output_df = pd.read_csv(path.joinpath(metrics_data))
    perCP_summary_df = pd.read_csv(path.joinpath(summary))

    # extracting gross and net
    cols = [
        "Company Name",
        "PD",
        "Rating",
        "R",
        "PEL",
        "Notional",
        "Expected Loss",
        "200Y SF Loss",
    ]
    df_gross = perCP_output_df[(perCP_output_df.Structure == "GROSS")]
    df_gross = df_gross[cols]
    df_net = perCP_output_df[(perCP_output_df.Structure == "NET")]
    df_net = df_net[cols]

    perCP_merged_df = df_gross.merge(
        df_net, how="left", on="Company Name", suffixes=["_Gross", "_Net"]
    )
    perCP_merged_df = perCP_merged_df.fillna(0)

    # combine percp_summary to add in the 75th percentile columns
    perCP_merged_df = perCP_merged_df.merge(perCP_summary_df, on="Company Name")

    # distribute total VaR to a company level
    perCP_merged_df = get_weights(
        perCP_merged_df, "200Y SF Loss_Gross", name="gross_tvar_weights"
    )
    perCP_merged_df = get_weights(
        perCP_merged_df, "200Y SF Loss_Net", name="net_tvar_weights"
    )

    perCP_merged_df["GrossVaR"] = perCP_merged_df["gross_tvar_weights"] * gross_var
    perCP_merged_df["NetVaR"] = perCP_merged_df["net_tvar_weights"] * net_var

    # final columns renaming
    perCP_merged_df = perCP_merged_df[
        [
            "Company Name",
            "DimDateSnapshotDateKey",
            "PD_Gross",
            "Rating_Gross",
            "R_Gross",
            "PEL_Gross",
            "Notional_Gross",
            "Notional_Net",
            "Expected Loss_Gross",
            "Expected Loss_Net",
            "GROSS 25 Perc Loss",
            "NET 25 Perc Loss",
            "GrossVaR",
            "NetVaR",
            "200Y SF Loss_Gross",
            "200Y SF Loss_Net",
        ]
    ]

    perCP_merged_df.columns = [
        "Company",
        "DimDateSnapshotDateKey",
        "PD",
        "Rating",
        "R",
        "PEL",
        "GrossExposure",
        "NetExposure",
        "GrossLoss",
        "NetLoss",
        "Gross75th",
        "Net75th",
        "GrossVaR",
        "NetVaR",
        "GrossTVaR",
        "NetTVaR",
    ]

    return perCP_merged_df


# *===========================================================================*
