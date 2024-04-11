from helpers.utilities import *
from engines.alban_engine import *
import warnings

from cloudevents.http import CloudEvent
from google.cloud import storage
import glob
import os
import base64
import functions_framework


ADAL_LOGGER = logging.getLogger("adal-python")
ADAL_LOGGER.setLevel(logging.ERROR)

AZURE_LOGGER = logging.getLogger("azure")
AZURE_LOGGER.setLevel(logging.ERROR)

MSAL_LOGGER = logging.getLogger("msal")
MSAL_LOGGER.setLevel(logging.ERROR)

URLLIB_LOGGER = logging.getLogger("urllib3")
URLLIB_LOGGER.setLevel(logging.ERROR)

# set up FRP correlations, lgd, and maturity parameters
frp_corr_list = [0.18, 0.24]
frp_lgd_list = [0.25]
frp_maturity_list = [1, 2, 3]

# set up Surety correlations, lgd, and maturity parameters
surety_corr_list = [0.18, 0.24, 0.3]
surety_lgd_list = [0.3, 0.4]
surety_maturity_list = [1, 2]

# set up CWI correlations, lgd, and maturity parameters
cwi_corr_list = [0.3]
cwi_lgd_list = [0.25]
cwi_maturity_list = [1]


def set_logger():
    # Check if log
    path = os.getcwd()
    try:
        _, file_name = os.path.abspath(__file__).rsplit("/", 1)
    except ValueError:
        _, file_name = os.path.abspath(__file__).rsplit("\\", 1)
    path = f"{path}/upload/Logs/"
    file_name = file_name.replace(".py", ".log")
    log_path = path + file_name

    if not (os.path.exists(log_path)):
        os.makedirs(path)
        file = open(log_path, "w")
        file.close()

    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging_wrap(logger, log_path)


# *================================= actions =================================*
# *--------------------------------- create ----------------------------------*
def main_frp_create(args):
    # read snapshot date and write to .txt file
    args = get_snapshot_date(args)
    format_date = str(args["snapshot_date"])
    f = open("sim_date_record_file.txt", "w")
    f.write(str(format_date))
    f.close()

    # read in data from start_script path
    final_dict = pd.read_excel(args["json_path"], sheet_name=None)

    for corr in frp_corr_list:
        for lgd in frp_lgd_list:
            for maturity in frp_maturity_list:
                # read in data from start_script path
                final_dict = pd.read_excel(args["json_path"], sheet_name=None)
                args[
                    "filename"
                ] = f"EC_Tool_Input_corr{corr}lgd{lgd}maturity{maturity}.xlsx"

                # change LGD and correlation based on input in bash file
                final_dict["Portfolio"]["R"] = corr
                final_dict["Portfolio"]["PEL"] = lgd
                final_dict["RatingToPD"]["PD"] = final_dict["RatingToPD"]["PD"].apply(
                    lambda x: 1 - (1 - x) ** maturity
                )

                # change Structure Type (fails data checks if not done)
                final_dict["Structure"].replace(
                    {"QSFRP": "QS", "XLFRP": "XL"}, inplace=True
                )
                # save_combined_file(final_dict, args["filename"], args["output_path"])
                save_combined_file(final_dict, args["filename"], args["output_path"])


def main_surety_create(args):
    # read snapshot date and write to .txt file
    args = get_snapshot_date(args)
    format_date = str(args["snapshot_date"])
    f = open("sim_date_record_file.txt", "w")
    f.write(str(format_date))
    f.close()

    # read in data from start_script path
    final_dict = pd.read_excel(args["json_path"], sheet_name=None)

    for corr in surety_corr_list:
        for lgd in surety_lgd_list:
            for maturity in surety_maturity_list:
                # read in data from start_script path
                final_dict = pd.read_excel(args["json_path"], sheet_name=None)
                args[
                    "filename"
                ] = f"EC_Tool_Input_corr{corr}lgd{lgd}maturity{maturity}.xlsx"

                # change LGD and correlation based on input in bash file
                final_dict["Portfolio"]["R"] = corr
                final_dict["Portfolio"]["PEL"] = lgd
                final_dict["RatingToPD"]["PD"] = final_dict["RatingToPD"]["PD"].apply(
                    lambda x: 1 - (1 - x) ** maturity
                )

                # implement custom RI for Notional columns
                final_dict["Portfolio"]["Notional Retained"] = final_dict[
                    "Portfolio"
                ].apply(
                    lambda row: get_surety_notional_retained(
                        row["Notional"], row["Company Name"]
                    ),
                    axis=1,
                )
                final_dict["Portfolio"]["Notional Ceded"] = final_dict[
                    "Portfolio"
                ].apply(
                    lambda row: get_surety_notional_ceded(
                        row["Notional"], row["Company Name"]
                    ),
                    axis=1,
                )

                # change Structure Type (fails data checks if not done)
                final_dict["Structure"].replace(
                    {"QSSurety": "QS", "XLSurety": "XL"}, inplace=True
                )
                save_combined_file(final_dict, args["filename"], args["output_path"])


def main_cwi_create(args):
    # read snapshot date and write to .txt file
    args = get_snapshot_date(args)
    format_date = str(args["snapshot_date"])
    f = open("sim_date_record_file.txt", "w")
    f.write(str(format_date))
    f.close()

    # read in static input files
    input_parameters = pd.read_csv("./data/CWI/input_parameters.csv")
    rating_mappings = pd.read_csv("./data/CWI/Rating-Mappings.csv")
    custom_rating = pd.read_excel(
        "./data/CWI/CostingExampleMertonpricing-input.xlsx", sheet_name="RatingToPD"
    )
    # package_list_xl = pd.read_excel('./data/CWI/PackageListReport-Insurance-16-01-2024.xlsx', sheet_name=None)
    package_list_xl = pd.read_excel(
        "./data/CWI/CWI-test-input-31-12-2023.xlsx", sheet_name=None
    )
    claim_experience_xl = pd.read_excel(
        "./data/CWI/ClaimExperienceReport-Insurance-16-01-2024.xlsx", sheet_name=None
    )
    claim_experience = claim_experience_xl["Claims Experience"]
    facilities = package_list_xl["Facilities"]
    certificates = package_list_xl["Certificates"]

    # implement CWI business logic calculations
    # create Claim Experience table
    claim_experience["Time to Claim"] = calculate_days_as_years(
        claim_experience.copy(), "Bond Start Date", "Date of Loss"
    )
    claim_experience["Total Paid"] = (
        claim_experience["Paid Indemnity"]
        + claim_experience["Paid (Expenses)"]
        + claim_experience["Paid Recoveries"]
    )
    claim_experience["Total Case Estimate"] = (
        claim_experience["OSLR Indemnity"]
        + claim_experience["OSLR Expenses"]
        + claim_experience["OSLR Recoveries"]
    )
    claim_experience["Claim Year"] = pd.to_datetime(
        claim_experience["Date of Loss"]
    ).dt.year
    claim_experience = claim_experience.merge(
        certificates[["Contract Id", "Contract Value"]].copy(),
        how="left",
        left_on="Contract ID",
        right_on="Contract Id",
    )
    claim_experience["Contract Value"] = claim_experience["Contract Value"].fillna(0)
    claim_experience["Contract Value"] = claim_experience["Contract Value"].replace(
        [np.inf, -np.inf], 0
    )
    claim_experience["Claim Severity"] = (
        claim_experience["Incurred"] / claim_experience["Contract Value"]
    )
    claim_experience["Claim Severity"] = claim_experience["Claim Severity"].fillna(0)
    claim_experience["Claim Severity"] = claim_experience["Claim Severity"].replace(
        [np.inf, -np.inf], 0
    )
    claim_experience["UW Year"] = pd.to_datetime(
        claim_experience["Bond Start Date"]
    ).dt.year

    # create aggregated table from Claim Experience (contract level)
    claim_experience_contract_agg = (
        claim_experience.copy()
        .groupby(["Contract ID"])
        .agg(
            ContractID=("Contract ID", "first"),
            SumOfTotalPaid=("Total Paid", "sum"),
            SumOfTotalCaseEstimate=("Total Case Estimate", "sum"),
            SumOfIncurred=("Incurred", "sum"),
            MinOfTimeToClaim=("Time to Claim", "min"),
            SumOfTimeToClaim=("Time to Claim", "sum"),
        )
    )

    # create aggregated table from Claim Experience (client/builder level)
    claim_experience_client_agg = certificates[["Client"]].drop_duplicates()
    claim_experience_client_agg = claim_experience_client_agg.merge(
        claim_experience[["Insured", "Total Paid", "Incurred"]].copy(),
        how="left",
        left_on="Client",
        right_on="Insured",
    )
    claim_experience_client_agg = (
        claim_experience_client_agg.copy()
        .groupby(["Client"])
        .agg(
            Insured=("Insured", "first"),
            SumOfTotalPaid=("Total Paid", "sum"),
            SumOfIncurred=("Incurred", "sum"),
        )
    )
    claim_experience_client_agg["Claim Ind"] = claim_experience_client_agg.apply(
        lambda row: 1 if row["SumOfIncurred"] > 0 else 0, axis=1
    )

    # create Historic Claims Analysis table
    historic_claims = certificates[
        ["Client", "Contract Id", "Certificate Number", "Date Certificate Issued"]
    ].copy()
    historic_claims = historic_claims.merge(
        claim_experience_contract_agg[
            ["ContractID", "SumOfIncurred", "SumOfTimeToClaim"]
        ].copy(),
        how="left",
        left_on="Contract Id",
        right_on="ContractID",
    )
    historic_claims = historic_claims.merge(
        facilities[["Client", "Date Cover\n To", "Builder Risk Rating"]].copy(),
        how="left",
        on="Client",
    )
    historic_claims["Claim Ind"] = historic_claims["SumOfIncurred"].apply(
        lambda x: 1 if x > 0 else 0
    )
    historic_claims["Validation Date"] = format_date
    historic_claims["Validation Date"] = pd.to_datetime(
        historic_claims["Validation Date"]
    )
    historic_claims["Exposure Years"] = historic_claims.apply(
        lambda row: row["SumOfTimeToClaim"]
        if row["Claim Ind"] != 0
        else datedif(row["Date Certificate Issued"], row["Validation Date"]),
        axis=1,
    )
    historic_claims["Issue Year"] = pd.to_datetime(
        historic_claims["Date Certificate Issued"]
    ).dt.year

    # create Loss Modelling table
    loss_modelling = certificates.copy().merge(
        claim_experience_contract_agg[
            [
                "ContractID",
                "SumOfTotalPaid",
                "SumOfTotalCaseEstimate",
                "SumOfIncurred",
                "MinOfTimeToClaim",
            ]
        ].copy(),
        how="left",
        left_on="Contract Id",
        right_on="ContractID",
    )
    loss_modelling[
        [
            "SumOfTotalPaid",
            "SumOfTotalCaseEstimate",
            "SumOfIncurred",
            "MinOfTimeToClaim",
        ]
    ] = loss_modelling[
        [
            "SumOfTotalPaid",
            "SumOfTotalCaseEstimate",
            "SumOfIncurred",
            "MinOfTimeToClaim",
        ]
    ].fillna(
        0
    )
    loss_modelling["Certificate UW Year"] = pd.to_datetime(
        loss_modelling["Date Certificate Issued"]
    ).dt.year
    loss_modelling["Valuation Date"] = pd.to_datetime(
        input_parameters["Valuation Date"][0]
    )
    loss_modelling = loss_modelling.merge(
        facilities[
            ["Client", "UW Year", "Builder Pricing Category", "Builder Risk Rating"]
        ].copy(),
        how="left",
        on="Client",
    )
    loss_modelling["AI Credit Rating"] = loss_modelling["Builder Risk Rating"].fillna(
        loss_modelling["Builder Pricing Category"]
    )
    loss_modelling = loss_modelling.merge(
        claim_experience_client_agg[["Claim Ind"]].copy(), how="left", on="Client"
    )
    loss_modelling = loss_modelling.merge(
        rating_mappings.copy(), how="left", on="AI Credit Rating"
    )
    loss_modelling["1Y PD"] = loss_modelling.apply(
        lambda row: calculate_1y_pd(row["Claim Ind"], row["SumOfIncurred"], row["PD"]),
        axis=1,
    )
    loss_modelling["1Y PD"] = loss_modelling["1Y PD"].fillna(
        0
    )  # Facilities 'Grand Arch Homes Pty Ltd', 'Gregory Builders Pty Ltd', 'Taisa Zielak' have 2 entries
    # in the Facilities table (and only one entry has a risk rating), which results in null 1Y PD entries
    loss_modelling = loss_modelling[
        loss_modelling["1Y PD"] != 0
    ]  # filter out certificates with zero PDs (i.e. nonzero 'SumOfIncurred')
    loss_modelling["Remaining Period (Years)"] = loss_modelling.apply(
        lambda row: calculate_remaining_period(
            row["Actual Completion Date"],
            row["Estimated Completion Date"],
            row["Valuation Date"],
            row["Contract Status"],
        ),
        axis=1,
    )
    loss_modelling["Claim Frequency"] = loss_modelling.apply(
        lambda row: calculate_claim_frequency(
            row["1Y PD"],
            row["Remaining Period (Years)"],
            row["Contract Status"],
            input_parameters["Defect Rate"][0],
            input_parameters["Propensity to Claim / Scaling"][0],
        ),
        axis=1,
    )
    loss_modelling["IBNR Loss"] = loss_modelling.apply(
        lambda row: np.minimum(
            row["Contract Value"]
            * row["Claim Frequency"]
            * input_parameters["Claims Severity Rate"][0],
            row["Sum Insured Limit"],
        )
        / (
            (1 + input_parameters["Discount Rate"][0])
            ** (row["Remaining Period (Years)"] / 2)
        ),
        axis=1,
    )
    loss_modelling["Ultimate Loss"] = loss_modelling.apply(
        lambda row: row["IBNR Loss"] + row["SumOfIncurred"], axis=1
    )

    # aggregate Loss Modelling table from certifcate level to a builder level
    loss_modelling["Client"] = loss_modelling[
        "Client"
    ].str.upper()  # inconsistent capitalisations for same companies
    loss_modelling = loss_modelling.groupby("Client", as_index=False).apply(
        lambda x: get_weights(x, col="Contract Value")
    )
    loss_modelling["Weighted Claim Frequency"] = (
        loss_modelling["weight"] * loss_modelling["Claim Frequency"]
    )
    loss_modelling_agg = (
        loss_modelling.copy()
        .groupby(["Client"], as_index=False)
        .agg(
            **{"Company Name": ("Client", "first")},
            Notional=("Contract Value", "sum"),
            ClaimFrequency=("Weighted Claim Frequency", "sum"),
        )
    )

    # read in static input parameters
    loss_modelling_agg["Country"] = input_parameters["Country"][0]
    loss_modelling_agg["Cardinality"] = input_parameters["Cardinality"][0]
    loss_modelling_agg["LNL"] = input_parameters["LNL"][0]
    loss_modelling_agg["Notional Retained"] = np.nan
    loss_modelling_agg["Notional Ceded"] = np.nan
    loss_modelling_agg["Tenor"] = input_parameters["Tenor"][0]
    loss_modelling_agg["PML"] = input_parameters["PML"][0]
    loss_modelling_agg["Shape"] = input_parameters["Shape"][0]
    loss_modelling_agg["Attach"] = input_parameters["Attach"][0]
    loss_modelling_agg["Detach"] = input_parameters["Detach"][0]
    loss_modelling_agg.insert(0, "PTF", "CWI")
    loss_modelling_agg.insert(
        3,
        "Rating",
        [i for i in range(1, len(loss_modelling_agg["ClaimFrequency"]) + 1)],
    )

    # create rating mapping
    custom_rating_2 = pd.DataFrame(
        {
            "Rating": loss_modelling_agg["Rating"],
            "PD": loss_modelling_agg["ClaimFrequency"],
        }
    )
    custom_rating_2.reset_index(drop=True, inplace=True)
    custom_rating = custom_rating._append(custom_rating_2, ignore_index=True)

    for corr in cwi_corr_list:
        for lgd in cwi_lgd_list:
            for maturity in cwi_maturity_list:
                # read in data from start_script path
                final_dict = pd.read_excel(args["json_path"], sheet_name=None)
                final_dict["Portfolio"] = loss_modelling_agg
                final_dict["RatingToPD"] = custom_rating
                args[
                    "filename"
                ] = f"EC_Tool_Input_corr{corr}lgd{lgd}maturity{maturity}.xlsx"

                # change LGD and correlation based on input in bash file
                final_dict["Portfolio"]["R"] = corr
                final_dict["Portfolio"]["PEL"] = lgd
                final_dict["RatingToPD"]["PD"] = final_dict["RatingToPD"]["PD"].apply(
                    lambda x: 1 - (1 - x) ** maturity
                )

                # change Structure Type (fails data checks if not done)
                final_dict["Structure"].replace(
                    {"QSFRP": "QS", "XLFRP": "XL"}, inplace=True
                )
                save_combined_file(final_dict, args["filename"], args["output_path"])


# *-------------------------------- simulate ---------------------------------*
def main_simulate(args):
    try:
        # read saved backdating date else write current date
        if os.path.exists("sim_date_record_file.txt"):
            f = open("sim_date_record_file.txt", "r")
            today = f.read()
            f.close()
            format_date = str(today).replace("-", "")
        else:
            today = pd.Timestamp.today()
            format_date = str(today.date()).replace("-", "")
            f = open("sim_date_record_file.txt", "w")
            f.write(str(format_date))
            f.close()

        project_root = Path(os.getcwd())

        # check credit business
        if args["mode"] == "frp":
            corr_list = frp_corr_list
            lgd_list = frp_lgd_list
            maturity_list = frp_maturity_list
        elif args["mode"] == "surety":
            corr_list = surety_corr_list
            lgd_list = surety_lgd_list
            maturity_list = surety_maturity_list
        elif args["mode"] == "cwi":
            corr_list = cwi_corr_list
            lgd_list = cwi_lgd_list
            maturity_list = cwi_maturity_list

        for corr in corr_list:
            for lgd in lgd_list:
                for maturity in maturity_list:
                    # read in data from start_script path
                    file_name = project_root.joinpath(
                        args["file_path"]
                        + f"EC_Tool_Input_corr{corr}lgd{lgd}maturity{maturity}.xlsx"
                    )

                    input_data = prep_simulation_input(file_name)

                    result_dict = multiple_simulation_attempts(input_data, args["runs"])

                    # perCP_summary and PerCP_Metrics_Data
                    df_dict = get_simulation_df_dict(
                        result_dict, ["perCP_summary", "PerCP_Metrics_Data"]
                    )
                    df_dict["perCP_summary"].insert(
                        1, "DimDateSnapshotDateKey", format_date
                    )
                    df_list = list(df_dict.values())
                    save_output(
                        None,
                        df_list,
                        [
                            f"perCP_summary_corr{corr}lgd{lgd}maturity{maturity}.csv",
                            f"PerCP_Metrics_Data_corr{corr}lgd{lgd}maturity{maturity}.csv",
                        ],
                    )

        # read in gross and net losses on simulation level
        # TODO: use losses decomp returned from API and remove static files that are read in below
        gross_sim = pd.read_csv("./upload/Output/GU_DECOMP.csv")
        net_sim = pd.read_csv("./upload/Output/NET_DECOMP.csv")
        gross_sim_var = np.percentile(-gross_sim["LOSS"], 99.5)
        net_sim_var = np.percentile(-net_sim["LOSS"], 99.5)

        # create final AGResults output file
        ag_results = pd.DataFrame(
            columns=[
                "Company",
                "DimDateSnapshotDateKey",
                "PD",
                "Maturity",
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
        )
        for corr in corr_list:
            for lgd in lgd_list:
                for maturity in maturity_list:
                    perCP_df = get_per_cp_df(
                        metrics_data=f"PerCP_Metrics_Data_corr{corr}lgd{lgd}maturity{maturity}.csv",
                        summary=f"perCP_summary_corr{corr}lgd{lgd}maturity{maturity}.csv",
                        gross_var=gross_sim_var,
                        net_var=net_sim_var,
                    )
                    perCP_df = perCP_df.round(20)
                    perCP_df["Maturity"] = f"{maturity}Y"
                    ag_results = ag_results._append(perCP_df, ignore_index=True)

        save_output(None, [ag_results], ["AGResults.csv"])

    except Exception as e:
        LOGGER.error(f"main_simulate() error: {e}")


# *--------------------------------- upload ----------------------------------*


def main_upload(args):
    pass


# *================================== main ===================================*
# Triggered by PubSub Cloud Schedule
@functions_framework.cloud_event
def main(cloud_event: CloudEvent) -> None:
    """This function is triggered by http request
    """

        # Print out the data from Pub/Sub, to prove that it worked
    print(
        "Hello, " + base64.b64decode(cloud_event.data["message"]["data"]).decode() + "!"
    )

    # Configure Google Cloud Storage client - download input file
    name = "FRPSep2023-FirstLoss-Change.xlsx"
    client = storage.Client()
    temp_file = f"/tmp/{name}"
    bucket_name = "ecm_automation_disc"
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(name)
    blob.download_to_filename(temp_file)

    args = ["frp", "create", "-s", "2023-09-30", temp_file]
    ARGS = parse_arguments(args)
    main_frp_create(ARGS)
    client.close()

    # Configure Google Cloud Storage client - upload output files
    output_client = storage.Client()

    # Assign default value to output_bucket
    output_bucket = "ecm_automation_p1_output_disc"

    # Upload all files in the /tmp directory to the output bucket
    for file_path in glob.glob('/tmp/*.xlsx'):
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            print(file_name)
            output_blob_name = file_name
            output_bucket_name = "ecm_automation_p1_output_disc"

            # # Check if the file starts with 'EC'
            # if file_name.startswith('EC'):
            output_bucket = output_client.get_bucket(output_bucket_name)
            print("success")

            output_blob = output_bucket.blob(output_blob_name)
            output_blob.upload_from_filename(file_path)

            print(f"Uploaded {file_name} to Cloud Storage.")

    print("Processing complete.")

    # Delete all files in the /tmp directory
    for file_path in glob.glob('/tmp/*'):
        if os.path.isfile(file_path):
            os.remove(file_path)


    return "success"

    # set_logger()
    # ARGS = parse_arguments()

    # if ARGS["which"] == "create":
    #     LOGGER.info("*-------------- Starting Create ---------------*")
    #     if ARGS["mode"] == "frp":
    #         main_frp_create(ARGS)
    #     elif ARGS["mode"] == "surety":
    #         main_surety_create(ARGS)
    #     elif ARGS["mode"] == "cwi":
    #         main_cwi_create(ARGS)

    # elif ARGS["which"] == "simulate":
    #     LOGGER.info(f"*----------- Starting Simulate: {ARGS['mode']} " f"-----------*")
    #     warnings.filterwarnings("ignore", category=FutureWarning)
    #     if ARGS["mode"] == "frp" or ARGS["mode"] == "surety" or ARGS["mode"] == "cwi":
    #         main_simulate(ARGS)
    #     warnings.resetwarnings()

    # elif ARGS["which"] == "upload":
    #     LOGGER.info("*-------------- Starting Upload ---------------*")
    #     main_upload(ARGS)

# *===========================================================================*
