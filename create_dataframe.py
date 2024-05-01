"""Import Pandas and custom packages"""

import logging
import pandas as pd
from pandas import DataFrame
import data_handling as dh
from cdm import Event, Cdm

# Configuration of logging
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

# Inputs
fields = [
    "event_id",
    "time_to_tca",
    "miss_distance",
    "risk",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",
    "mahalanobis_distance",
]

PATH = "./data/train_data.csv"


def get_cdm_from_event_id(uevent: int, listofcdms: list) -> list:
    """Get cdm based on event_id

    Args:
        event (int): event_id
    """
    x1 = []
    for report in listofcdms:
        if report.event_id == uevent:
            x1.append(report.__dict__)
    return x1


def get_closest_cdm_to_tca_by_event_id(event: Event, listofcdms: list) -> dict:
    """Retrieves last cdm of event object

    Args:
        event (Event): conjunction event object

    Returns:
        dict: last cdm
    """
    list_of_cdm_of_event = get_cdm_from_event_id(event, listofcdms)
    cdm_sorted = sorted(
        list_of_cdm_of_event, key=lambda x: x["time_to_tca"], reverse=True
    )
    closest_cdm_to_tca = cdm_sorted.pop()
    return closest_cdm_to_tca


def main() -> DataFrame:
    """Main run of script

    Returns:
        DataFrame: curated dataframe
    """
    logger.info("Start the execution of the script")
    logger.info("Loading raw data")
    # Load dataframe
    df = dh.load_data(PATH, fields)

    # Convert risk to decimals
    df = dh.convert_pc_from_log_to_dec(df)

    # Create list of Event objects
    events_list = []
    for event in df.event_id.unique():
        events_list.append(Event(event))

    # Create list of Cdm objects
    cdm_list = []
    for row in df.index:
        cdm_object = Cdm(
            event_id=df["event_id"][row],
            time_to_tca=df["time_to_tca"][row],
            miss_distance=df["miss_distance"][row],
            pc=df["pc"][row],
            relative_position_r=df["relative_position_r"][row],
            relative_position_t=df["relative_position_t"][row],
            relative_position_n=df["relative_position_n"][row],
            relative_velocity_r=df["relative_velocity_r"][row],
            relative_velocity_t=df["relative_velocity_t"][row],
            relative_velocity_n=df["relative_velocity_n"][row],
            relative_speed=df["relative_speed"][row],
            mahalanobis_distance=df["mahalanobis_distance"][row],
        )
        cdm_list.append(cdm_object)

    # Crete event list of dictionaries
    events_list_dict = []
    for event in events_list:
        events_list_dict.append(event.__dict__)

    # Create list of dataframes
    # The last CDM is lost and its miss distance is added as a column TARGET_MD
    logger.info("Building curated Dataframe")
    list_of_dataframes = []
    for event in range(len(events_list)):
        event_df = pd.DataFrame(get_cdm_from_event_id(event, cdm_list)).iloc[:-1, :]
        event_df["TARGET_MD"] = get_closest_cdm_to_tca_by_event_id(event, cdm_list).get(
            "miss_distance"
        )
        list_of_dataframes.append(event_df)

    result = pd.concat(list_of_dataframes)
    logger.info("Dataframe built")

    # Save Dataframe
    result.to_pickle("./dataframe_prueba.pkl")
    logger.info("Dataframe saved to file.")

    return result


if __name__ == "__main__":
    main()
