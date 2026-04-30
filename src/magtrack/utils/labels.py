import zipfile

import pandas as pd


def load_labels_table(zip_path: str) -> pd.DataFrame:
    """Load Labels.csv from a zip file and return a pivoted DataFrame.

    Each row represents a station stop with its arrival and departure
    times (in seconds from journey start).

    Parameters
    ----------
    zip_path : str
        Path to the zip archive containing ``Labels.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``trip``, ``station``, ``arrival``, ``departure``.
        If a trip changes at a station, that station appears in two rows:
        one with ``departure=NaN`` (arrival on the old trip) and one with
        ``arrival=NaN`` (departure on the new trip).
    """
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("Labels.csv") as f:
            df = pd.read_csv(f)

    result = (
        df.pivot_table(
            index=["journey", "trip", "station"],
            columns="type",
            values="seconds_from_journey_start",
            aggfunc="first",
            sort=False,
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    result = result[["trip", "station", "arrival", "departure"]]
    trip_order = result.groupby("trip")["departure"].min().rename("_trip_min_dep")
    result = result.join(trip_order, on="trip")
    result = result.sort_values(
        by=["_trip_min_dep", "departure"], na_position="last"
    ).drop(columns="_trip_min_dep").reset_index(drop=True)
    return result
