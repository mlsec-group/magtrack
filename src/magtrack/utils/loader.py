import os.path
import zipfile
from os import PathLike
from pickle import UnpicklingError

import pandas as pd


def read_pickle(file_path: PathLike[str]) -> pd.DataFrame:
    """Read a pickle file, trying all supported compressions until one succeeds."""
    for compression in [None, 'infer', 'zstd', 'zip', 'gzip', 'bz2', 'xz', 'tar']:
        try:
            return pd.read_pickle(file_path, compression=compression)
        except UnpicklingError:
            continue
    raise UnpicklingError(f"Could not unpickle the file: {file_path}")


def get_metadata_from_dataset(file_path: PathLike[str]) -> pd.DataFrame:
    """
    Get metadata CSV file from a .pkl dataset.

    Parameters:
    - file_path: PathLike[str], path to the pkl file

    Returns:
    - pandas DataFrame, containing the extracted metadata
    """
    metadata_csv_path = os.path.join(
        os.path.dirname(file_path),
        os.path.basename(file_path).replace('.pkl', '.meta.csv')
    )
    meta_df = pd.read_csv(metadata_csv_path)
    return meta_df


def get_data_from_zipfile_by_name(file: str, file_name: str) -> pd.DataFrame:
    """
    Extracts data from a defined file from a recording zip archive and converts it into a DataFrame.

    Parameters:
    - file: str, path to the zip file
    - file_name: str, name of the file within the zip archive containing the data

    Returns:
    - pandas DataFrame, containing the extracted data
    """
    with zipfile.ZipFile(file, 'r') as zip_file:
        with zip_file.open(file_name) as data_file:
            return pd.read_csv(data_file)
