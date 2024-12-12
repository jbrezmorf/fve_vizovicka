import pandas as pd
import zipfile
from io import BytesIO
import pathlib
script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"
spot_dir = script_dir / "spot_trh"

def set_datetime_index_simple(df, date_col='date', hour_col='hour'):
    """
    Create a unique datetime index by combining unique dates and hours 0-23.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least `date_col` and `hour_col`.
    date_col : str, optional
        Name of the column containing date information.
    hour_col : str, optional
        Name of the column containing hour information.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new DatetimeIndex.
    """
    # Ensure date column is in datetime format
    unique_dates = pd.to_datetime(df[date_col]).dt.date.unique()
    unique_dates = sorted(unique_dates)  # Sort unique dates

    # Generate all date_time pairs
    date_time_list = [
        pd.Timestamp(date) + pd.Timedelta(hours=hour)
        for date in unique_dates
        for hour in range(24)
    ]

    # Sort the resulting datetime list
    date_time_list.sort()

    # Set this list as the new index
    new_df = df.drop(columns=[date_col, hour_col], errors='ignore')
    new_df.index = pd.Index(date_time_list, name='date_time')
    return new_df


# Function to read and process a single zipped XLSX file
def read_process_zipped_xlsx(zip_path: pathlib.Path):
    # Open the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as z:
        xlsx_name_in_zip = zip_path.with_suffix(".xls").name
        # Read the xlsx file into memory
        with z.open(xlsx_name_in_zip) as f:
            # Read Excel with specified options:
            # - sheet_name='DT ČR': specific sheet
            # - header=5: because row 6 is the header (0-based indexing)
            # - skiprows=range(5): skip the first 5 rows before header
            # - usecols=["A","B","I"]: date, hour, price columns
            df = pd.read_excel(
                f,
                sheet_name="DT ČR",
                header=5,
                usecols=[0, 1, 8]  # A=0, B=1, I=8 (zero-based indices)
            )

    # Rename columns to something more manageable
    cols = ["date", "hour", "price"]
    for col, new_col in zip(df.columns, cols):
        print(f"Renaming column {col} to {new_col}")
    df.columns = cols
    df = set_datetime_index_simple(df, date_col='date', hour_col='hour')
    return df

def get_spot_price(years):
    spot_file = workdir / "spot_price.csv"
    if spot_file.exists():
        combined_df = pd.read_csv(spot_file, parse_dates=["date_time"])# Example usage:
    else:
        df_list = [read_process_zipped_xlsx(spot_dir / f"Rocni_zprava_o_trhu_{y}_V0.zip") for y in years]
        combined_df = pd.concat(df_list)

        # Sort by the index if needed (e.g., if times are out of order)
        combined_df.sort_index(inplace=True)
        combined_df.reset_index(inplace=True)
        combined_df.to_csv(spot_file, index=False)
    return combined_df


if __name__ == "__main__":
    get_spot_price()