import pandas as pd

def load_and_sort_csv(path: str) -> pd.DataFrame:
    """Loads a CSV file, parses 'Date' column, and sets it as index.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame with DatetimeIndex.
    """
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.index = pd.to_datetime(df.index)
    return df
