import pandas as pd
from io import StringIO
from fastapi import UploadFile

def read_csv_to_dataframe(file: UploadFile) -> pd.DataFrame:
    """
    Reads an uploaded CSV file into a pandas DataFrame, parsing 'timestamp' as datetime.
    Args:
        file (UploadFile): The uploaded CSV file.
    Returns:
        pd.DataFrame: The parsed DataFrame with 'timestamp' as datetime.
    """
    content = file.file.read()
    content_str = content.decode('utf-8')
    df = pd.read_csv(StringIO(content_str), parse_dates=['timestamp'])
    return df
