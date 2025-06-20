import pandas as pd
import numpy as np
import re
from typing import List

def evaluate_logic(df: pd.DataFrame, conditions: List[str], logic: str) -> pd.Series:
    """
    Evaluates a list of conditions and a logic string on a DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        conditions (List[str]): List of condition strings, e.g., ['EMA_20 > EMA_50', 'RSI_14 < 30']
        logic (str): Logic string using COND1, COND2, etc., e.g., 'COND1 AND COND2'
    Returns:
        pd.Series: Boolean Series where the logic is True.
    """
    cond_results = {}
    for idx, cond in enumerate(conditions):
        cond_name = f'COND{idx+1}'
        # Use pandas.eval for safe evaluation
        cond_results[cond_name] = df.eval(cond)
    # Replace logic string with numpy logical operators
    logic_eval = logic.upper()
    logic_eval = re.sub(r'AND', '&', logic_eval)
    logic_eval = re.sub(r'OR', '|', logic_eval)
    logic_eval = re.sub(r'NOT', '~', logic_eval)
    # Build the final boolean Series
    local_dict = cond_results.copy()
    result = eval(logic_eval, {"__builtins__": None, "np": np}, local_dict)
    return result
