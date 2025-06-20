import pandas as pd

def add_ema(df: pd.DataFrame, period: int = 14, price_col: str = 'close', out_col: str = 'ema') -> pd.DataFrame:
    """
    Adds an Exponential Moving Average (EMA) column to the DataFrame.
    """
    df[out_col] = df[price_col].ewm(span=period, adjust=False).mean()
    return df

def add_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'close', out_col: str = 'rsi') -> pd.DataFrame:
    """
    Adds a Relative Strength Index (RSI) column to the DataFrame.
    """
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[out_col] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_col: str = 'close', macd_col: str = 'macd', signal_col: str = 'macd_signal') -> pd.DataFrame:
    """
    Adds MACD and signal line columns to the DataFrame.
    """
    ema_fast = df[price_col].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow_period, adjust=False).mean()
    df[macd_col] = ema_fast - ema_slow
    df[signal_col] = df[macd_col].ewm(span=signal_period, adjust=False).mean()
    return df
