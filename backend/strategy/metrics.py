from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

def calculate_cagr(portfolio_values: pd.Series, periods_per_year: int = 252) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    start = portfolio_values.iloc[0]
    end = portfolio_values.iloc[-1]
    n_years = len(portfolio_values) / periods_per_year
    if start <= 0 or n_years <= 0:
        return 0.0
    cagr = (end / start) ** (1 / n_years) - 1
    return cagr * 100

def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if returns.std() == 0:
        return 0.0
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    return sharpe

def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sortino = (excess_returns.mean() / downside.std()) * np.sqrt(periods_per_year)
    return sortino

def calculate_calmar(portfolio_values: pd.Series, periods_per_year: int = 252) -> float:
    cagr = calculate_cagr(portfolio_values, periods_per_year) / 100
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values / running_max - 1).min()
    if drawdown == 0:
        return 0.0
    calmar = cagr / abs(drawdown)
    return calmar

def calculate_max_drawdown(portfolio_values: pd.Series) -> Dict[str, float]:
    running_max = portfolio_values.cummax()
    drawdowns = running_max - portfolio_values
    drawdown_pct = drawdowns / running_max * 100
    max_dd = drawdowns.max()
    max_dd_pct = drawdown_pct.max()
    return {'max_drawdown_$': max_dd, 'max_drawdown_pct': max_dd_pct}

def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    volatility = returns.std() * np.sqrt(periods_per_year)
    return volatility * 100

def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
    returns = returns.dropna()
    var = returns.quantile(confidence)
    return abs(var) * 100

def calculate_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if len(strategy_returns) != len(benchmark_returns):
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
    cov = np.cov(strategy_returns, benchmark_returns)[0][1]
    var_bench = np.var(benchmark_returns)
    if var_bench == 0:
        return 0.0
    return cov / var_bench
