import pandas as pd
from typing import List, Dict, Any
from .metrics import (
    calculate_cagr, calculate_sharpe, calculate_sortino, calculate_calmar,
    calculate_max_drawdown, calculate_volatility, calculate_var, calculate_beta
)

def execute_strategy(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    order_type: str = 'market',
    initial_balance: float = 10000.0,
    position_size: float = 1.0,
    benchmark_returns: pd.Series = None
) -> Dict[str, Any]:
    """
    Simulate order execution and track portfolio performance.
    Args:
        df (pd.DataFrame): DataFrame with price and indicators.
        entry_signal (pd.Series): Boolean Series for entry points.
        exit_signal (pd.Series): Boolean Series for exit points.
        order_type (str): 'market' or 'limit'.
        initial_balance (float): Starting portfolio balance.
        position_size (float): Number of units per trade.
    Returns:
        Dict[str, Any]: Trade list and performance summary.
    """
    trades = []
    in_position = False
    entry_price = 0.0
    balance = initial_balance
    equity_curve = []
    max_equity = initial_balance
    drawdown = 0.0
    entry_time = None
    for i in range(len(df)):
        price = df['close'].iloc[i]
        if not in_position and entry_signal.iloc[i]:
            in_position = True
            entry_price = price if order_type == 'market' else df['open'].iloc[i+1] if i+1 < len(df) else price
            entry_time = df['timestamp'].iloc[i]
            trades.append({'type': 'entry', 'price': entry_price, 'timestamp': entry_time})
        elif in_position and exit_signal.iloc[i]:
            exit_price = price if order_type == 'market' else df['open'].iloc[i+1] if i+1 < len(df) else price
            pnl = (exit_price - entry_price) * position_size
            balance += pnl
            exit_time = df['timestamp'].iloc[i]
            trades.append({
                'type': 'exit',
                'price': exit_price,
                'timestamp': exit_time,
                'pnl': pnl,
                'balance': balance
            })
            in_position = False
            entry_time = None
        equity_curve.append(balance)
        max_equity = max(max_equity, balance)
        drawdown = max(drawdown, (max_equity - balance))
    # Build portfolio value and returns series
    portfolio_values = pd.Series(equity_curve, index=df.index)
    returns = portfolio_values.pct_change().dropna()
    # Metrics
    max_dd = calculate_max_drawdown(portfolio_values)
    cagr = calculate_cagr(portfolio_values)
    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    calmar = calculate_calmar(portfolio_values)
    volatility = calculate_volatility(returns)
    var = calculate_var(returns)
    beta = calculate_beta(returns, benchmark_returns) if benchmark_returns is not None else None
    # Trade stats
    total_trades = len([t for t in trades if t['type'] == 'exit'])
    total_pnl = balance - initial_balance
    win_rate = (sum(1 for t in trades if t.get('pnl', 0) > 0 and t['type'] == 'exit') / total_trades * 100) if total_trades else 0.0
    summary = {
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'max_drawdown_$': max_dd['max_drawdown_$'],
        'max_drawdown_pct': max_dd['max_drawdown_pct'],
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Volatility': volatility,
        'VaR_95': var,
        'Beta': beta,
        'final_balance': balance
    }
    return {'trades': trades, 'summary': summary}
