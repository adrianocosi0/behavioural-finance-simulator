import numpy as np
import pandas as pd
from scipy.stats import norm

def buy_and_hold(prices: pd.DataFrame, weights: np.ndarray, rebalance_freq=None) -> dict:
    """
    Simulate a simple buy-and-hold portfolio.
    Optionally rebalances at a given frequency (e.g. '365D' for yearly).
    Returns a dict with 'portfolio_value' and optionally other stats.
    """
    # Assume prices is daily, weights is array with sum==1, no cash flows
    prices = prices.ffill().dropna()
    returns = prices.pct_change().fillna(0)
    
    if rebalance_freq is None:
        # One-time allocation
        cum_returns = (returns + 1).cumprod()
        portfolio_value = (cum_returns * weights).sum(axis=1)
    else:
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        current_weights = weights.copy()
        value = 1.0
        last_rebalance = prices.index[0]
        for date in prices.index:
            if (date - last_rebalance) >= pd.Timedelta(rebalance_freq):
                current_weights = weights.copy()
                last_rebalance = date
            day_return = returns.loc[date]
            value = value * (1 + (day_return * current_weights).sum())
            portfolio_value.loc[date] = value
    return {'portfolio_value': portfolio_value}

def calculate_cagr(portfolio_value: pd.Series) -> float:
    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    return (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1

def calculate_volatility(portfolio_returns: pd.Series) -> float:
    return portfolio_returns.std() * np.sqrt(252)

def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """
    Calculate drawdown series, maximum date and value
    """
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    return {'drawdown_data':drawdown, 'drawdown_date':drawdown.index[drawdown.argmin()], 'drawdown_value':drawdown.min()}

def compute_portfolio_value(prices, weights):
    normed = prices / prices.iloc[0]
    weighted = normed * weights
    portfolio = weighted.sum(axis=1)
    return portfolio

def simulate_panic(
    prices: pd.DataFrame,
    weights: np.ndarray,
    panic_drawdown: float = -0.2,
    rebalance_freq: str = "365D"  # default: yearly
) -> dict:
    """
    Simulate selling out to cash if drawdown exceeds a threshold.
    Always rebalances at the specified frequency until panic triggers.
    After panic, portfolio stays in cash (flat value).
    Returns dict with 'portfolio_value' only.
    """
    prices = prices.ffill().dropna()
    returns = prices.pct_change().fillna(0)
    portfolio_value = pd.Series(index=prices.index, dtype=float)
    current_weights = weights.copy()
    value = 1.0
    last_rebalance = prices.index[0]
    in_market = True

    for i, date in enumerate(prices.index):
        if in_market and ((date - last_rebalance) >= pd.Timedelta(rebalance_freq) or i == 0):
            current_weights = weights.copy()
            last_rebalance = date

        if in_market:
            day_return = returns.loc[date]
            value = value * (1 + (day_return * current_weights).sum())
            # Calculate drawdown from running max so far
            if i > 0:
                running_max = portfolio_value.iloc[:i].max()
                drawdown = (value - running_max) / running_max if running_max > 0 else 0
                if drawdown < panic_drawdown:
                    in_market = False
                    value = value  # Flat value from now on
        # If not in market, value stays flat
        portfolio_value.iloc[i] = value

    return {'portfolio_value': portfolio_value}

def compute_portfolio_vol(weights, asset_names, vols, corr):
    # weights: list or np.array, asset_names: list of asset str
    weights = np.array(weights)
    n = len(asset_names)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = vols[asset_names[i]] ** 2
            else:
                pair = tuple(sorted((asset_names[i], asset_names[j])))
                rho = corr.get(pair, 0)
                cov[i, j] = vols[asset_names[i]] * vols[asset_names[j]] * rho
    port_var = weights @ cov @ weights
    return np.sqrt(port_var)
    
def var_gaussian(portfolio_returns, level=5, modified=True, period = 1):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = portfolio_returns.skew()
        k = portfolio_returns.kurtosis()
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(portfolio_returns.mean()*period + z*portfolio_returns.std()*np.sqrt(period))

def compute_cdarr(returns, alpha=0.05):
    # Conditional drawdown at risk
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    drawdowns = drawdowns.abs()
    # Select the worst alpha% drawdowns
    threshold = drawdowns.quantile(1 - alpha)
    cdarr = drawdowns[drawdowns >= threshold].mean()
    return cdarr

def rolling_max_drawdown(returns, window=252):
    cum_returns = (1 + returns).cumprod()
    # Calculate rolling max for the window
    roll_max = cum_returns.rolling(window, min_periods=1).max()
    roll_drawdown = (cum_returns - roll_max) / roll_max
    # The rolling minimum drawdown in each window is the max drawdown
    max_dd = roll_drawdown.rolling(window, min_periods=1).min()
    return max_dd

def compute_cdarr_rolling_mdd(returns, window=252, alpha=0.05):
    mdds = rolling_max_drawdown(returns, window)
    # Take the worst alpha% of rolling MDDs
    threshold = mdds.quantile(1 - alpha)
    cdarr = mdds[mdds <= threshold].mean()  # MDDs are negative
    return abs(cdarr)  # Return as positive drawdown

def estimate_max_drawdown_from_allocation(weights, asset_names, asset_vols, asset_corr):
    sigma = compute_portfolio_vol(weights, asset_names, asset_vols, asset_corr)
    expected_mdd = 2 * sigma  # Use k=2 to check for resistence to shocks
    return expected_mdd
    
def simulate_panic_and_reentry(
    prices: pd.DataFrame,
    weights: np.ndarray,
    panic_drawdown: float = -0.2,
    min_reentry_days: int = 10,
    recovery_threshold: float = 0.95,
    rebalance_freq: str = "365D"
) -> dict:
    prices = prices.ffill().dropna()
    returns = prices.pct_change().fillna(0)
    portfolio_value = pd.Series(index=prices.index, dtype=float)
    current_weights = weights.copy()
    value = 1.0
    last_rebalance = prices.index[0]
    in_market = True
    cash_value = None
    panic_day = None
    prev_peak = None
    last_entry_idx = 0

    # Extra: Track what value would have been if stayed in market
    would_be_value = value

    for i, date in enumerate(prices.index):
        if in_market and ((date - last_rebalance) >= pd.Timedelta(rebalance_freq) or i == 0):
            current_weights = weights.copy()
            last_rebalance = date

        day_return = returns.iloc[i]

        if in_market:
            value = value * (1 + (day_return * current_weights).sum())
            running_max = portfolio_value.iloc[last_entry_idx:i].max() if i > 0 else value
            drawdown = (value - running_max) / running_max if running_max > 0 else 0
            portfolio_value.iloc[i] = value

            if drawdown < panic_drawdown:
                # Panic! Sell all, move to cash
                in_market = False
                cash_value = value
                panic_day = date
                prev_peak = running_max
                would_be_value = value  # reset virtual tracker to cash value
        else:
            # Out of market: portfolio stays in cash
            portfolio_value.iloc[i] = cash_value
            # Track what would have happened if we'd stayed invested (virtual value)
            would_be_value = would_be_value * (1 + (day_return * weights).sum())

            # Check for re-entry conditions
            if (date - panic_day).days >= min_reentry_days and would_be_value >= (recovery_threshold * prev_peak):
                in_market = True
                value = cash_value  # Re-enter at cash value
                panic_day = None
                prev_peak = value
                last_rebalance = date
                portfolio_value.iloc[i] = value  # update on re-entry day
                would_be_value = value
                last_entry_idx = i

    return {'portfolio_value': portfolio_value}
    
def soft_recency_bias_softmax_blend_with_anchor(
    prices: pd.DataFrame,
    lookback_window: int = 60,
    recency_strength: float = 0.5,
    temperature: float = 0.1,
    anchor_weights: pd.Series = None
):
    """
    Blends softmax (soft recency) weights with anchor (strategic) weights.

    prices: DataFrame of price history (assets in columns)
    lookback_window: How many periods to look back for returns
    recency_strength: 0 = pure anchor weight, 1 = pure softmax recency bias
    temperature: Softmax sharpness (lower = more extreme bias)
    anchor_weights: Series with asset weights, index matching prices.columns, sums to 1
    """
    # Recent returns
    recent_returns = (prices.iloc[-1] / prices.iloc[-lookback_window]) - 1

    # Softmax transformation
    exp_returns = np.exp(recent_returns / temperature)
    softmax_weights = exp_returns / exp_returns.sum()

    # Use anchor weights (strategic allocation) as baseline
    if anchor_weights is None:
        anchor_weights = pd.Series(
            np.ones(len(prices.columns)) / len(prices.columns),
            index=prices.columns
        )

    # Blend
    weights = recency_strength * softmax_weights + (1 - recency_strength) * anchor_weights
    weights = weights / weights.sum()  # Ensure weights sum to 1 (for numeric stability)
    return weights

def simulate_recency_bias(
    prices: pd.DataFrame, 
    weights: np.ndarray, 
    lookback_window: int = 60, 
    extreme: bool = True,
    recency_strength: float = 1.0,
    temperature: float = 0.1
) -> dict:
    """
    Simulate increasing allocation to recent winners every lookback_window days.
    If extreme=True, use ranking-based weighting (original method).
    If extreme=False, use a blend of softmax-based theoretical weighting and considering anchor weights (soft_recency_bias_softmax_blend_with_anchor).
    """
    prices = prices.ffill().dropna()
    returns = prices.pct_change().fillna(0)
    portfolio_value = pd.Series(index=prices.index, dtype=float)
    value = 1.0
    rebal_dates = prices.index[::lookback_window]
    current_weights = weights.copy()
    asset_names = prices.columns
    for i, date in enumerate(prices.index):
        if date in rebal_dates:
            if i >= lookback_window:
                if extreme:
                    # Extreme method: ranking
                    past_returns = (prices.iloc[i-lookback_window:i] / prices.iloc[i-lookback_window]) - 1
                    asset_returns = past_returns.iloc[-1]
                    ranked = asset_returns.rank()
                    current_weights = ranked / ranked.sum()
                else:
                    # Theoretical method: softmax blend
                    anchor_weights = pd.Series(weights, index=asset_names)
                    current_weights = soft_recency_bias_softmax_blend_with_anchor(
                        prices.iloc[:i+1],
                        lookback_window=lookback_window,
                        recency_strength=recency_strength,
                        temperature=temperature,
                        anchor_weights=anchor_weights
                    )
        day_return = returns.iloc[i]
        value = value * (1 + (day_return * current_weights).sum())
        portfolio_value.iloc[i] = value
    return {'portfolio_value': portfolio_value}
