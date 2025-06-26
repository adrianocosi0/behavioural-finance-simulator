import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from behavioural_finance_simulations import simulation_helpers as sh

# --- Mobile detection and plotly rendering helpers ---
def is_mobile_browser():
    """Detect if running on a mobile browser by checking the user agent."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx and hasattr(ctx, "user_agent") and ctx.user_agent:
            ua = ctx.user_agent.lower()
            mobile_signals = ["mobi", "android", "iphone", "ipad", "phone"]
            return any(signal in ua for signal in mobile_signals)
    except Exception:
        pass
    return False

def show_plotly(fig, key=None, height=None):
    """Show Plotly as static image on mobile, interactive on desktop."""
    if is_mobile_browser():
        img_bytes = fig.to_image(format="png", scale=2, height=height)
        st.image(img_bytes)
    else:
        st.plotly_chart(fig, use_container_width=True)

def set_default_margins(fig: go.Figure, margin=dict(l=10, r=10, t=35, b=10)):
    fig.update_layout(margin=margin)
    return fig

st.title("Behavioral Finance Portfolio Simulator")

st.sidebar.header("Simulation Parameters")

# Define consistent colors for each asset
asset_colors = {
    'SPY':  '#2ca02c',   # Green
    'AGG':  '#1f77b4',   # Blue,   
    'IJS':  '#ff7f0e',   # Orange
}
# Fallback for any extra assets
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

asset_vols = {
    "SPY": 0.16,
    "IJS": 0.20,
    "AGG": 0.06,
}
asset_corr = {
    ("SPY", "IJS"): 0.85,
    ("SPY", "AGG"): -0.2,
    ("IJS", "AGG"): -0.15,
}

with st.expander("ℹ️ Quick Tips & About This App", expanded=False):
    st.markdown("""
    **This simulator lets you explore how behavioral finance strategies affect your portfolio.**
    
    - **Recency bias**: Favors assets that have recently performed well, using a softmax formula over recent returns.  
    - **Panic selling**: Sells out after a big drop, choose if re-enterering after recovery.
    - **Buy & hold**: No rebalancing, just stays invested.
    - **Rebalancing**: Periodically brings weights back to your chosen allocation.
    
    **Try adjusting the sliders in the sidebar** to see how these strategies perform from 2005–2025.
    
    **Interesting findings:**  
    - Recency bias with a 40–60 day window often does better than simple rebalancing—this is related to the momentum effect in markets!
    - Panic selling... is bad
    
    _See explanations below for each section and slider._
    """)
    
with st.expander("ℹ️ Asset Information", expanded=False):
    st.markdown("""
    **Asset Tickers Explained:**
    
    - **SPY**: 📈 S&P 500 Equity (Growth) &mdash; Tracks the performance of 500 large US companies, often used as a benchmark for US stocks.
    - **AGG**: 💵 World Aggregate Bonds (Stability) &mdash; Represents a diversified basket of global investment-grade bonds, providing portfolio stability and income.
    - **IJS**: 📊 Small Cap Value Equity (Value Tilt) &mdash; iShares S&P Small-Cap 600 Value ETF, often more volatile but with higher long-term growth potential.
    """)

@st.cache_data
def load_prices():
    return pd.read_csv('data/clean_data_SPY_AGG_IJS.csv', index_col=0, parse_dates=True, delimiter=';')
prices = load_prices()

st.markdown("### 💰 Your Initial Investment")
initial_investment = st.number_input(
    "Set your initial investment amount (€):",
    min_value=1000, max_value=1_000_000,
    value=10_000, step=100,
    format="%d"
)

# Asset/product names
asset_names = list(prices.columns)
asset_map = {
    'SPY': '📈 S&P 500 Equity (Growth)',
    'AGG': '💵 World Aggregate Bonds (Stability)',
    'IJS': '📊 Small Cap Value Equity (Value Tilt)'
}

default_weights = [0.4, 0.1, 0.5] if len(asset_names) == 3 else [1.0 / len(asset_names)] * len(asset_names)

# --- 1. Asset Allocation Controls in Main Area ---
st.header("1. Choose Your Asset Allocation")
st.info("**Tip:** Adjust the sliders to set what percentage you want in each asset. The pie chart shows your chosen mix. Weights always sum to 1.", icon="🔧")

col1, col2 = st.columns([2, 1], gap="small")

with col1:
    starting_weights = []
    for i, asset in enumerate(asset_names):
        w = st.slider(
            f"**{asset}**\n{asset_map[asset]}",
            min_value=0.0,
            max_value=1.0,
            value=np.round(float(default_weights[i]) if i < len(default_weights) else float(1.0 / len(asset_names)), 2),
            step=0.01,
            key=f"weight_{asset}"
        )
        starting_weights.append(w)
    starting_weights = np.array(starting_weights)
    if starting_weights.sum() > 0:
        starting_weights = np.round(starting_weights / starting_weights.sum(),2)
    else:
        st.warning("Sum of weights is zero, using equal allocation.")
        starting_weights = np.ones(len(asset_names)) / len(asset_names)

with col2:
    pie_chart = go.Figure(data=[go.Pie(
        labels=asset_names,
        values=starting_weights,
        hole=0.3,
        marker=dict(colors=[asset_colors.get(a, default_colors[i % len(default_colors)]) for i, a in enumerate(asset_names)]),
        textinfo='label+percent'
    )])
    pie_chart.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        height=260,  # makes it more compact; adjust as you like
    )
    show_plotly(pie_chart, key="pie_chart", height=260)
    
# After starting_weights is computed and before or after the pie chart:
agg_idx = asset_names.index("AGG") if "AGG" in asset_names else None
agg_pct = starting_weights[agg_idx] if agg_idx is not None else 0
eq_pct = sum([starting_weights[i] for i, a in enumerate(asset_names) if a in ["SPY", "IJS"]])

st.markdown(f"**Current Asset Allocation:**")
st.markdown(f"- <span style='color:#1f77b4;font-weight:bold'>Bonds (AGG)</span>: {agg_pct*100:.1f}%", unsafe_allow_html=True)
st.markdown(f"- <span style='color:#2ca02c;font-weight:bold'>Equities (SPY + IJS)</span>: {eq_pct*100:.1f}%", unsafe_allow_html=True)


est_mdd = sh.estimate_max_drawdown_from_allocation(starting_weights, asset_names, asset_vols, asset_corr)
st.markdown(
    f"""
    <div style="border-left: 4px solid #d62728; padding: 0.5em 1em; background: #f9f9fa; border-radius: 6px;">
        <span style="font-size: 1.3em;">📉 <b>Estimated Maximum Drawdown (1 year):</b></span><br>
        For your current allocation, a typical portfolio like this might experience a maximum drawdown of
        <span style="color:#d62728; font-weight:bold;">{est_mdd*100:.1f}%</span> in a bad year. Are you ready
        to possibly lose
        <span style="color:#d62728; font-weight:bold;">{est_mdd*initial_investment:.2f} Euros</span> in a year?<br>
        <span style="color: #555;">This is a quick risk estimate based on your asset mix and long-term market statistics.</span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Date Range Selection ---
st.header("2. Select Date Range")
st.info("**Tip:** Drag the slider to focus on specific years. By default, the full period, 2005-2025, is selected.", icon="📅")
all_dates = prices.index
start_date = all_dates.min().to_pydatetime()
end_date = all_dates.max().to_pydatetime()
date_range = st.slider(
    "Select date range",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    format="YYYY-MM-DD"
)

# --- Filter Data Based on Date Range ---
filtered_prices = prices.loc[pd.Timestamp(date_range[0]):pd.Timestamp(date_range[1])]

# --- Show Filtered DataFrame (AFTER date selection) ---
with st.expander("Show Filtered Price DataFrame"):
    st.write(filtered_prices)

st.markdown("---")

#mini assets performance graph
show_mini = st.checkbox("Show normalized asset performance", value=True)
if show_mini:
    st.subheader("Normalized Asset Performance")
    st.info("Shows each asset's growth, normalized to 1 at the start date. Good for spotting trends and volatility.", icon="📊")
    normed_prices = filtered_prices / filtered_prices.iloc[0]
    mini_fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, col in enumerate(asset_names):
        mini_fig.add_trace(go.Scatter(
            x=normed_prices.index,
            y=normed_prices[col],
            mode='lines',
            name=asset_map.get(col, col),
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.8,
            showlegend=True,
            hovertemplate=f"{asset_map.get(col, col)}<br>Date: %{{x|%Y-%m-%d}}<br>Norm: %{{y:.2f}}<extra></extra>"
        ))
    mini_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        height=240,
        template="plotly_white"
    )
    mini_fig = set_default_margins(mini_fig)
    show_plotly(mini_fig, key="mini_fig", height=240)

# --- User Controls ---
rebalance_freq = st.sidebar.selectbox(
    "Rebalance Frequency", options=["None", "Monthly", "Quarterly", "Yearly"], index=3,
    help="How often to rebalance your portfolio back to your chosen allocation."
)
rebalance_map = {
    "None": None,
    "Monthly": "30D",
    "Quarterly": "90D",
    "Yearly": "365D"
}

lookback_window = st.sidebar.slider(
    "Recency Bias Lookback Window (days)",
    20, 100, 60, step=5,
    help="How many days of past returns to use for measuring recency bias. 40–60 gives momentum."
)
recency_strength = st.sidebar.slider(
    "Recency Bias Strength (0=anchor, 1=full bias)",
    0.0, 1.0, 0.8, step=0.05,
    help="0 = stick to anchor allocation --> no bias, 1 = follow recency signal fully. Try 0.5–0.8 for a mix."
)
temperature = st.sidebar.slider(
    "Recency Softmax Temperature",
    0.01, 1.0, 0.1, step=0.01,
    help="How 'sharp' the recency bias is. Lower = more aggressive tilt to recent winners."
)
panic_drawdown = st.sidebar.slider(
    "Panic Drawdown Threshold (%)",
    -50, -5, -20, step=1,
    help="If the portfolio drops by this % from its peak, sell all (panic)."
) / 100
min_reentry_days = st.sidebar.slider(
    "Panic Min Days to Reentry",
    5, 60, 10, step=1,
    help="Minimum days to wait after panic before considering re-entry."
)
recovery_threshold = st.sidebar.slider(
    "Panic Recovery Threshold (%)",
    80, 100, 95, step=1,
    help="After panic, only re-enter when the portfolio would have recovered to this % of its previous peak."
) / 100

@st.cache_data
def cached_buy_and_hold(prices, weights, rebalance_freq=None):
    return sh.buy_and_hold(prices, weights, rebalance_freq=rebalance_freq)

@st.cache_data
def cached_simulate_recency_bias(prices, weights, lookback_window, recency_strength, temperature, extreme):
    return sh.simulate_recency_bias(prices, weights, lookback_window, recency_strength, temperature, extreme)

@st.cache_data
def cached_simulate_panic(prices, weights, panic_drawdown, rebalance_freq):
    return sh.simulate_panic(prices, weights, panic_drawdown, rebalance_freq)

@st.cache_data
def cached_simulate_panic_and_reentry(prices, weights, panic_drawdown, min_reentry_days, recovery_threshold, rebalance_freq):
    return sh.simulate_panic_and_reentry(prices, weights, panic_drawdown, min_reentry_days, recovery_threshold, rebalance_freq)


result_bh = cached_buy_and_hold(filtered_prices, starting_weights)
result_rebal = cached_buy_and_hold(filtered_prices, starting_weights, rebalance_freq=rebalance_map[rebalance_freq])
result_recency = cached_simulate_recency_bias(
    filtered_prices, starting_weights, lookback_window, recency_strength, temperature, extreme=False
)
result_panic = cached_simulate_panic(filtered_prices, starting_weights, panic_drawdown, rebalance_map[rebalance_freq])
result_panic_reentry = cached_simulate_panic_and_reentry(
    filtered_prices, starting_weights, panic_drawdown, min_reentry_days, recovery_threshold, rebalance_map[rebalance_freq]
)

# --- Organize results ---
series_dict = {
    'No Rebalancing (Buy & Hold)': result_bh,
    f'Rebalance': result_rebal,
    'Recency Bias': result_recency,
    'Panic Selling': result_panic,
    'Panic Selling and Re-entry': result_panic_reentry
}

# --- Select Series to Plot ---
st.header("3. Select behavioural strategies to compare")
st.markdown(
    """
    <div style="position: relative; margin-bottom: 1em;">
        <span style="font-size: 2em; color: #FF5733; font-weight: bold;">
            ⬆️⬅️
        </span>
        <span style="font-size: 1.1em; color: #FF5733; font-weight: bold; margin-left: 0.5em;">
            <u>Simulation Parameters:</u> Use the <b>SIDEBAR</b> to adapt strategies!
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
series_options = list(series_dict.keys())
selected_series = st.multiselect(
    "Select strategies to plot",
    options=series_options,
    default=["Rebalance"],  # Default to Rebalance
    help="Choose which behavioral strategies to display"
)

# --- Interactive Plot with Plotly ---
st.header("Portfolio Value Over Time")
st.info("This chart shows how your portfolio value would have evolved under different strategies. Hover for exact values and dates.", icon="📈")
fig = go.Figure()
for name in selected_series:
    res = series_dict[name]
    pv = res['portfolio_value'] * initial_investment  # <-- Scale values!
    if name == "Panic Selling":
        # Special plotting: solid while in market, dotted after panic
        diffs = pv.diff().fillna(0)
        # Find the first index (after the first value) where the value becomes flat
        panic_mask = diffs == 0
        panic_indices = panic_mask[1:].index[panic_mask[1:]]
        if len(panic_indices) > 0:
            panic_day = panic_indices[0]
            before_panic = pv.loc[:panic_day]
            after_panic = pv.loc[panic_day:]
        else:
            before_panic = pv
            after_panic = None
        fig.add_trace(go.Scatter(
            x=before_panic.index, y=before_panic.values,
            mode='lines',
            name=f'{name} (In Market)',
            opacity=0.75,
            line=dict(color='red', dash='solid', width=2)
        ))
        if after_panic is not None and len(after_panic) > 1:
            fig.add_trace(go.Scatter(
                x=after_panic.index, y=after_panic.values,
                mode='lines',
                name=f'{name} (Out of Market)',
                opacity=0.75,
                line=dict(color='red', dash='dot', width=2)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv.values,
            mode='lines',
            name=name,
            opacity=0.75,
            hovertemplate=f"{name}<br>Date: %{{x|%Y-%m-%d}}<br>Value: €%{{y:,.2f}}<extra></extra>"
        ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Portfolio Value (€)",
    legend_title="Strategy",
    hovermode="x unified",
    template="plotly_white"
)
fig = set_default_margins(fig)
show_plotly(fig, key="main_fig")

# Display final accumulated wealth for each strategy
st.subheader("🏁 Final Accumulated Wealth")
for name in selected_series:
    final_val = series_dict[name]['portfolio_value'].iloc[-1] * initial_investment
    st.markdown(f"**{name}:** €{final_val:,.2f}")

# --- 5. Summary Table and Ideas ---
def stats_dict(result):
    pv = result['portfolio_value'].loc[date_range[0]:date_range[1]]
    returns = pv.pct_change().dropna()
    cagr = sh.calculate_cagr(pv)
    vol = sh.calculate_volatility(returns)
    mdd = sh.calculate_max_drawdown(pv)['drawdown_value']
    return {
        'CAGR (%)': 100 * cagr,
        'Volatility (%)': 100 * vol,
        'Max Drawdown (%)': 100 * mdd
    }

summary = {name: stats_dict(series_dict[name]) for name in selected_series}
st.header("Performance Summary (%)")
st.info("**CAGR = Compound Annual Growth Rate. Volatility and Max Drawdown are key risk measures.**", icon="🧮")
st.dataframe(pd.DataFrame(summary).T.round(2))
