import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Silver Momentum Dashboard | Michael Oliver Signals",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS FOR INSTITUTIONAL STYLING
# ==============================================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E3A5F;
        --secondary-color: #4A90A4;
        --accent-color: #C0C0C0;
        --bullish-color: #00C853;
        --bearish-color: #FF1744;
        --warning-color: #FFB300;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.3rem;
        padding-top: 1rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    
    /* Card styling */
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #1E3A5F;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bullish-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 4px solid #00C853;
    }
    
    .bearish-card {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 4px solid #FF1744;
    }
    
    .neutral-card {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        border-left: 4px solid #FFB300;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: white;
    }
    
    /* Status indicators */
    .status-bullish {
        color: #00C853;
        font-weight: bold;
    }
    
    .status-bearish {
        color: #FF1744;
        font-weight: bold;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #E3F2FD;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1976D2;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        period: Time period for historical data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {str(e)}")
        return pd.DataFrame()


def calculate_roc(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Rate of Change (ROC) momentum oscillator.
    
    ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        ROC values as percentage
    """
    return ((series - series.shift(period)) / series.shift(period)) * 100


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        series: Price series
        period: RSI period
        
    Returns:
        RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def find_local_peaks(series: pd.Series, window: int = 5) -> list:
    """
    Find local peaks in a price series.
    
    Args:
        series: Price series
        window: Window size for peak detection
        
    Returns:
        List of peak indices
    """
    peaks = []
    for i in range(window, len(series) - window):
        if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1)) and \
           all(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1)):
            peaks.append(i)
    return peaks


def analyze_pullback(price_series: pd.Series, peak_window: int = 5) -> dict:
    """
    Analyze the current pullback from recent peak.
    
    Args:
        price_series: Price series
        peak_window: Window for peak detection
        
    Returns:
        Dictionary with pullback analysis
    """
    result = {
        'peak_date': None,
        'peak_price': None,
        'pullback_days': 0,
        'pullback_pct': 0.0,
        'is_pullback': False,
        'current_price': price_series.iloc[-1]
    }
    
    peaks = find_local_peaks(price_series, window=peak_window)
    
    if not peaks:
        # If no local peaks found, find the highest price in last 60 days
        recent_data = price_series.tail(60)
        peak_idx = recent_data.idxmax()
        result['peak_date'] = peak_idx
        result['peak_price'] = recent_data.loc[peak_idx]
    else:
        # Get the most recent significant peak
        peak_idx = peaks[-1]
        result['peak_date'] = price_series.index[peak_idx]
        result['peak_price'] = price_series.iloc[peak_idx]
    
    current_price = price_series.iloc[-1]
    
    if result['peak_price'] and current_price < result['peak_price']:
        result['is_pullback'] = True
        result['pullback_pct'] = ((result['peak_price'] - current_price) / result['peak_price']) * 100
        result['pullback_days'] = (price_series.index[-1] - result['peak_date']).days
    
    return result


def calculate_correlation(series1: pd.Series, series2: pd.Series, window: int = 30) -> float:
    """
    Calculate rolling correlation between two series.
    
    Args:
        series1: First price series
        series2: Second price series
        window: Correlation window
        
    Returns:
        Correlation coefficient
    """
    combined = pd.DataFrame({
        's1': series1,
        's2': series2
    }).dropna()
    
    if len(combined) < window:
        return np.nan
    
    return combined['s1'].tail(window).corr(combined['s2'].tail(window))


def get_momentum_signal(roc: float, rsi: float) -> tuple:
    """
    Determine overall momentum signal.
    
    Args:
        roc: Current ROC value
        rsi: Current RSI value
        
    Returns:
        Tuple of (signal_text, signal_color)
    """
    if roc > 5 and rsi > 60:
        return "Strong Bullish", "#00C853"
    elif roc > 0 and rsi > 50:
        return "Bullish", "#4CAF50"
    elif roc < -5 and rsi < 40:
        return "Strong Bearish", "#FF1744"
    elif roc < 0 and rsi < 50:
        return "Bearish", "#EF5350"
    else:
        return "Neutral", "#FFB300"


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/silver-bars.png", width=80)
    st.title("Dashboard Settings")
    st.markdown("---")
    
    # Momentum Parameters
    st.subheader("📊 Momentum Parameters")
    roc_period = st.slider(
        "ROC Period",
        min_value=5,
        max_value=30,
        value=14,
        step=1,
        help="Rate of Change lookback period in days"
    )
    
    rsi_period = st.slider(
        "RSI Period",
        min_value=7,
        max_value=21,
        value=14,
        step=1,
        help="Relative Strength Index period"
    )
    
    gsr_ma_period = st.slider(
        "GSR Moving Average",
        min_value=20,
        max_value=100,
        value=50,
        step=5,
        help="Gold/Silver Ratio moving average period"
    )
    
    st.markdown("---")
    
    # Support Levels
    st.subheader("🎯 Support Levels")
    support_level = st.number_input(
        "Concrete Support ($)",
        min_value=20.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
        help="Michael Oliver's concrete support floor"
    )
    
    resistance_level = st.number_input(
        "Key Resistance ($)",
        min_value=30.0,
        max_value=150.0,
        value=80.0,
        step=1.0,
        help="Major resistance level"
    )
    
    st.markdown("---")
    
    # Data Settings
    st.subheader("📅 Data Settings")
    lookback_options = {
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "3 Years": 1095
    }
    lookback_selection = st.selectbox(
        "Historical Lookback",
        options=list(lookback_options.keys()),
        index=1
    )
    lookback_days = lookback_options[lookback_selection]
    
    st.markdown("---")
    
    # Refresh
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        auto_refresh = st.checkbox("Auto", value=False, help="Auto-refresh every 5 minutes")
    
    st.markdown("---")
    st.caption("📊 Data: Yahoo Finance")
    st.caption(f"⏰ Updated: {datetime.now().strftime('%H:%M:%S')}")


# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

# Header
st.markdown('<h1 class="main-header">🥈 Michael Oliver Silver Momentum Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time tracking of silver momentum signals, GSR breakdown, and market correlations</p>', unsafe_allow_html=True)

# Ticker definitions
TICKERS = {
    "Silver_Futures": "SI=F",
    "Silver_ETF": "SLV",
    "Gold_Futures": "GC=F",
    "Gold_ETF": "GLD",
    "SP500": "^GSPC",
    "DXY": "DX-Y.NYB"
}

# Fetch data with loading indicator
with st.spinner("📊 Fetching real-time market data..."):
    # Try futures first, fall back to ETFs
    silver_data = fetch_market_data(TICKERS["Silver_Futures"], period=f"{lookback_days}d")
    if silver_data.empty:
        silver_data = fetch_market_data(TICKERS["Silver_ETF"], period=f"{lookback_days}d")
        silver_source = "SLV ETF"
    else:
        silver_source = "SI=F Futures"
    
    gold_data = fetch_market_data(TICKERS["Gold_Futures"], period=f"{lookback_days}d")
    if gold_data.empty:
        gold_data = fetch_market_data(TICKERS["Gold_ETF"], period=f"{lookback_days}d")
        gold_source = "GLD ETF"
    else:
        gold_source = "GC=F Futures"
    
    sp500_data = fetch_market_data(TICKERS["SP500"], period=f"{lookback_days}d")
    dxy_data = fetch_market_data(TICKERS["DXY"], period=f"{lookback_days}d")

# Validate data
if silver_data.empty:
    st.error("⚠️ Unable to fetch silver data. Please check your internet connection and try again.")
    st.stop()

# ==============================================================================
# CALCULATE ALL METRICS
# ==============================================================================

# Current prices
current_silver = float(silver_data['Close'].iloc[-1])
prev_silver = float(silver_data['Close'].iloc[-2])
silver_change_pct = ((current_silver - prev_silver) / prev_silver) * 100

current_gold = float(gold_data['Close'].iloc[-1]) if not gold_data.empty else None
current_sp500 = float(sp500_data['Close'].iloc[-1]) if not sp500_data.empty else None

# Momentum calculations
silver_roc = calculate_roc(silver_data['Close'], roc_period)
silver_rsi = calculate_rsi(silver_data['Close'], rsi_period)
current_roc = float(silver_roc.iloc[-1]) if not pd.isna(silver_roc.iloc[-1]) else 0
current_rsi = float(silver_rsi.iloc[-1]) if not pd.isna(silver_rsi.iloc[-1]) else 50

# Gold/Silver Ratio
if not gold_data.empty:
    # Align dates
    common_dates = silver_data.index.intersection(gold_data.index)
    gsr = gold_data.loc[common_dates, 'Close'] / silver_data.loc[common_dates, 'Close']
    gsr_ma = gsr.rolling(window=gsr_ma_period).mean()
    current_gsr = float(gsr.iloc[-1])
    current_gsr_ma = float(gsr_ma.iloc[-1]) if not pd.isna(gsr_ma.iloc[-1]) else current_gsr
    gsr_signal = "Bullish" if current_gsr < current_gsr_ma else "Bearish"
else:
    gsr = pd.Series()
    current_gsr = None
    current_gsr_ma = None
    gsr_signal = "N/A"

# S&P 500 analysis
if not sp500_data.empty:
    sp500_52w_high = float(sp500_data['Close'].rolling(window=252, min_periods=1).max().iloc[-1])
    sp500_52w_low = float(sp500_data['Close'].rolling(window=252, min_periods=1).min().iloc[-1])
    sp500_distance_from_high = ((current_sp500 - sp500_52w_high) / sp500_52w_high) * 100
    crash_target = sp500_52w_high * 0.5
else:
    sp500_52w_high = None
    sp500_distance_from_high = None
    crash_target = None

# Pullback analysis
pullback_info = analyze_pullback(silver_data['Close'])

# Momentum signal
momentum_signal, signal_color = get_momentum_signal(current_roc, current_rsi)

# Support/Resistance analysis
distance_to_support = ((current_silver - support_level) / support_level) * 100
distance_to_resistance = ((resistance_level - current_silver) / current_silver) * 100


# ==============================================================================
# KEY METRICS ROW
# ==============================================================================

st.markdown("### 📈 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="🥈 Silver Price",
        value=f"${current_silver:.2f}",
        delta=f"{silver_change_pct:+.2f}%",
        delta_color="normal"
    )

with col2:
    delta_color = "normal" if current_roc > 0 else "inverse"
    st.metric(
        label=f"📊 ROC ({roc_period})",
        value=f"{current_roc:.2f}%",
        delta=momentum_signal,
        delta_color="off"
    )

with col3:
    if current_gsr:
        gsr_delta = "vs MA" if current_gsr < current_gsr_ma else "vs MA"
        st.metric(
            label="⚖️ Gold/Silver Ratio",
            value=f"{current_gsr:.1f}",
            delta=f"{gsr_signal} ({current_gsr_ma:.1f} MA)",
            delta_color="normal" if gsr_signal == "Bullish" else "inverse"
        )

with col4:
    st.metric(
        label=f"💪 RSI ({rsi_period})",
        value=f"{current_rsi:.1f}",
        delta="Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral",
        delta_color="off"
    )

with col5:
    if current_sp500:
        st.metric(
            label="📉 S&P 500",
            value=f"{current_sp500:,.0f}",
            delta=f"{sp500_distance_from_high:.1f}% from high",
            delta_color="inverse" if sp500_distance_from_high < -5 else "normal"
        )

st.markdown("---")

# ==============================================================================
# MAIN CHART TABS
# ==============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Silver Momentum",
    "⚖️ Gold/Silver Ratio",
    "🎯 Support Monitor",
    "📉 Pullback Tracker",
    "📊 S&P 500 Correlation"
])

# ------------------------------------------------------------------------------
# TAB 1: Silver Momentum
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(f"Silver Price & {roc_period}-Period Momentum Oscillator")
    
    fig1 = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f"Silver Price ({silver_source})",
            f"{roc_period}-Period Rate of Change (ROC)",
            f"{rsi_period}-Period RSI"
        )
    )
    
    # Candlestick chart
    fig1.add_trace(
        go.Candlestick(
            x=silver_data.index,
            open=silver_data['Open'],
            high=silver_data['High'],
            low=silver_data['Low'],
            close=silver_data['Close'],
            name="Silver",
            increasing_line_color='#00C853',
            decreasing_line_color='#FF1744'
        ),
        row=1, col=1
    )
    
    # 20-day and 50-day moving averages
    ma20 = silver_data['Close'].rolling(window=20).mean()
    ma50 = silver_data['Close'].rolling(window=50).mean()
    
    fig1.add_trace(
        go.Scatter(x=ma20.index, y=ma20, mode='lines', name='20 MA',
                   line=dict(color='orange', width=1.5)),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Scatter(x=ma50.index, y=ma50, mode='lines', name='50 MA',
                   line=dict(color='blue', width=1.5)),
        row=1, col=1
    )
    
    # Support and resistance levels
    fig1.add_hline(y=support_level, line_dash="dash", line_color="green",
                   annotation_text=f"Support: ${support_level}", row=1, col=1)
    fig1.add_hline(y=resistance_level, line_dash="dash", line_color="red",
                   annotation_text=f"Resistance: ${resistance_level}", row=1, col=1)
    
    # ROC chart
    colors = ['#00C853' if val >= 0 else '#FF1744' for val in silver_roc.dropna()]
    fig1.add_trace(
        go.Bar(
            x=silver_roc.dropna().index,
            y=silver_roc.dropna(),
            name='ROC',
            marker_color=colors
        ),
        row=2, col=1
    )
    fig1.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
    
    # RSI chart
    fig1.add_trace(
        go.Scatter(
            x=silver_rsi.index,
            y=silver_rsi,
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)'
        ),
        row=3, col=1
    )
    
    # RSI overbought/oversold levels
    fig1.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig1.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig1.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    fig1.update_layout(
        height=800,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=40, t=80, b=40)
    )
    
    fig1.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig1.update_yaxes(title_text="ROC (%)", row=2, col=1)
    fig1.update_yaxes(title_text="RSI", row=3, col=1)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Momentum summary
    col1, col2, col3 = st.columns(3)
    with col1:
        if current_roc > 0:
            st.success(f"✅ **Positive Momentum**: ROC at {current_roc:.2f}%")
        else:
            st.error(f"⚠️ **Negative Momentum**: ROC at {current_roc:.2f}%")
    with col2:
        if current_rsi > 70:
            st.warning(f"⚠️ **Overbought**: RSI at {current_rsi:.1f}")
        elif current_rsi < 30:
            st.info(f"💎 **Oversold**: RSI at {current_rsi:.1f} - Potential buying opportunity")
        else:
            st.info(f"📊 **Neutral RSI**: {current_rsi:.1f}")
    with col3:
        st.info(f"📈 **Signal**: {momentum_signal}")


# ------------------------------------------------------------------------------
# TAB 2: Gold/Silver Ratio
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Gold to Silver Ratio (GSR) - Momentum Breakdown Analysis")
    
    if not gsr.empty:
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f"Gold/Silver Ratio with {gsr_ma_period}-Day MA",
                "GSR Rate of Change"
            )
        )
        
        # GSR line
        fig2.add_trace(
            go.Scatter(
                x=gsr.index,
                y=gsr,
                mode='lines',
                name='G/S Ratio',
                line=dict(color='#FFD700', width=2)
            ),
            row=1, col=1
        )
        
        # GSR Moving Average
        fig2.add_trace(
            go.Scatter(
                x=gsr_ma.index,
                y=gsr_ma,
                mode='lines',
                name=f'{gsr_ma_period}-Day MA',
                line=dict(color='#1E3A5F', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Fill between GSR and MA
        fig2.add_trace(
            go.Scatter(
                x=gsr.index,
                y=gsr,
                fill='tonexty',
                fillcolor='rgba(255, 215, 0, 0.2)',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # GSR ROC
        gsr_roc = calculate_roc(gsr, 14)
        gsr_roc_colors = ['#00C853' if val < 0 else '#FF1744' for val in gsr_roc.dropna()]
        
        fig2.add_trace(
            go.Bar(
                x=gsr_roc.dropna().index,
                y=gsr_roc.dropna(),
                name='GSR ROC',
                marker_color=gsr_roc_colors
            ),
            row=2, col=1
        )
        
        fig2.update_layout(
            height=600,
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig2.update_yaxes(title_text="Ratio", row=1, col=1)
        fig2.update_yaxes(title_text="ROC (%)", row=2, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # GSR Analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current GSR", f"{current_gsr:.2f}")
        with col2:
            st.metric(f"{gsr_ma_period}-Day MA", f"{current_gsr_ma:.2f}")
        with col3:
            gsr_diff = current_gsr - current_gsr_ma
            if gsr_diff < 0:
                st.success(f"📉 GSR {abs(gsr_diff):.2f} BELOW MA\n\n**Bullish for Silver**")
            else:
                st.warning(f"📈 GSR {gsr_diff:.2f} ABOVE MA\n\n**Bearish signal**")
        
        # Historical context
        st.markdown("### 📊 GSR Historical Context")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("All-Time High (Recent)", f"{gsr.max():.1f}")
        with col2:
            st.metric("All-Time Low (Recent)", f"{gsr.min():.1f}")
        with col3:
            st.metric("Mean", f"{gsr.mean():.1f}")
        with col4:
            percentile = stats.percentileofscore(gsr.dropna(), current_gsr)
            st.metric("Current Percentile", f"{percentile:.0f}%")
    else:
        st.warning("Gold data unavailable for GSR calculation")


# ------------------------------------------------------------------------------
# TAB 3: Support Monitor
# ------------------------------------------------------------------------------
with tab3:
    st.subheader(f"Support Floor Monitoring - ${support_level} Concrete Level")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_silver,
            title={'text': "Silver vs Support Level", 'font': {'size': 20}},
            number={'suffix': " USD", 'font': {'size': 36}},
            delta={
                'reference': support_level,
                'relative': False,
                'valueformat': '.2f',
                'prefix': '$'
            },
            gauge={
                'axis': {
                    'range': [support_level * 0.6, resistance_level * 1.2],
                    'tickwidth': 1,
                    'tickcolor': "darkblue"
                },
                'bar': {'color': "#C0C0C0", 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [support_level * 0.6, support_level * 0.9], 'color': '#FFCDD2'},
                    {'range': [support_level * 0.9, support_level], 'color': '#FFF9C4'},
                    {'range': [support_level, support_level * 1.1], 'color': '#DCEDC8'},
                    {'range': [support_level * 1.1, resistance_level], 'color': '#C8E6C9'},
                    {'range': [resistance_level, resistance_level * 1.2], 'color': '#B2DFDB'}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': support_level
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            font={'color': "#1E3A5F", 'family': "Arial"}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown("### 📍 Price Position Analysis")
        
        if current_silver >= support_level:
            pct_above = distance_to_support
            st.success(f"""
            ✅ **ABOVE SUPPORT**
            
            Silver is **${current_silver - support_level:.2f}** ({pct_above:.1f}%) above the ${support_level} concrete support floor.
            
            This suggests the bullish structure remains intact according to Oliver's framework.
            """)
        else:
            pct_below = abs(distance_to_support)
            st.error(f"""
            ⚠️ **BELOW SUPPORT**
            
            Silver is **${support_level - current_silver:.2f}** ({pct_below:.1f}%) below the ${support_level} support level.
            
            This is a warning signal that may require reassessment.
            """)
        
        st.markdown("---")
        
        # Distance metrics
        st.markdown("### 📏 Key Levels")
        
        levels_df = pd.DataFrame({
            'Level': ['Current Price', 'Support Floor', 'Key Resistance', 'Distance to Support', 'Distance to Resistance'],
            'Value': [
                f"${current_silver:.2f}",
                f"${support_level:.2f}",
                f"${resistance_level:.2f}",
                f"{distance_to_support:+.1f}%",
                f"{distance_to_resistance:.1f}%"
            ]
        })
        
        st.dataframe(levels_df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: Pullback Tracker
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("📉 Pullback Analysis & Duration Tracker")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart with peak annotation
        fig_pullback = go.Figure()
        
        fig_pullback.add_trace(
            go.Scatter(
                x=silver_data.index,
                y=silver_data['Close'],
                mode='lines',
                name='Silver Price',
                line=dict(color='#C0C0C0', width=2)
            )
        )
        
        # Add peak marker
        if pullback_info['peak_date'] is not None:
            fig_pullback.add_trace(
                go.Scatter(
                    x=[pullback_info['peak_date']],
                    y=[pullback_info['peak_price']],
                    mode='markers+text',
                    name='Recent Peak',
                    marker=dict(color='red', size=15, symbol='triangle-down'),
                    text=[f"Peak: ${pullback_info['peak_price']:.2f}"],
                    textposition='top center'
                )
            )
            
            # Add pullback zone
            if pullback_info['is_pullback']:
                fig_pullback.add_vrect(
                    x0=pullback_info['peak_date'],
                    x1=silver_data.index[-1],
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text=f"Pullback: {pullback_info['pullback_days']} days",
                    annotation_position="top left"
                )
        
        fig_pullback.add_hline(y=support_level, line_dash="dash", line_color="green",
                               annotation_text=f"Support: ${support_level}")
        
        fig_pullback.update_layout(
            height=500,
            template="plotly_white",
            title="Silver Price with Pullback Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)"
        )
        
        st.plotly_chart(fig_pullback, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Pullback Metrics")
        
        if pullback_info['is_pullback']:
            st.warning(f"""
            ⚠️ **Active Pullback Detected**
            
            **Peak Date:** {pullback_info['peak_date'].strftime('%Y-%m-%d') if pullback_info['peak_date'] else 'N/A'}
            
            **Peak Price:** ${pullback_info['peak_price']:.2f}
            
            **Current Price:** ${current_silver:.2f}
            
            **Duration:** {pullback_info['pullback_days']} days
            
            **Depth:** {pullback_info['pullback_pct']:.2f}%
            """)
            
            # Pullback severity gauge
            severity = min(100, pullback_info['pullback_pct'] * 5)
            st.progress(severity / 100)
            st.caption(f"Pullback Severity: {severity:.0f}%")
        else:
            st.success("""
            ✅ **No Significant Pullback**
            
            Price is at or near recent highs. Momentum remains strong.
            """)
        
        # Historical pullback statistics
        st.markdown("### 📈 Historical Context")
        
        # Calculate average pullbacks
        rolling_max = silver_data['Close'].rolling(window=20).max()
        drawdowns = (silver_data['Close'] - rolling_max) / rolling_max * 100
        
        st.metric("Current Drawdown", f"{drawdowns.iloc[-1]:.1f}%")
        st.metric("Max Drawdown (Period)", f"{drawdowns.min():.1f}%")
        st.metric("Average Drawdown", f"{drawdowns[drawdowns < 0].mean():.1f}%")


# ------------------------------------------------------------------------------
# TAB 5: S&P 500 Correlation
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("📊 S&P 500 Correlation & Market Health Monitor")
    
    if not sp500_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_corr = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.5, 0.5],
                subplot_titles=("S&P 500 Index", "Silver Price (Normalized)")
            )
            
            # S&P 500
            fig_corr.add_trace(
                go.Scatter(
                    x=sp500_data.index,
                    y=sp500_data['Close'],
                    mode='lines',
                    name='S&P 500',
                    line=dict(color='#1E88E5', width=2)
                ),
                row=1, col=1
            )
            
            # 52-week high
            fig_corr.add_hline(y=sp500_52w_high, line_dash="dash", line_color="green",
                              annotation_text=f"52W High: {sp500_52w_high:,.0f}", row=1, col=1)
            
            # 50% crash level
            fig_corr.add_hline(y=crash_target, line_dash="dot", line_color="red",
                              annotation_text=f"50% Crash: {crash_target:,.0f}", row=1, col=1)
            
            # Normalized Silver
            silver_norm = (silver_data['Close'] / silver_data['Close'].iloc[0]) * 100
            fig_corr.add_trace(
                go.Scatter(
                    x=silver_data.index,
                    y=silver_norm,
                    mode='lines',
                    name='Silver (Normalized)',
                    line=dict(color='#C0C0C0', width=2)
                ),
                row=2, col=1
            )
            
            fig_corr.update_layout(
                height=600,
                template="plotly_white",
                showlegend=True
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Market Health Indicators")
            
            # Correlation
            correlation = calculate_correlation(
                silver_data['Close'],
                sp500_data['Close'],
                window=30
            )
            
            st.metric(
                "30-Day Correlation",
                f"{correlation:.3f}" if not pd.isna(correlation) else "N/A",
                help="Positive = move together, Negative = inverse relationship"
            )
            
            st.metric(
                "S&P 500 52-Week High",
                f"{sp500_52w_high:,.0f}"
            )
            
            st.metric(
                "Current Distance from High",
                f"{sp500_distance_from_high:.2f}%",
                delta="Bearish" if sp500_distance_from_high < -10 else "Normal",
                delta_color="inverse" if sp500_distance_from_high < -10 else "off"
            )
            
            st.markdown("---")
            
            st.markdown("### 📉 50% Crash Monitor")
            
            st.metric("50% Crash Target", f"{crash_target:,.0f}")
            
            # Progress to crash
            if sp500_distance_from_high < 0:
                progress = (sp500_52w_high - current_sp500) / (sp500_52w_high - crash_target) * 100
                progress = min(100, max(0, progress))
            else:
                progress = 0
            
            st.progress(progress / 100)
            st.caption(f"Progress toward 50% crash: {progress:.1f}%")
            
            # Warning levels
            if sp500_distance_from_high < -20:
                st.error("🔴 SEVERE CORRECTION: >20% from highs")
            elif sp500_distance_from_high < -10:
                st.warning("🟡 CORRECTION ZONE: >10% from highs")
            elif sp500_distance_from_high < -5:
                st.info("🟠 PULLBACK: 5-10% from highs")
            else:
                st.success("🟢 NEAR HIGHS: <5% from highs")
    else:
        st.warning("S&P 500 data unavailable")


# ==============================================================================
# SUMMARY SECTION
# ==============================================================================

st.markdown("---")
st.markdown("### 📋 Dashboard Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("""
    **🥈 Silver Analysis**
    """)
    summary_data = {
        "Metric": ["Current Price", "ROC Momentum", "RSI", "vs Support"],
        "Value": [
            f"${current_silver:.2f}",
            f"{current_roc:.2f}%",
            f"{current_rsi:.1f}",
            f"{distance_to_support:+.1f}%"
        ],
        "Signal": [
            "📈" if silver_change_pct > 0 else "📉",
            "✅" if current_roc > 0 else "⚠️",
            "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "⚪",
            "✅" if current_silver > support_level else "⚠️"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

with summary_col2:
    st.markdown("""
    **⚖️ GSR & Correlation**
    """)
    if current_gsr:
        gsr_data = {
            "Metric": ["GSR", f"GSR {gsr_ma_period}MA", "Signal", "S&P Corr."],
            "Value": [
                f"{current_gsr:.2f}",
                f"{current_gsr_ma:.2f}",
                gsr_signal,
                f"{correlation:.3f}" if not pd.isna(correlation) else "N/A"
            ]
        }
        st.dataframe(pd.DataFrame(gsr_data), hide_index=True, use_container_width=True)

with summary_col3:
    st.markdown("""
    **📊 Market Context**
    """)
    if current_sp500:
        market_data = {
            "Metric": ["S&P 500", "From 52W High", "Crash Target", "Pullback Days"],
            "Value": [
                f"{current_sp500:,.0f}",
                f"{sp500_distance_from_high:.1f}%",
                f"{crash_target:,.0f}",
                str(pullback_info['pullback_days']) if pullback_info['is_pullback'] else "0"
            ]
        }
        st.dataframe(pd.DataFrame(market_data), hide_index=True, use_container_width=True)


# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("""
<div class="footer">
    📊 Data sourced from Yahoo Finance via yfinance library | Dashboard inspired by Michael Oliver's momentum analysis<br>
    ⚠️ <strong>Disclaimer:</strong> This dashboard is for educational and informational purposes only. Not financial advice.<br>
    Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()
