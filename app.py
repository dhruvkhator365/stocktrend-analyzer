import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from transformers.pipelines import pipeline
import torch
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_sentiment_analyzer():
    return pipeline("text-classification", model="./finbert_model/ProsusAI/finbert")

# Load only when needed (in the sentiment section)
sentiment_analyzer = None


# Paper trading simulator
class PaperTradingSimulator:
    def __init__(self):
        self.portfolio = {}
        self.balance = 10000
        self.history = []
        self.commission = 0.0005
        
    def execute_trade(self, ticker, shares, price, action):
        cost = shares * price * (1 + self.commission)
        if action == "BUY" and cost <= self.balance:
            self.balance -= cost
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + shares
            self.history.append({
                "date": datetime.now(),
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "action": action,
                "value": cost
            })
            return True
        elif action == "SELL" and self.portfolio.get(ticker, 0) >= shares:
            credit = shares * price * (1 - self.commission)
            self.balance += credit
            self.portfolio[ticker] -= shares
            self.history.append({
                "date": datetime.now(),
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "action": action,
                "value": credit
            })
            return True
        return False

# Page config
st.set_page_config(
    page_title="📈 Smart Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    .stButton>button {
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }
    .error-message {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🔍 Stock Analysis Dashboard")
    
    analysis_type = st.radio(
        "Navigation",
        ["📊 Market Analysis", "📈 Paper Trading", "📰 Sentiment Analysis"],
        label_visibility="collapsed"
    )
    
    st.subheader("Data Configuration")
    ticker = st.text_input(
        "Stock Ticker", 
        value="AAPL",
        help="Enter the stock symbol (e.g., AAPL for Apple, TSLA for Tesla) and Include suffix like .NS for NSE (e.g., INFY.NS, TCS.NS)"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2023-01-01"),
            max_value=datetime.today()-timedelta(days=7),
            help="Select the starting date of historical data"
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.today(),
            max_value=datetime.today(),
            help="Select the ending date of historical data"
        )
    
    timeframe = st.selectbox(
        "Analysis Timeframe",
        options=["Daily", "15-Min", "1-Hour", "Weekly"],
        index=0,
        help="Choose how frequently data should be analyzed"
    )
    
    st.caption("ℹ️ This dashboard is for educational purposes only. Not financial advice.")

# Fetch data
@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end + timedelta(days=1))

        if data is None or data.empty:
            st.warning("⚠️ No data returned from yfinance.")
            return None

        # 🧹 Flatten MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

        # ✅ Rename columns back to standard OHLC names
        renamed_cols = {}
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            col_with_ticker = f"{col}_{ticker}"
            if col_with_ticker in data.columns:
                renamed_cols[col_with_ticker] = col
        data.rename(columns=renamed_cols, inplace=True)

        # 🧪 Check if essential OHLC columns are present
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            st.error(f"❌ Missing required columns in fetched data: {missing}")
            st.dataframe(data.head())
            return None

        # ✅ Use only necessary columns
        data = data[required_cols].copy()

        # 🔢 Add indicators
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['EMA_10'] = data['Close'].ewm(span=10).mean()
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        data['Signal'] = data['MACD'].ewm(span=9).mean()

        return data.dropna()

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None



data = fetch_data(ticker, start_date, end_date)
if data is None or data.empty:
    st.error("No data available for the selected ticker and date range.")
    st.stop()

# Resample
def resample_data(data, timeframe):
    timeframe_mapping = {
        "Daily": "D",
        "15-Min": "15T",
        "1-Hour": "1H",
        "Weekly": "W"
    }

    if timeframe == "Daily":
        return data

    try:
        # Check required columns before resampling
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            st.error(f"Missing columns before resampling: {missing}")
            return pd.DataFrame()

        # Resample
        resampled_df = data.resample(timeframe_mapping[timeframe]).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Add indicators
        if not resampled_df.empty:
            resampled_df['SMA_10'] = resampled_df['Close'].rolling(10).mean()
            resampled_df['EMA_10'] = resampled_df['Close'].ewm(span=10).mean()
            resampled_df['MACD'] = resampled_df['Close'].ewm(span=12).mean() - resampled_df['Close'].ewm(span=26).mean()
            resampled_df['Signal'] = resampled_df['MACD'].ewm(span=9).mean()

        return resampled_df.dropna()
    except Exception as e:
        st.error(f"Error during resampling: {str(e)}")
        return pd.DataFrame()


resampled_data = resample_data(data, timeframe)

# Sentiment
def generate_sentiment_scores(dates):
    return pd.Series(
        np.random.uniform(-1, 1, len(dates)),
        index=dates,
        name='Sentiment'
    )

sentiment_scores = generate_sentiment_scores(resampled_data.index)

# Charts & Visualizations
def plot_candlestick(df, title):
    required_cols = ['Open', 'High', 'Low', 'Close']

    # 🔍 Check for missing columns BEFORE calling dropna
    if not set(required_cols).issubset(df.columns):
        available_cols = df.columns.tolist()
        st.error(f"❌ Candlestick chart requires columns {required_cols}, but found only: {available_cols}")
        st.dataframe(df.head())
        return

    # ✅ Now it's safe to drop rows with NaNs
    df_clean = df.dropna(subset=required_cols)
    if df_clean.empty:
        st.warning("⚠️ All OHLC rows are NaN after cleaning. Cannot plot candlestick.")
        st.dataframe(df.head())
        return

    df_clean = df_clean.copy()
    df_clean.index = pd.to_datetime(df_clean.index)

    fig = go.Figure(data=[go.Candlestick(
        x=df_clean.index,
        open=df_clean['Open'],
        high=df_clean['High'],
        low=df_clean['Low'],
        close=df_clean['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)





def plot_sma_ema(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax.plot(df.index, df['SMA_10'], label='SMA 10', color='orange')
    ax.plot(df.index, df['EMA_10'], label='EMA 10', color='purple')
    ax.set_title('SMA vs EMA Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_macd(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['Signal'], label='Signal', color='red')
    ax.set_title('MACD & Signal Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_sentiment_trend(sentiment_series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sentiment_series.index, sentiment_series, color='purple', marker='o')
    ax.set_title("Simulated Sentiment Over Time")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_portfolio_pie(portfolio, price_lookup):
    labels = []
    values = []
    for ticker_, shares in portfolio.items():
        labels.append(ticker_)
        values.append(shares * price_lookup.get(ticker_, 0))
    if values:
        fig = px.pie(names=labels, values=values, title="📊 Portfolio Allocation", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

def plot_trade_pnl(history_df):
    if not history_df.empty:
        history_df['cumulative'] = history_df['value'].cumsum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history_df['date'], history_df['cumulative'], label='Cumulative Value', color='green')
        ax.set_title('📈 Trade Performance Over Time')
        ax.set_ylabel('Total Value')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
# Predict tomorrow's price using Random Forest
def predict_tomorrow_price(df):
    try:
        df = df.dropna(subset=['Close', 'SMA_10', 'EMA_10', 'MACD', 'Signal', 'Volume'])
        
        # Create target: next day's closing price
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        features = df[['Close', 'SMA_10', 'EMA_10', 'MACD', 'Signal', 'Volume']]
        target = df['Target']

        # Train/Test Split (last row will be used for prediction)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features, target)

        latest_features = features.iloc[-1].values.reshape(1, -1)
        predicted = model.predict(latest_features)[0]

        return predicted
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Market Analysis
if analysis_type == "📊 Market Analysis":
    st.header(f"{ticker} Market Analysis - {timeframe} View")
    st.markdown("""
    Understand the recent performance of the selected stock using trend indicators 
    (SMA, EMA, MACD), candlestick charts, and a simulated market sentiment overlay.
    """)
    
    with st.expander("📘 How to use this section"):
        st.markdown("""
        - Blue line shows closing price trend.
        - Candlestick chart shows price action.
        - SMA, EMA show trend direction.
        - MACD helps identify trend shifts.
        - Simulated sentiment shows emotional tone.
        """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        last_close = float(resampled_data['Close'].iloc[-1])
        st.metric("Current Price", f"${last_close:.2f}")
    with col2:
        volume = int(resampled_data['Volume'].iloc[-1])
        st.metric("Volume (Latest)", f"{volume:,}")
    with col3:
        avg_sentiment = float(sentiment_scores.mean())
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

    predicted_price = predict_tomorrow_price(resampled_data)
    if predicted_price:
        st.metric("🔮 Predicted Close (Tomorrow)", f"${predicted_price:.2f}")
        
    st.subheader("📉 Candlestick Chart")

    plot_candlestick(resampled_data, f"{ticker} Candlestick Chart")
    
    st.subheader("📊 Trend Indicators")
    plot_sma_ema(resampled_data)
    plot_macd(resampled_data)
    
    st.subheader("📈 Simulated Sentiment")
    plot_sentiment_trend(sentiment_scores)

# Paper Trading
elif analysis_type == "📈 Paper Trading":
    st.header("💰 Paper Trading Simulator")
    st.markdown("""
    Practice buying and selling stocks without real money. You start with $10,000. All trades include 0.05% commission.
    """)
    
    with st.expander("📘 How to use this section"):
        st.markdown("""
        - Use the form to place a BUY or SELL order.
        - Available cash and portfolio value will update after each trade.
        - Trade history and visualizations shown below.
        """)
    
    if 'simulator' not in st.session_state:
        st.session_state.simulator = PaperTradingSimulator()
    
    simulator = st.session_state.simulator
    current_price = float(resampled_data['Close'].iloc[-1]) if not resampled_data.empty else 0
    
    with st.form("trade_form"):
        shares = st.number_input("Number of Shares", 1, 1000, 10)
        action = st.selectbox("Select Action", ["BUY", "SELL"])
        submitted = st.form_submit_button(f"{action} {ticker}")
        
        if submitted:
            result = simulator.execute_trade(ticker, shares, current_price, action)
            if result:
                st.success(f"✅ {action} order for {shares} shares of {ticker} executed at ${current_price:.2f}")
            else:
                st.warning("⚠️ Trade could not be executed — you may not have enough funds or shares.")
    
    st.subheader("📌 Portfolio Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Cash", f"${simulator.balance:,.2f}")
    
    total_value = simulator.balance
    for ticker_, shares in simulator.portfolio.items():
        total_value += shares * current_price
    with col2:
        st.metric("Portfolio Value", f"${total_value:,.2f}")
    
    if simulator.portfolio:
        st.write("📦 Current Holdings")
        for stock, shares in simulator.portfolio.items():
            st.write(f"{stock}: {shares} shares")
        plot_portfolio_pie(simulator.portfolio, {ticker: current_price})
    
    if simulator.history:
        st.subheader("📅 Trade History")
        history_df = pd.DataFrame(simulator.history)
        st.dataframe(history_df)
        plot_trade_pnl(history_df)

# Sentiment Placeholder
elif analysis_type == "📰 Sentiment Analysis":
    st.header("📰 News Sentiment Analysis")
    st.markdown("Analyzing recent news headlines using FinBERT sentiment classification.")

    if sentiment_analyzer is None:
        sentiment_analyzer = load_sentiment_analyzer()
    
    query = ticker
    api_key = "6f938738cdb94727b7bfbe60e71ab2e0"  # Your NewsAPI key
    news_url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"

    try:
        import requests
        response = requests.get(news_url)
        articles = response.json().get("articles", [])

        if not articles:
            st.warning("No recent news articles found for this ticker.")
        else:
            headlines = [article["title"] for article in articles if article.get("title")]
            publish_dates = [article["publishedAt"] for article in articles if article.get("title")]

            sentiments = sentiment_analyzer(headlines)

            df_sentiment = pd.DataFrame({
                "Date": publish_dates,
                "Headline": headlines,
                "Sentiment": [s["label"] for s in sentiments]
            })
            df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
            df_sentiment.sort_values("Date", ascending=False, inplace=True)

            st.subheader("🗞️ Recent News & Sentiment")
            st.dataframe(df_sentiment[["Date", "Headline", "Sentiment"]])

            st.subheader("📊 Sentiment Breakdown")
            sentiment_counts = df_sentiment["Sentiment"].value_counts()

            

            col1, col2 = st.columns([1, 1])  # Equal columns

            with col1:
                st.markdown("**Sentiment Breakdown (Pie Chart)**")
                fig1, ax1 = plt.subplots(figsize=(3, 3))
                ax1.pie(sentiment_counts, labels=list(map(str, sentiment_counts.index)), autopct="%1.1f%%", startangle=140)
                ax1.axis("equal")
                st.pyplot(fig1, use_container_width=False)

            with col2:
                st.markdown("**Sentiment Distribution (Bar Chart)**")
                fig2, ax2 = plt.subplots(figsize=(3.6, 3))
                sentiment_counts.plot(
                    kind="bar",
                    color=["green" if s == "positive" else "red" if s == "negative" else "gray" for s in sentiment_counts.index],
                    ax=ax2
                )
                ax2.set_ylabel("Count")
                ax2.set_title("")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2, use_container_width=False)


    except Exception as e:
        st.error(f"Failed to fetch or analyze news: {str(e)}")


# Glossary
with st.expander("📘 Glossary"):
    st.markdown("""
    - **SMA**: Simple Moving Average – average of closing prices over a period.
    - **EMA**: Exponential Moving Average – gives more weight to recent prices.
    - **MACD**: Difference between two EMAs; used to identify momentum.
    - **Sentiment Score**: Simulated score (-1 to +1) indicating mood of the market.
    - **Portfolio Value**: Total value of cash and all shares held.
    """)