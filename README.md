# 📈 Smart Stock Analysis Dashboard

A full-featured, multi-functional stock analysis dashboard built with **Streamlit**.  
This app empowers users to visualize stock data, analyze market trends, simulate paper trading, and gauge news sentiment using **FinBERT** — all through an intuitive web interface.

> 🔍 Supports both **US** and **Indian (NSE)** stocks. Candlestick charts, technical indicators, real-time news sentiment, and a portfolio simulator — all in one place.

---
![app view](/app.view.png)
---

## 🌟 Features

### 📊 Market Analysis
- Interactive **Candlestick Charts** with OHLC data
- Technical Indicators:
  - Simple Moving Average (**SMA**)
  - Exponential Moving Average (**EMA**)
  - Moving Average Convergence Divergence (**MACD**)
- Customizable timeframes: **Daily**, **15-Min**, **1-Hour**, **Weekly**
- Simulated market **sentiment overlay**

### 📰 News Sentiment Analysis
- Real-time news headlines using **NewsAPI**
- Sentiment classification using **FinBERT** (fine-tuned BERT model)
- Sentiment bar chart and pie chart visualizations

### 💰 Paper Trading Simulator
- Virtual portfolio starting with $10,000
- Buy/Sell simulation with commission handling
- Real-time portfolio valuation
- Visualized trade performance over time

### 🛠️ Tech Stack
- [Streamlit](https://streamlit.io)
- [yFinance](https://pypi.org/project/yfinance/) – stock data source
- [Plotly](https://plotly.com/python/) – interactive charts
- [FinBERT](https://huggingface.co/ProsusAI/finbert) – sentiment analysis
- [NewsAPI](https://newsapi.org) – real-time news headlines
- [Matplotlib, NumPy, Pandas, Scikit-learn, Transformers] – data handling and ML

---

## 📸 Screenshots

### 🕯️ Candlestick Chart with Indicators  
![Candlestick Chart](/candlestick.png)

### 📰 News Sentiment Dashboard  
![Sentiment Analysis](/new_sentiment.png)
![Sentiment breakdown](/sentiment_breakdown.png)
### 💼 Paper Trading Simulator  
![Paper Trading](/papertrading.png)

> ℹ️ _If you haven't added screenshots yet, upload them to an `assets/` folder and update the paths above._

---

## 📝 Summary

**Smart Stock Analysis Dashboard** is an advanced, interactive web application built with **Streamlit**, designed for stock market analysis, visualization, and forecasting. It enables users to explore historical stock performance with candlestick charts, analyze market trends using technical indicators (SMA, EMA, MACD), assess sentiment from real-time news headlines using **FinBERT**, and simulate virtual trades using a built-in paper trading simulator. Additionally, it includes a **machine learning–powered feature to predict the next trading day’s closing price** using Random Forest Regressor.

The app supports both **US** and **Indian (NSE)** stocks via Yahoo Finance, making it a powerful tool for learning, analysis, and prototyping trading strategies.

---
