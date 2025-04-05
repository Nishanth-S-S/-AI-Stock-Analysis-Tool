import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
from PIL import Image
import requests
import google.generativeai as genai
import streamlit as st
import os

# --- CONFIGURATION ---
NEWS_API_KEY = "c7a2cdbd13d440839be331c12ddaef91"
GEMINI_API_KEY = "AIzaSyD7a8CuvWOMNd7fvcUX_y37LQltiyI5ImQ"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Streamlit Setup ---
st.set_page_config(page_title="📈 Stock Analyzer", layout="wide")
st.title(":bar_chart: AI Stock Analysis Tool")

# --- Functions ---
def analyze_stock_indicators(ticker):
    today = date.today()
    past_date = today - relativedelta(years=6)
    data = yf.download(ticker, start=past_date, end=today)

    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()

    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Save charts
    charts = []

    plt.figure(figsize=(10,5))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['EMA_50'], label='EMA 50')
    plt.plot(data['EMA_100'], label='EMA 100')
    plt.title('EMA')
    plt.legend()
    plt.savefig("ema.png")
    charts.append("ema.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title('RSI')
    plt.legend()
    plt.savefig("rsi.png")
    charts.append("rsi.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['Signal Line'], label='Signal Line', color='red')
    plt.title('MACD')
    plt.legend()
    plt.savefig("macd.png")
    charts.append("macd.png")
    plt.close()

    prompt = f"""
These charts show {ticker}'s closing price, EMA (50 and 100), RSI, and MACD.
Please analyze the stock based on these technical indicators and suggest if the trend looks bullish, bearish, or neutral.
Conclude with "Recommendation from indicators: BUY / SELL / HOLD"
"""
    imgs = [Image.open(path) for path in charts]
    response = model.generate_content([prompt] + imgs)
    return response.text, charts

def get_news_data(ticker):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 5
    }
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

def analyze_news_sentiment(ticker, articles):
    headlines = [article['title'] for article in articles]
    prompt = f"""
You are a financial AI assistant. Based on the following news headlines for {ticker}, summarize the sentiment and recommend:
BUY, SELL, or HOLD.
Explain your reasoning and conclude with "Recommendation from news: BUY / SELL / HOLD"

Headlines:
{chr(10).join("- " + h for h in headlines)}
"""
    response = model.generate_content(prompt)
    return response.text

def final_recommendation(indicator_text, news_text):
    lines = indicator_text.splitlines() + news_text.splitlines()
    votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for line in lines:
        for rec in votes:
            if rec in line.upper():
                votes[rec] += 1
    if votes["BUY"] > votes["SELL"] and votes["BUY"] > votes["HOLD"]:
        return "🚀 FINAL RECOMMENDATION: BUY"
    elif votes["SELL"] > votes["BUY"] and votes["SELL"] > votes["HOLD"]:
        return "📉 FINAL RECOMMENDATION: SELL"
    else:
        return "📈 FINAL RECOMMENDATION: HOLD"

# --- User Input ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

if st.button("Run Analysis"):
    col1, col2 = st.columns(2)

    with st.spinner("Analyzing technical indicators..."):
        indicator_text, chart_paths = analyze_stock_indicators(ticker)

    with col1:
        st.subheader(":bar_chart: Technical Indicators")
        for path in chart_paths:
            st.image(path)
        st.markdown(indicator_text)

    with st.spinner("Fetching and analyzing news..."):
        articles = get_news_data(ticker)
        if articles:
            news_text = analyze_news_sentiment(ticker, articles)
        else:
            news_text = "No recent news found. Recommendation from news: HOLD"

    with col2:
        st.subheader(":newspaper: News Analysis")
        for i, article in enumerate(articles, 1):
            st.markdown(f"{i}. [{article['title']}]({article['url']}) _(Source: {article['source']['name']})_", unsafe_allow_html=True)
        st.markdown(news_text)

    st.divider()
    st.subheader(":crystal_ball: Final AI Recommendation")
    st.success(final_recommendation(indicator_text, news_text))
