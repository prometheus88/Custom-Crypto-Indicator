import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time  # For rate limiting
import datetime  # For timestamped file naming

# API Key Configuration
API_KEY = "YOURAPIKEY"
BASE_URL = "https://api.coingecko.com/api/v3"

# Rate limit settings
REQUESTS_PER_MINUTE = 25
DELAY = 60 / REQUESTS_PER_MINUTE  # Delay in seconds between requests


# Fetch top 50 cryptocurrencies
def fetch_top_cryptos():
    url = f"{BASE_URL}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=1&x_cg_api_key={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


# Fetch historical price and volume data
def fetch_historical_data(coin_id, days=7):
    url = f"{BASE_URL}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&x_cg_api_key={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    prices = [entry[1] for entry in data["prices"]]
    volumes = [entry[1] for entry in data["total_volumes"]]
    return prices, volumes


# Calculate RSI
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_list = []
    for i in range(period, len(prices) - 1):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))
        rsi_list.append(rsi)

    return rsi_list


# Perform Cross-Correlation
def cross_correlation(series1, series2):
    n = len(series1)
    lags = np.arange(-n + 1, n)  # Possible lags
    corr = np.correlate(series1 - np.mean(series1), series2 - np.mean(series2), mode='full')
    return lags, corr


# Calculate Momentum Score
def calculate_momentum_score(rsi, max_corr, max_lag):
    last_rsi = rsi[-1]
    # RSI Weight
    if last_rsi > 70:
        rsi_score = -5  # Overbought
    elif last_rsi < 30:
        rsi_score = 5   # Oversold
    else:
        rsi_score = 0   # Neutral

    # Correlation Weight
    corr_score = 0
    if max_corr > 0.7:  # High correlation
        if max_lag > 0:  
            # RSI lags volume (momentum building)
            corr_score = 5
        elif max_lag < 0:  
            # Volume lags RSI (momentum weakening)
            corr_score = -5
    elif max_corr < 0.3:  
        # Weak correlation
        corr_score = 0

    # Final Score
    return rsi_score + corr_score


# Analyze Top 50 Cryptos with Rate Limiting and Console Output
def main():
    # Fetch top 50 cryptos
    top_cryptos = fetch_top_cryptos()
    results = []

    print("\n--- Momentum Analysis for Top 50 Cryptocurrencies ---\n")

    for idx, crypto in enumerate(top_cryptos):
        try:
            coin_id = crypto['id']
            name = crypto['name']
            symbol = crypto['symbol']

            # Fetch price and volume data
            prices, volumes = fetch_historical_data(coin_id, days=7)

            # Calculate RSI
            rsi = calculate_rsi(prices, period=14)
            volumes = volumes[-len(rsi):]  # Match volume length with RSI

            # Cross-Correlation
            lags, corr = cross_correlation(rsi, volumes)
            max_lag = lags[np.argmax(corr)]
            max_corr = np.max(corr)

            # Momentum Score
            momentum_score = calculate_momentum_score(rsi, max_corr, max_lag)

            # Rating
            if momentum_score == 10:
                rating = "Strong Buy"
            elif 5 <= momentum_score < 10:
                rating = "Buy"
            elif -5 <= momentum_score < 5:
                rating = "Neutral"
            elif -10 < momentum_score < -5:
                rating = "Sell"
            elif momentum_score == -10:
                rating = "Strong Sell"

            # Store Results
            results.append({
                "Name": name,
                "Symbol": symbol.upper(),
                "RSI": round(rsi[-1], 2),
                "Max Corr": round(max_corr, 4),
                "Lag (Hrs)": max_lag,
                "Momentum Score": momentum_score,
                "Rating": rating
            })

            # Console Output for Each Crypto
            print(
                f"{idx + 1:>2}. {name} ({symbol.upper()}) - RSI: {rsi[-1]:.2f}, Corr: {max_corr:.4f}, Lag: {max_lag}, Score: {momentum_score}, Rating: {rating}"
            )

            # Rate limiting (Pause after each request)
            time.sleep(DELAY)

        except Exception as e:
            print(f"Error analyzing {crypto['name']} ({crypto['symbol']}): {e}")
            time.sleep(DELAY)  # Add delay even if error occurs to stay within limits

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crypto_custom_indicator_{timestamp}.json"

    # Save results to JSON
    df.to_json(filename, index=False)

    # Final Output
    print(f"\nAnalysis complete. Results saved to '{filename}'.\n")


if __name__ == "__main__":
    main()
