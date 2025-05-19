import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import time

# Set Seaborn style for consistency
plt.style.use('seaborn-v0_8-whitegrid')

# Load historical data for sentiment simulation
try:
    df = pd.read_csv('preprocessed_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_data.csv' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Verify required columns
required_columns = ['Date', 'Sentiment_Score', 'PnL$']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
    exit(1)

# Ensure Date is in datetime format
try:
    df['Date'] = pd.to_datetime(df['Date'])
except Exception as e:
    print(f"Error parsing dates: {e}")
    exit(1)

# Mock API: Simulate real-time sentiment data
def get_mock_sentiment_and_pnl(df, current_date):
    # Use historical data to simulate sentiment and PnL$
    idx = np.random.randint(0, len(df))
    sentiment = df.iloc[idx]['Sentiment_Score']
    pnl = df.iloc[idx]['PnL$'] if sentiment >= 3 else 0  # Only trade if sentiment >= 3
    return sentiment, pnl

# Real-time trading simulation
start_date = datetime(2025, 5, 19, 17, 36)  # Current time: May 19, 2025, 17:36
simulation_days = 7  # Simulate 7 days
trade_log = []
current_date = start_date

print("\n=== Real-Time Trading Simulation ===")
print(f"Starting simulation at {current_date}")

for _ in range(simulation_days):
    sentiment, pnl = get_mock_sentiment_and_pnl(df, current_date)
    if sentiment >= 3:
        trade_log.append({
            'Date': current_date,
            'Sentiment_Score': sentiment,
            'PnL$': pnl
        })
        print(f"{current_date}: Sentiment = {sentiment}, Trade Executed, PnL$ = ${pnl:.2f}")
    else:
        print(f"{current_date}: Sentiment = {sentiment}, No Trade (Sentiment < 3)")
    current_date += timedelta(days=1)
    time.sleep(0.1)  # Simulate delay

# Convert trade log to DataFrame
trade_df = pd.DataFrame(trade_log)

# Calculate performance metrics
if not trade_df.empty:
    total_trades = len(trade_df)
    wins = (trade_df['PnL$'] > 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0
    total_pnl = trade_df['PnL$'].sum()
    avg_pnl = trade_df['PnL$'].mean()
    sharpe_ratio = (trade_df['PnL$'].mean() / trade_df['PnL$'].std()) * np.sqrt(252) if trade_df['PnL$'].std() != 0 else 0

    print("\n=== Simulation Performance ===")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%} ({wins}/{total_trades} trades)")
    print(f"Total PnL$: ${total_pnl:.2f}")
    print(f"Average PnL$: ${avg_pnl:.2f}")
    print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
else:
    print("\nNo trades executed during simulation.")

# Plot 1: Real-Time PnL$ Over Time
plt.figure(figsize=(12, 6))
if not trade_df.empty:
    plt.plot(trade_df['Date'], trade_df['PnL$'].cumsum(), marker='o', color='#10B981', linewidth=2, markersize=8)
    plt.title('Real-Time Trading PnL$ (Simulation)', fontsize=14, pad=10)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative PnL$ (USD)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
else:
    plt.text(0.5, 0.5, 'No Trades Executed', fontsize=12, ha='center')
try:
    plt.savefig('realtime_pnl.png', dpi=300, bbox_inches='tight')
    print("Saved: realtime_pnl.png")
except Exception as e:
    print(f"Error saving realtime_pnl.png: {e}")
plt.close()

# Plot 2: Trade Frequency by Sentiment
plt.figure(figsize=(10, 6))
if not trade_df.empty:
    sns.histplot(trade_df['Sentiment_Score'], bins=[2.5, 3.5, 4.5], color='#3B82F6', alpha=0.6)
    plt.title('Trade Frequency by Sentiment Score', fontsize=14, pad=10)
    plt.xlabel('Sentiment Score', fontsize=12)
    plt.ylabel('Number of Trades', fontsize=12)
    plt.xticks([3, 4], ['Greed (3)', 'Extreme Greed (4)'])
    plt.grid(True, axis='y')
    plt.tight_layout()
else:
    plt.text(0.5, 0.5, 'No Trades Executed', fontsize=12, ha='center')
try:
    plt.savefig('trade_frequency_sentiment.png', dpi=300, bbox_inches='tight')
    print("Saved: trade_frequency_sentiment.png")
except Exception as e:
    print(f"Error saving trade_frequency_sentiment.png: {e}")
plt.close()

# Key Insights
print("\n=== Key Insights ===")
if not trade_df.empty:
    print(f"1. Trading Performance: Executed {total_trades} trades with a {win_rate:.2%} win rate.")
    print(f"2. Profitability: Achieved ${total_pnl:.2f} total PnL$ with ${avg_pnl:.2f} average per trade.")
    print(f"3. Risk-Return: Sharpe Ratio of {sharpe_ratio:.2f} indicates risk-adjusted performance.")
    print("4. Application: Real-time sentiment-based trading can leverage market optimism.")
else:
    print("1. No Trades: Sentiment scores were below 3 during the simulation.")
print("\nPlots saved successfully. Include 'realtime_pnl.png' and 'trade_frequency_sentiment.png' in the updated PDF report.")