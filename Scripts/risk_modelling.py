import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn style for consistency
plt.style.use('seaborn-v0_8-whitegrid')

# Load data with error handling
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

# Ensure Date is in correct format
try:
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
except Exception as e:
    print(f"Error parsing dates: {e}")
    exit(1)

# Optimized strategy: Trades with Sentiment_Score >= 3
optimized_trades = df[df['Sentiment_Score'] >= 3]['PnL$'].copy()
if optimized_trades.empty:
    print("No trades meet the sentiment criteria (Sentiment_Score >= 3).")
    exit(1)

# Calculate risk metrics
# VaR and CVaR (95% confidence)
confidence_level = 0.95
var = np.percentile(optimized_trades, (1 - confidence_level) * 100)
cvar = optimized_trades[optimized_trades <= var].mean()

# Sharpe Ratio (annualized)
sharpe_ratio = (optimized_trades.mean() / optimized_trades.std()) * np.sqrt(252) if optimized_trades.std() != 0 else 0

# Maximum Drawdown
cumulative_pnl = optimized_trades.cumsum()
peak = cumulative_pnl.cummax()
drawdown = peak - cumulative_pnl
max_drawdown = drawdown.max()
max_drawdown_pct = (max_drawdown / peak[drawdown.idxmax()]) * 100 if peak[drawdown.idxmax()] != 0 else 0

# Print risk metrics
print("\n=== Risk Metrics for Optimized Strategy ===")
print(f"VaR (95% Confidence): ${var:.2f}")
print(f"CVaR (95% Confidence): ${cvar:.2f}")
print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")

# Plot 1: VaR/CVaR Distribution
plt.figure(figsize=(10, 6))
sns.histplot(optimized_trades, bins=20, color='#3B82F6', alpha=0.6)
plt.axvline(var, color='red', linestyle='--', label=f'VaR (95%): ${var:.2f}')
plt.axvline(cvar, color='purple', linestyle='--', label=f'CVaR (95%): ${cvar:.2f}')
plt.title('PnL$ Distribution with VaR and CVaR', fontsize=14, pad=10)
plt.xlabel('PnL$ (USD)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, axis='y')
plt.tight_layout()
try:
    plt.savefig('var_cvar_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: var_cvar_distribution.png")
except Exception as e:
    print(f"Error saving var_cvar_distribution.png: {e}")
plt.close()

# Plot 2: Drawdown Over Time
plt.figure(figsize=(12, 6))
plt.plot(df[df['Sentiment_Score'] >= 3]['Date'], drawdown, color='#EF4444', linewidth=2)
plt.title('Drawdown Over Time (Optimized Strategy)', fontsize=14, pad=10)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown (USD)', fontsize=12)
plt.xticks(df['Date'][::10], rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
try:
    plt.savefig('drawdown_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved: drawdown_over_time.png")
except Exception as e:
    print(f"Error saving drawdown_over_time.png: {e}")
plt.close()

# Key Insights
print("\n=== Key Insights ===")
print(f"1. VaR (95%): A daily loss is unlikely to exceed ${-var:.2f} with 95% confidence.")
print(f"2. CVaR (95%): In the worst 5% of cases, the average loss is ${-cvar:.2f}.")
print(f"3. Sharpe Ratio: {sharpe_ratio:.2f} indicates risk-adjusted performance.")
print(f"4. Max Drawdown: The largest peak-to-trough loss was ${max_drawdown:.2f} ({max_drawdown_pct:.2f}%).")
print("5. Application: Use VaR/CVaR to set risk limits and monitor drawdowns for capital protection.")
print("\nPlots saved successfully. Include 'var_cvar_distribution.png' and 'drawdown_over_time.png' in the updated PDF report.")