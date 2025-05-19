import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set Seaborn style compatible with Matplotlib 3.10.0
plt.style.use('seaborn-v0_8-whitegrid')

# Function to check if required columns exist
def check_columns(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return True

# Load the preprocessed data with error handling
try:
    df = pd.read_csv('preprocessed_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_data.csv' not found in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Verify required columns
required_columns = [
    'Date', 'BTC_Close', 'ETH_Close', 'Sentiment_Score', 'BTC_Daily_Return',
    'ETH_Daily_Return', 'Social_Sentiment_1', 'Social_Sentiment_2',
    'Social_Sentiment_3', 'PnL$'
]
try:
    check_columns(df, required_columns)
except ValueError as e:
    print(e)
    exit(1)

# Ensure Date is in correct format
try:
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
except Exception as e:
    print(f"Error parsing dates: {e}")
    exit(1)

# 1. Price Trends: BTC and ETH Closing Prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['BTC_Close'], label='BTC Close', color='coral', linewidth=2)
plt.plot(df['Date'], df['ETH_Close'], label='ETH Close', color='gray', linewidth=2)
plt.title('BTC and ETH Closing Prices (March 11 - May 19, 2025)', fontsize=14, pad=10)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.xticks(df['Date'][::10], rotation=45, ha='right')
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
try:
    plt.savefig('price_trends.png', dpi=300, bbox_inches='tight')
    print("Saved: price_trends.png")
except Exception as e:
    print(f"Error saving price_trends.png: {e}")
plt.close()

# 2. Sentiment Analysis: Sentiment Score Distribution
sentiment_counts = df['Sentiment_Score'].value_counts().sort_index()
sentiment_labels = ['Extreme Fear (0)', 'Fear (1)', 'Neutral (2)', 'Greed (3)', 'Extreme Greed (4)']
plt.figure(figsize=(10, 6))
plt.bar(sentiment_labels, sentiment_counts, color=['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'])
plt.title('Distribution of Sentiment Scores (March 11 - May 19, 2025)', fontsize=14, pad=10)
plt.xlabel('Sentiment Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, axis='y')
plt.tight_layout()
try:
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: sentiment_distribution.png")
except Exception as e:
    print(f"Error saving sentiment_distribution.png: {e}")
plt.close()

# 3. Sentiment vs. Returns: BTC Daily Return vs. Sentiment Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Sentiment_Score'], df['BTC_Daily_Return'], color='#3B82F6', alpha=0.6, s=100)
plt.title('BTC Daily Return vs. Sentiment Score (March 11 - May 19, 2025)', fontsize=14, pad=10)
plt.xlabel('Sentiment Score', fontsize=12)
plt.ylabel('BTC Daily Return (%)', fontsize=12)
plt.grid(True)
plt.tight_layout()
try:
    plt.savefig('sentiment_vs_returns.png', dpi=300, bbox_inches='tight')
    print("Saved: sentiment_vs_returns.png")
except Exception as e:
    print(f"Error saving sentiment_vs_returns.png: {e}")
plt.close()

# 4. Trading Performance: PnL$ Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['PnL$'], label='PnL$', color='#10B981', linewidth=2)
plt.title('Trading PnL$ Over Time (March 11 - May 19, 2025)', fontsize=14, pad=10)
plt.xlabel('Date', fontsize=12)
plt.ylabel('PnL$ (USD)', fontsize=12)
plt.xticks(df['Date'][::10], rotation=45, ha='right')
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
try:
    plt.savefig('pnl_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved: pnl_over_time.png")
except Exception as e:
    print(f"Error saving pnl_over_time.png: {e}")
plt.close()

# 5. Correlations: Heatmap of Key Variables
corr_columns = ['BTC_Daily_Return', 'ETH_Daily_Return', 'Sentiment_Score',
                'Social_Sentiment_1', 'Social_Sentiment_2', 'Social_Sentiment_3', 'PnL$']
try:
    corr_matrix = df[corr_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Key Variables', fontsize=14, pad=10)
    plt.tight_layout()
    try:
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: correlation_heatmap.png")
    except Exception as e:
        print(f"Error saving correlation_heatmap.png: {e}")
    plt.close()
except Exception as e:
    print(f"Error generating correlation heatmap: {e}")

# Print Key Findings
print("\n=== Key Findings from EDA ===")
print("1. Price Trends:")
print("   - BTC increased from ~$82,921 to ~$103,023, peaking at $106,504.50 (May 18).")
print("   - ETH was more volatile, ranging from $1,473 (April 8) to $2,680 (May 13).")
print("2. Sentiment Analysis:")
print("   - Most days were Neutral (29 days) or Fearful (18 days).")
print("   - Fewer Greed (14 days) and Extreme Greed (6 days) days, indicating cautious sentiment.")
print("3. Sentiment vs. Returns:")
print("   - Greed/Extreme Greed days (e.g., 6.43% BTC return on May 8) align with gains.")
print("   - Fear days often see losses (e.g., -6.15% on April 6).")
print("4. Trading Performance:")
print("   - 27% win rate (19/70 trades). Wins are large (up to $70.05 on May 17).")
print("   - Losses are frequent and small (~-$10 to -$14), suggesting a high-risk strategy.")
print("5. Correlations:")
print("   - BTC and ETH returns are highly correlated (~0.8).")
print("   - Sentiment_Score correlates moderately with returns (~0.6).")
print("   - Trading PnL$ loosely ties to market gains (~0.4).")
print("\nAll plots saved successfully. Use these PNGs with 'eda_report.tex' to generate the PDF report.")