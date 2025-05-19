import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, levene
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
required_columns = ['Sentiment_Score', 'PnL$']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
    exit(1)

# Split data into high-sentiment (Sentiment_Score >= 3) and low-sentiment (Sentiment_Score < 3)
high_sentiment = df[df['Sentiment_Score'] >= 3]['PnL$']
low_sentiment = df[df['Sentiment_Score'] < 3]['PnL$']

# Check for sufficient data
if len(high_sentiment) < 2 or len(low_sentiment) < 2:
    print("Insufficient data for hypothesis testing. Need at least 2 samples per group.")
    exit(1)

# Descriptive statistics
high_mean = high_sentiment.mean()
low_mean = low_sentiment.mean()
high_std = high_sentiment.std()
low_std = low_sentiment.std()

# Levene's test for equal variances
levene_stat, levene_p = levene(high_sentiment, low_sentiment)

# One-tailed t-test (assuming unequal variances if Levene's p < 0.05)
equal_var = levene_p > 0.05
t_stat, p_value = ttest_ind(high_sentiment, low_sentiment, equal_var=equal_var, alternative='greater')
p_value_one_tailed = p_value  # Already one-tailed due to 'greater'

# Print results
print("\n=== Hypothesis Test Results ===")
print("H₀: Mean PnL$ (Sentiment_Score >= 3) <= Mean PnL$ (Sentiment_Score < 3)")
print("H₁: Mean PnL$ (Sentiment_Score >= 3) > Mean PnL$ (Sentiment_Score < 3)")
print("\nDescriptive Statistics:")
print(f"High-Sentiment (n={len(high_sentiment)}): Mean PnL$ = ${high_mean:.2f}, Std = ${high_std:.2f}")
print(f"Low-Sentiment (n={len(low_sentiment)}): Mean PnL$ = ${low_mean:.2f}, Std = ${low_std:.2f}")
print("\nLevene's Test for Equal Variances:")
print(f"Statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")
print(f"Equal variances: {'Assumed' if equal_var else 'Not assumed'}")
print("\nOne-Tailed t-Test:")
print(f"t-statistic = {t_stat:.4f}, p-value = {p_value_one_tailed:.4f}")
print(f"Result: {'Reject H₀' if p_value_one_tailed < 0.05 else 'Fail to reject H₀'} (α = 0.05)")

# Plot: PnL$ Distribution by Sentiment Group
plt.figure(figsize=(10, 6))
sns.histplot(high_sentiment, bins=15, color='#10B981', alpha=0.5, label='High Sentiment (Score >= 3)')
sns.histplot(low_sentiment, bins=15, color='coral', alpha=0.5, label='Low Sentiment (Score < 3)')
plt.axvline(high_mean, color='#10B981', linestyle='--', label=f'High Mean: ${high_mean:.2f}')
plt.axvline(low_mean, color='coral', linestyle='--', label=f'Low Mean: ${low_mean:.2f}')
plt.title('PnL$ Distribution by Sentiment Group', fontsize=14, pad=10)
plt.xlabel('PnL$ (USD)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, axis='y')
plt.tight_layout()
try:
    plt.savefig('pnl_distribution_by_sentiment.png', dpi=300, bbox_inches='tight')
    print("Saved: pnl_distribution_by_sentiment.png")
except Exception as e:
    print(f"Error saving pnl_distribution_by_sentiment.png: {e}")
plt.close()

# Key Insights
print("\n=== Key Insights ===")
if p_value_one_tailed < 0.05:
    print(f"1. Significant Result: Trading on high-sentiment days (Sentiment_Score >= 3) yields significantly higher PnL$ (p = {p_value_one_tailed:.4f}).")
else:
    print(f"1. Non-Significant Result: No evidence that high-sentiment days yield higher PnL$ (p = {p_value_one_tailed:.4f}).")
print(f"2. Mean Comparison: High-sentiment mean PnL$ (${high_mean:.2f}) vs. low-sentiment (${low_mean:.2f}).")
print("3. Application: If significant, prioritize trading on Greed/Extreme Greed days to maximize profits.")
print("\nPlot saved successfully. Include 'pnl_distribution_by_sentiment.png' in the updated PDF report.")