import pandas as pd
import csv
from datetime import datetime

# Step 1: Load all datasets with proper quoting for text-heavy files
fear_greed = pd.read_csv('fear_greed.csv')
news = pd.read_csv('news.csv', quoting=csv.QUOTE_ALL)
prices = pd.read_csv('prices.csv')
social_media = pd.read_csv('social_media.csv', quoting=csv.QUOTE_ALL)
portfolio = pd.read_csv('portfolio.csv')
trades = pd.read_csv('trades.csv', quoting=csv.QUOTE_ALL)

# Verify column names
if 'DateTime' not in portfolio.columns:
    raise ValueError("Error: 'DateTime' column not found in portfolio.csv. Found columns: {}".format(portfolio.columns.tolist()))

if 'DateTime' not in trades.columns or 'Date' not in trades.columns:
    raise ValueError("Error: 'DateTime' or 'Date' column not found in trades.csv. Found columns: {}".format(trades.columns.tolist()))

# Step 2: Clean news.csv (replace any broken URLs, if present)
news['Headline_2'] = news['Headline_2'].replace('tmp://www.example.com', 'N/A')

# Step 3: Standardize date formats
# Convert MM/DD/YYYY to YYYY-MM-DD for fear_greed, news, prices, social_media, trades
for df in [fear_greed, news, prices, social_media, trades]:
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

# Extract date from portfolio (YYYY-MM-DD HH:MM:SS UTC to YYYY-MM-DD)
portfolio['Date'] = pd.to_datetime(portfolio['DateTime'].str.replace(' UTC', '')).dt.strftime('%Y-%m-%d')

# Step 4: Merge datasets on Date
merged_df = prices.merge(fear_greed, on='Date', how='left')\
                 .merge(news, on='Date', how='left')\
                 .merge(social_media, on='Date', how='left')\
                 .merge(portfolio.drop(columns=['DateTime', 'BTC_Price', 'ETH_Price']), on='Date', how='left')\
                 .merge(trades.drop(columns=['DateTime', 'Day']), on='Date', how='left')

# Step 5: Feature Engineering
# Add Day_of_Week
merged_df['Day_of_Week'] = pd.to_datetime(merged_df['Date']).dt.day_name()

# Convert sentiments to numerical scores
sentiment_mapping = {'Extreme Greed': 4, 'Greed': 3, 'Neutral': 2, 'Fear': 1, 'Extreme Fear': 0}
social_sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

merged_df['Sentiment_Score'] = merged_df['Sentiment'].map(sentiment_mapping)
merged_df['Social_Sentiment_1'] = merged_df['Sentiment_1'].map(social_sentiment_mapping)
merged_df['Social_Sentiment_2'] = merged_df['Sentiment_2'].map(social_sentiment_mapping)
merged_df['Social_Sentiment_3'] = merged_df['Sentiment_3'].map(social_sentiment_mapping)

# Calculate daily returns
merged_df['BTC_Daily_Return'] = (merged_df['BTC_Close'] - merged_df['BTC_Open']) / merged_df['BTC_Open'] * 100
merged_df['ETH_Daily_Return'] = (merged_df['ETH_Close'] - merged_df['ETH_Open']) / merged_df['ETH_Open'] * 100

# Step 6: Save preprocessed data
merged_df.to_csv('preprocessed_data.csv', index=False)

print("Preprocessing complete. Saved as 'preprocessed_data.csv'.")