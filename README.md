# DSA-210 Project Proposal: Optimizing Crypto Trading and Portfolio Risk Through Sentiment and Market Volatility Analysis

## Motivation
Cryptocurrency markets are characterized by **high volatility**, **sentiment-driven movements**, and **unique opportunities**. As an active trader, I’m motivated to leverage data science to **maximize trading profits** and **stabilize portfolio risk**. This project explores: **How do news sentiment, social media trends, and market volatility metrics influence my trading performance and portfolio stability?** By combining my personal trading and portfolio data with external sentiment and volatility indicators, I aim to uncover actionable insights—such as optimal trading windows or risk-hedging triggers—blending the art of crypto trading with rigorous scientific analysis.

## Data Sources
I’ll collect a comprehensive dataset from **March 11 to May 30, 2025 (81 days)**, integrating personal and external data to quantify trading outcomes, portfolio risk, and their drivers. Below are the specific variables I’ll measure, accompanied by sample tables for clarity:

### Personal Data

#### Trading Logs
Detailed records of my cryptocurrency trades from my trading account.  
**Variables to Measure:**  
- **Date:** Exact timestamp of trade execution (e.g., `2025-03-11 14:30:00 UTC`).  
- **Coin:** Cryptocurrency traded (focus on a few pairs).  
- **Buy Price:** USD price at purchase (e.g., `$62,345.67` for BTC).  
- **Sell Price:** USD price at sale (e.g., `$63,012.89` for BTC).  
- **Amount:** Quantity traded (e.g., `0.0125 BTC`).  
- **Profit/Loss:** Net USD gain/loss per trade (e.g., `Sell Price × Amount - Buy Price × Amount = $8.34`).  
**Purpose:** Quantify trading performance and correlate with external factors.  
**Sample Table:**

| Date                    | Coin | Buy Price  | Sell Price | Amount     | Profit/Loss |
|-------------------------|------|------------|------------|------------|-------------|
| 2025-03-11 14:30:00 UTC | BTC  | $62,345.67 | $63,012.89 | 0.0125 BTC | $8.34       |
| 2025-03-12 09:15:00 UTC | ETH  | $2,450.12  | $2,480.00  | 0.5 ETH    | $14.94      |

#### Portfolio Logs
Daily assessment of my crypto holdings and their risk profile.  
**Variables to Measure:**  
- **Date:** Recorded daily at `08:00 UTC` (morning snapshot).  
- **USD Value:** Total value based on CoinMarketCap’s `08:00 UTC` price (e.g., `0.045 BTC × $62,500 = $2,812.50`).  
- **Daily Return:** Percentage change from previous day (e.g., `(Value_today - Value_yesterday) / Value_yesterday × 100 = +2.3%`).  
**Purpose:** Track portfolio growth and calculate volatility as a risk metric.  
**Sample Table:**

| Date                    | USD Value | Daily Return |
|-------------------------|-----------|--------------|
| 2025-03-11 08:00:00 UTC | $2,812.50 | +2.3%        |
| 2025-03-12 08:00:00 UTC | $2,875.00 | +2.2%        |

### External Data

#### Crypto News
Sentiment from daily headlines on CoinTelegraph.  
**Variables to Measure:**  
- **Date:** Publication date (e.g., `2025-03-11`).  
- **Headline:** Full text of top 3 daily headlines (e.g., `"Bitcoin ETF Approval Boosts Prices"`).  
- **Sentiment Score:** Computed later via NLTK VADER (range `-1` to `+1`, e.g., `+0.75` for positive news).  
**Purpose:** Capture media-driven market sentiment.  
**Sample Table:**

| Date       | Headline                            | Sentiment Score |
|------------|-------------------------------------|-----------------|
| 2025-03-11 | "Bitcoin ETF Approval Boosts Prices" | +0.75           |
| 2025-03-12 | "Crypto Market Faces Regulatory Hurdles" | -0.45       |

#### X Posts
Social media sentiment from cryptocurrency-related posts.  
**Variables to Measure:**  
- **Date:** Post timestamp (e.g., `2025-03-11 09:15:00 UTC`).  
- **Text:** Full post content, filtered by hashtags `#Bitcoin` and `#Crypto` (e.g., `"BTC breaking $65K, bullish!"`).  
- **Sentiment Score:** Computed later via NLTK VADER (e.g., `+0.62`, averaged across 50 posts/day).  
**Purpose:** Assess crowd sentiment and its impact on my trades.  
**Sample Table:**

| Date                    | Text                          | Sentiment Score |
|-------------------------|-------------------------------|-----------------|
| 2025-03-11 09:15:00 UTC | "BTC breaking $65K, bullish!" | +0.62           |
| 2025-03-12 10:00:00 UTC | "Market dip, time to panic?"  | -0.38           |

#### Crypto Fear & Greed Index
Measure daily market sentiment.
**Variables to Measure:**  
- **Date:** Recorded daily (e.g., `2025-03-11`).  
- **Index Value:** Score from `0` (Extreme Fear) to `100` (Extreme Greed) (e.g., `68`).  
**Purpose:** Quantify broader market mood as a volatility proxy.  
**Sample Table:**

| Date       | Index Value |
|------------|-------------|
| 2025-03-11 | 68          |
| 2025-03-12 | 55          |

#### Market Prices
Daily price movements for BTC and ETH from CoinMarketCap API.  
**Variables to Measure:**  
- **Date:** Daily at `00:00 UTC` (e.g., `2025-03-11`).  
- **Open Price:** USD value at midnight UTC (e.g., `$62,400` for BTC).  
- **Close Price:** USD value at `23:59 UTC` (e.g., `$63,100` for BTC).  
- **High Price:** Daily peak (e.g., `$63,500`).  
- **Low Price:** Daily trough (e.g., `$62,200`).  
- **Volatility:** Calculated as `(High - Low) / Open × 100` (e.g., `2.08%`).  
**Purpose:** Provide context for portfolio value and risk calculations.  
**Sample Table:**

| Date       | Open Price | Close Price | High Price | Low Price | Volatility |
|------------|------------|-------------|------------|-----------|------------|
| 2025-03-11 | $62,400    | $63,100     | $63,500    | $62,200   | 2.08%      |
| 2025-03-12 | $63,000    | $62,800     | $63,300    | $62,500   | 1.27%      |

## Data Collection Plan
I’ll gather these variables systematically to ensure consistency and reliability:

- **Trading Logs:**  
  - *Method:* Manually export trade history from my trading account after each session (typically 1–3 trades/day), input into `trades.csv`.  
  - *Tools:* Trading account trade history CSV export, manual entry into Pandas dataframe.  
  - *Frequency:* After every trade, logged within 24 hours.

- **Portfolio Logs:**  
  - *Method:* Record holdings at `08:00 UTC` daily using CoinMarketCap’s live prices, entered into `portfolio.csv`.  
  - *Tools:* CoinMarketCap website for price checks, manual CSV updates.  
  - *Frequency:* Daily, starting March 11.

- **Crypto News:**  
  - *Method:* Scrape CoinTelegraph’s top 3 headlines daily using a Python script (`scrape_news.py`) with BeautifulSoup, saved to `news.csv`.  
  - *Tools:* Python, BeautifulSoup, Requests library.  
  - *Frequency:* Daily at `10:00 UTC`, targeting fresh morning news.

- **X Posts:**  
  - *Method:* Fetch 50 posts/day with hashtags `#Bitcoin` and `#Crypto` using Tweepy (`scrape_x.py`), stored in `x_posts.csv`.  
  - *Tools:* Python, Tweepy (requires X Developer API key—I’ll apply by March 15).  
  - *Frequency:* Daily at `12:00 UTC`, capturing peak activity.

- **Fear & Greed Index:**  
  - *Method:* Manually copy daily values into `fear_greed.csv` until an API is sourced.  
  - *Tools:* Manual entry, potential Python script if automated later.  
  - *Frequency:* Daily at `09:00 UTC`.

- **Market Prices:**  
  - *Method:* Pull BTC and ETH daily prices via CoinMarketCap API using a Python script (`fetch_prices.py`), saved to `prices.csv`.  
  - *Tools:* Python, CoinMarketCap API (free tier, key obtained by March 10).  
  - *Frequency:* Daily at `01:00 UTC`, post-midnight update.

- **Storage and Integration:**  
  - Merge all data into a master CSV (`crypto_data.csv`) with columns matching the variables above, indexed by date.  
  - Use Pandas to align timestamps and handle missing values (e.g., interpolate prices, drop incomplete trade days).  
  - Backup daily CSVs in the GitHub repo under a `/data` folder.

- **Quality Control:**  
  - Cross-check trade data with exchange records weekly.  
  - Verify portfolio values against CoinMarketCap API outputs.  
  - Ensure sentiment sources (news, X) are crypto-specific to avoid noise.

## Analysis Plan
- **By April 18:** Clean data (e.g., remove outliers >$500 profit/loss), calculate sentiment scores, compute portfolio volatility (7-day rolling standard deviation), and explore correlations (e.g., profit vs. sentiment, risk vs. Fear & Greed) with visualizations (scatter plots, heatmaps).  
- **By May 23:** Build machine learning models (e.g., Random Forest for trade profit prediction, Gradient Boosting for risk forecasting) using Scikit-learn, evaluating with R² and RMSE.  
- **By May 30:** Present findings (e.g., “Days with Fear & Greed >70 increase portfolio volatility by 1.5%”) in a final report.

## Tools
- **Programming:** Python 3.11 for all scripting and analysis.  
- **Libraries:**  
  - `Pandas`: Data manipulation and merging.  
  - `BeautifulSoup`/`Requests`: News scraping.  
  - `Tweepy`: X post retrieval.  
  - `NLTK`: Sentiment analysis.  
  - `Scikit-learn`: Machine learning models.  
  - `Matplotlib`/`Seaborn`: Visualizations.  
- **GitHub:** Version control and submission platform (this repo).  
- **Dependencies:** Listed in `requirements.txt` for reproducibility.

## Ethical Considerations
- **Data Privacy:** Personal trading logs will be anonymized to protect financial information, ensuring no sensitive details (e.g., account numbers) are exposed.
- **Responsible Data Use:** News and social media data will be scraped responsibly, adhering to website terms of service and citing sources appropriately (e.g., CoinTelegraph, X posts).
- **AI Tool Transparency:** Any use of AI tools (e.g., LLMs) for coding or analysis will be documented, including prompts and outputs, as per the academic integrity policy.

## Project Structure
- `/data`: Raw and processed datasets (e.g., trades.csv, processed_data.csv)
- `/scripts`: Python scripts (e.g., scrape_news.py, pre_processing.py)
- `/outputs`: Visualizations and intermediate results
- `/reports`: Final reports and documentation (e.g., Hypothesis Testing Results.pdf)
- `README.md`: Project proposal and instructions
- `requirements.txt`: Python dependencies

## Setup and Reproducibility
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download or generate datasets as described in the Data Collection Plan.
4. Run scripts in the `/scripts` folder (e.g., `python scrape_news.py`).

## Notes
- Data collection begins **March 11**, post-submission, to align with the course timeline.  
- If X API access is delayed, I’ll manually sample 10 posts/day and scale up later, noting limitations.  
- I’ll apply for API keys (**CoinMarketCap** by March 10, **X** by March 15) to ensure automation readiness.
