import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

# Function to scrape headlines from CoinTelegraph
def scrape_crypto_headlines():
    url = "https://cointelegraph.com/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        # Send HTTP request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find headline elements (adjust selector based on site structure)
        articles = soup.select("article a span")  # CoinTelegraph's headline spans
        headlines = [article.get_text().strip() for article in articles if article.get_text().strip()]

        # Filter for BTC, ETH, and other crypto news
        btc_headline = None
        eth_headline = None
        other_headline = None

        for headline in headlines:
            headline_lower = headline.lower()
            if "bitcoin" in headline_lower or "btc" in headline_lower:
                if not btc_headline:
                    btc_headline = headline
            elif "ethereum" in headline_lower or "eth" in headline_lower:
                if not eth_headline:
                    eth_headline = headline
            elif not other_headline and any(keyword in headline_lower for keyword in ["crypto", "blockchain", "defi", "nft", "solana", "cardano"]):
                other_headline = headline

            # Stop if we have all three headlines
            if btc_headline and eth_headline and other_headline:
                break

        # Fallback if not enough headlines
        if not btc_headline:
            btc_headline = "No BTC headline found"
        if not eth_headline:
            eth_headline = "No ETH headline found"
        if not other_headline:
            other_headline = "No other crypto headline found"

        return btc_headline, eth_headline, other_headline

    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        return "Error fetching BTC headline", "Error fetching ETH headline", "Error fetching other headline"

# Function to save headlines to CSV
def save_to_csv(btc_headline, eth_headline, other_headline):
    date_str = datetime.now().strftime("%m/%d/%Y")
    data = {
        "Date": [date_str],
        "Headline_1": [btc_headline],
        "Headline_2": [eth_headline],
        "Headline_3": [other_headline]
    }

    df = pd.DataFrame(data)

    # Append to CSV (create if doesn't exist)
    try:
        with open("news.csv", "a", newline="") as f:
            df.to_csv(f, header=f.tell()==0, index=False)
        print("Data saved to news.csv")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Function to display headlines
def display_headlines(btc_headline, eth_headline, other_headline):
    date_str = datetime.now().strftime("%m/%d/%Y")
    print(f"\nHeadlines for {date_str}:")
    print(f"BTC: {btc_headline}")
    print(f"ETH: {eth_headline}")
    print(f"Other: {other_headline}")

# Main function
def main():
    print("Scraping crypto news headlines...")
    btc_headline, eth_headline, other_headline = scrape_crypto_headlines()

    # Display the headlines
    display_headlines(btc_headline, eth_headline, other_headline)

    # Save to CSV
    save_to_csv(btc_headline, eth_headline, other_headline)

    print("\nYou can now manually edit 'news.csv' to adjust headlines if needed.")

if __name__ == "__main__":
    main()