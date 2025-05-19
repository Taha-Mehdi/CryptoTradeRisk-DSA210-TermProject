import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/"
}

# Function to fetch Fear & Greed Index for a specific date
def fetch_fear_greed(date):
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://api.alternative.me/fng/?date={date_str}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["data"]:
            index_value = int(data["data"][0]["value"])
            # Determine sentiment based on index value
            if index_value <= 24:
                sentiment = "Extreme Fear"
            elif index_value <= 49:
                sentiment = "Fear"
            elif index_value <= 74:
                sentiment = "Neutral"
            elif index_value <= 89:
                sentiment = "Greed"
            else:
                sentiment = "Extreme Greed"
        else:
            print(f"No data for {date_str}. Using default.")
            index_value, sentiment = 50, "Neutral"

        print(f"Fetched Fear & Greed Index for {date_str}: {index_value} ({sentiment})")
        return index_value, sentiment

    except requests.RequestException as e:
        print(f"Error fetching data for {date_str}: {e}")
        return 50, "Neutral"
    except Exception as e:
        print(f"Error parsing data for {date_str}: {e}")
        return 50, "Neutral"

# Main function to scrape and save data for the date range
def main():
    start_date = datetime(2025, 3, 11)
    end_date = datetime(2025, 5, 30)
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # Load existing data if available
    try:
        existing_df = pd.read_csv("fear_greed.csv")
        existing_dates = set(existing_df["Date"])
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=["Date", "Value", "Sentiment"])
        existing_dates = set()

    # Collect new data
    data = []
    for date in date_range:
        date_str = date.strftime("%m/%d/%Y")
        if date_str in existing_dates:
            continue  # Skip dates already collected

        print(f"Fetching Fear & Greed Index for {date_str}...")
        index_value, sentiment = fetch_fear_greed(date)
        data.append([date_str, index_value, sentiment])
        time.sleep(1)  # Avoid overwhelming the API

    if data:
        # Create DataFrame for new data
        new_df = pd.DataFrame(data, columns=["Date", "Value", "Sentiment"])
        # Append to existing data
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Save to CSV
        updated_df.to_csv("fear_greed.csv", index=False)
        print("Updated Fear & Greed Index data in fear_greed.csv")
    else:
        print("No new data to add.")

if __name__ == "__main__":
    main()