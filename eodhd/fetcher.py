from datetime import datetime

import requests


class DataFetcher:
    def __init__(self, api_token):
        self.api_token = api_token

    def get_oldest_available_date(self, ticker):
        """Fetch the oldest available date for a ticker"""
        url = f"https://eodhd.com/api/eod/{ticker}?api_token={self.api_token}&fmt=json&order=a&limit=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"Oldest available date data found for {ticker}: {data[0]['date']}")
                return data[0]['date']
        print(f"Error fetching oldest date for {ticker}: {response.status_code}")
        return "1980-01-01"  # Fallback date

    def fetch_historical_data(self, ticker, period='d', start_date=None, end_date=None):
        """Fetch all historical data in a single API call"""
        if start_date is None:
            start_date = self.get_oldest_available_date(ticker)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Fetching data for {ticker} from {start_date}...")
        url = f"https://eodhd.com/api/eod/{ticker}?api_token={self.api_token}&fmt=json&from={start_date}&to={end_date}&period={period}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data for {ticker}: {response.status_code}")
            return []