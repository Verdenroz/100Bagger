import pandas as pd


class MarketDataProcessor:
    @staticmethod
    def process_raw_data(data):
        """Convert raw API data to a pandas DataFrame with proper types"""
        if not data:
            return None

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Convert to numeric types where needed
        for col in ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    @staticmethod
    def calculate_indicators(df):
        """Calculate all technical indicators needed for strategies"""
        if df is None or df.empty:
            return None

        # Calculate 200-day EMA (original strategy)
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Calculate 50-day and 200-day SMA (for Golden Cross strategy)
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # Calculate 200-week SMA (for Munger strategy)
        # Using 5 trading days per week, 200 weeks = 1000 days
        df['SMA_200W'] = df['close'].rolling(window=1000).mean()

        # Alternative calculation if using weekly data directly
        # df['SMA_200W'] = df['close'].rolling(window=200).mean()

        # Store previous day's values to detect crossings
        df['previous_close'] = df['close'].shift(1)
        df['previous_ema200'] = df['EMA_200'].shift(1)
        df['previous_sma50'] = df['SMA_50'].shift(1)
        df['previous_sma200'] = df['SMA_200'].shift(1)
        df['previous_sma200w'] = df['SMA_200W'].shift(1)

        # Drop rows with NaN values in key columns
        # Note: SMA_200W will have many more NaN values due to longer lookback
        return df.dropna(subset=['close', 'EMA_200', 'SMA_50', 'SMA_200', 'SMA_200W'])
