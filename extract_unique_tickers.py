#!/usr/bin/env python3
"""
Extract unique tickers from bagger analysis results and save to CSV.
"""

import pandas as pd
import sys
from pathlib import Path

def extract_unique_tickers():
    """Extract unique tickers from bagger analysis results."""
    
    # Try parquet first, fallback to CSV
    parquet_file = "bagger_analysis_results.parquet"
    csv_file = "bagger_analysis_results.csv"
    
    try:
        if Path(parquet_file).exists():
            print(f"Reading from {parquet_file}")
            df = pd.read_parquet(parquet_file)
        elif Path(csv_file).exists():
            print(f"Reading from {csv_file}")
            df = pd.read_csv(csv_file)
        else:
            print("Error: Neither parquet nor CSV file found!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Extract unique tickers
    unique_tickers = df['ticker'].unique()
    unique_tickers = sorted(unique_tickers)
    
    print(f"Found {len(unique_tickers)} unique tickers")
    
    # Create DataFrame and save to CSV
    ticker_df = pd.DataFrame({'ticker': unique_tickers})
    output_file = "unique_tickers.csv"
    ticker_df.to_csv(output_file, index=False)
    
    print(f"Saved unique tickers to {output_file}")
    print(f"First 10 tickers: {unique_tickers[:10]}")

if __name__ == "__main__":
    extract_unique_tickers()