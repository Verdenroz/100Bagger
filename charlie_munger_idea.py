import concurrent.futures
import os

import pandas as pd
from dotenv import load_dotenv

from eodhd.fetcher import DataFetcher
from eodhd.processor import MarketDataProcessor
from strategies.trading import EMA200CrossoverStrategy, GoldenCrossStrategy, MungerStrategy
from visuals.strategy_plot import StrategyPlotVisualizer


class StockAnalyzer:
    def __init__(self, api_token):
        self.data_fetcher = DataFetcher(api_token)
        self.data_processor = MarketDataProcessor()
        self.strategies = [
            EMA200CrossoverStrategy(),
            GoldenCrossStrategy(),
            MungerStrategy()
        ]
        self.visualizer = StrategyPlotVisualizer()

    def analyze_ticker(self, ticker, initial_capital=10000):
        """Complete analysis process for a ticker"""
        try:
            # Fetch and process data
            raw_data = self.data_fetcher.fetch_historical_data(ticker)
            if not raw_data:
                print(f"No data fetched for {ticker}")
                return None, [], []

            df = self.data_processor.process_raw_data(raw_data)

            if df is None or df.empty:
                print(f"No valid data available for {ticker}")
                return None, [], []

            # Calculate indicators
            df = self.data_processor.calculate_indicators(df)

            if df is None or df.empty:
                print(f"Insufficient data for technical indicators for {ticker}")
                return None, [], []

            # DEBUG: Print dataframe info and sample
            print(f"\nDEBUG INFO for {ticker}:")
            print(f"DataFrame shape: {df.shape}")
            print(f"Contains NaN values: {df.isna().any().any()}")
            print(f"Column types: {df.dtypes}")
            print(f"First few rows:\n{df.head()}")

            # Calculate performances for each strategy
            performances = []
            metrics_list = []

            for strategy in self.strategies:
                try:
                    print(f"Calculating performance for {ticker} with {strategy.name}...")
                    performance, metrics = strategy.calculate_performance(df, initial_capital)

                    # DEBUG: Check performance data
                    print(f"  Portfolio shape: {performance.shape}")
                    print(f"  Contains strategy_portfolio: {'strategy_portfolio' in performance.columns}")
                    if 'strategy_portfolio' in performance.columns:
                        print(
                            f"  Strategy portfolio range: {performance['strategy_portfolio'].min()} to {performance['strategy_portfolio'].max()}")

                    performances.append(performance)
                    metrics_list.append(metrics)
                except Exception as e:
                    print(f"Error calculating performance for {ticker} with {strategy.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add empty placeholders to maintain indexes
                    performances.append(pd.DataFrame())
                    metrics_list.append({"Strategy Name": strategy.name, "Error": str(e)})

            # Create visualizations
            try:
                self.visualizer.plot_equity_curves(performances, metrics_list, ticker)
                self.visualizer.plot_signals(df, ticker)
            except Exception as e:
                print(f"Error creating visualizations for {ticker}: {e}")
                import traceback
                traceback.print_exc()

            print(f"Analysis completed for {ticker}")
            return df, performances, metrics_list

        except Exception as e:
            print(f"Error in analysis process for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None, [], []

    def analyze_multiple_tickers(self, tickers, initial_capital=10000):
        """Analyze multiple tickers concurrently"""
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(self.analyze_ticker, ticker, initial_capital): ticker
                for ticker in tickers
            }

            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df, performances, metrics_list = future.result()
                    if df is not None:
                        results[ticker] = (df, performances, metrics_list)
                except Exception as e:
                    print(f"Error analyzing {ticker}: {e}")

        # Wait for all visualization tasks to complete
        print("Waiting for all visualizations to complete...")
        self.visualizer.wait_for_completion()
        print("All visualizations completed!")

        return results

    def print_summary(self, results):
        """Print summary of results for all tickers"""
        print("\nSummary of Results:")
        for ticker, (_, performances, metrics_list) in results.items():
            print(f"\n{ticker}:")
            for metrics in metrics_list:
                print(f"  {metrics['Strategy Name']}:")
                print(
                    f"    Return: {metrics['Strategy Return (%)']:.2f}% (vs Buy & Hold: {metrics['Buy & Hold Return (%)']:.2f}%)")
                print(f"    Annualized: {metrics['Annualized Strategy Return (%)']:.2f}%")
                print(f"    Win Rate: {metrics['Win Rate']:.2%}")
                print(f"    Max Drawdown: {metrics['Maximum Drawdown (%)']:.2f}%")


def main():
    load_dotenv()
    api_token = os.environ.get("EODHD_API_TOKEN")
    if not api_token:
        print("API token not found. Please set EODHD_API_TOKEN environment variable.")
        return

    # Initialize analyzer
    analyzer = StockAnalyzer(api_token)

    # Define tickers to analyze
    tickers = [
        "JNJ.US",  # Johnson & Johnson (Healthcare)
        "JPM.US",  # JPMorgan Chase (Financials)
        "XOM.US",  # Exxon Mobil (Energy)
        "AAPL.US",  # Apple (Technology)
        "PG.US",  # Procter & Gamble (Consumer Staples)
        "VZ.US",  # Verizon Communications (Telecommunications)
        "BA.US",  # Boeing (Industrials)
        "KO.US",  # Coca-Cola (Consumer Staples)
    ]

    # For a single ticker analysis:
    # df, performances, metrics_list = analyzer.analyze_ticker("AAPL.US")

    # For multiple tickers:
    results = analyzer.analyze_multiple_tickers(tickers)

    # Print summary
    analyzer.print_summary(results)


if __name__ == "__main__":
    main()
