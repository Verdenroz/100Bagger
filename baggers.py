import polars as pl
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BaggerType(Enum):
    """Enumeration of different bagger types."""
    HUNDRED_BAGGER = "100-bagger"
    MULTIBAGGER = "multibagger"
    FALLEN_HUNDRED_BAGGER = "fallen_100-bagger"
    FALLEN_MULTIBAGGER = "fallen_multibagger"
    NO_BAGGER = "no_bagger"


@dataclass
class BaggerResult:
    """Result of bagger analysis for a single ticker."""
    ticker: str
    bagger_type: BaggerType
    max_return_multiple: float
    final_return_multiple: float
    start_price: float
    max_price: float
    final_price: float
    start_date: str
    max_date: str
    final_date: str
    total_days: int
    days_to_peak: int


class TickerBaggerAnalyzer:
    """Analyzes individual tickers to categorize them as different types of baggers."""

    def __init__(self, partitioned_data_dir: str = "stock_data_partitioned"):
        """Initialize the analyzer.

        Args:
            partitioned_data_dir: Directory containing partitioned parquet files
        """
        self.partitioned_data_dir = Path(partitioned_data_dir)

    def analyze_ticker(self, ticker: str, min_days: int = 252) -> Optional[BaggerResult]:
        """Analyze a single ticker for bagger classification.

        Args:
            ticker: Stock ticker symbol to analyze
            min_days: Minimum number of trading days required for analysis (default: 252 = ~1 year)

        Returns:
            BaggerResult if ticker can be analyzed, None if insufficient data or errors
        """
        try:
            # Load ticker data
            ticker_data = self._load_ticker_data(ticker)
            if ticker_data is None or len(ticker_data) < min_days:
                return None

            # Sort by date and calculate returns
            df = (ticker_data
                  .sort("date")
                  .with_columns([
                pl.col("date").str.to_date(),
                pl.col("adjusted_close").alias("price")
            ])
                  .filter(pl.col("price").is_not_null() & (pl.col("price") > 0)))

            if len(df) < min_days:
                return None

            # Calculate cumulative returns from start
            start_price = df["price"][0]
            df = df.with_columns([
                (pl.col("price") / start_price).alias("return_multiple")
            ])

            # Find maximum return and final return
            max_return_idx = df["return_multiple"].arg_max()
            max_return_multiple = df["return_multiple"][max_return_idx]
            final_return_multiple = df["return_multiple"][-1]

            # Extract key metrics
            start_date = str(df["date"][0])
            max_date = str(df["date"][max_return_idx])
            final_date = str(df["date"][-1])

            max_price = df["price"][max_return_idx]
            final_price = df["price"][-1]

            total_days = len(df)
            days_to_peak = max_return_idx + 1

            # Classify bagger type
            bagger_type = self._classify_bagger(max_return_multiple, final_return_multiple)

            return BaggerResult(
                ticker=ticker,
                bagger_type=bagger_type,
                max_return_multiple=max_return_multiple,
                final_return_multiple=final_return_multiple,
                start_price=start_price,
                max_price=max_price,
                final_price=final_price,
                start_date=start_date,
                max_date=max_date,
                final_date=final_date,
                total_days=total_days,
                days_to_peak=days_to_peak
            )

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            return None

    def _load_ticker_data(self, ticker: str) -> Optional[pl.DataFrame]:
        """Load data for a specific ticker from partitioned parquet files.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with ticker data or None if not found
        """
        ticker_dir = self.partitioned_data_dir / f"code={ticker}"
        parquet_file = ticker_dir / "data.parquet"

        if not parquet_file.exists():
            return None

        try:
            return pl.read_parquet(parquet_file)
        except Exception:
            return None

    def _classify_bagger(self, max_return: float, final_return: float) -> BaggerType:
        """Classify the bagger type based on max and final returns.

        Args:
            max_return: Maximum return multiple achieved
            final_return: Final return multiple

        Returns:
            BaggerType classification
        """
        # Current 100-bagger (still 100x+)
        if final_return >= 100:
            return BaggerType.HUNDRED_BAGGER

        # Current multibagger (still 10x+)
        elif final_return >= 10:
            return BaggerType.MULTIBAGGER

        # Fallen 100-bagger (peaked at 100x+ but not anymore)
        elif max_return >= 100 and final_return < 100:
            return BaggerType.FALLEN_HUNDRED_BAGGER

        # Fallen multibagger (peaked at 10x+ but not anymore)
        elif max_return >= 10 and final_return < 10:
            return BaggerType.FALLEN_MULTIBAGGER

        # No significant returns
        else:
            return BaggerType.NO_BAGGER

    def get_available_tickers(self) -> list[str]:
        """Get list of all available tickers in the partitioned data.

        Returns:
            List of ticker symbols
        """
        tickers = []
        for partition_dir in self.partitioned_data_dir.iterdir():
            if partition_dir.is_dir() and partition_dir.name.startswith("code="):
                ticker = partition_dir.name.replace("code=", "")
                tickers.append(ticker)
        return sorted(tickers)


def analyze_single_ticker_example():
    """Example function showing how to analyze a single ticker."""
    analyzer = TickerBaggerAnalyzer()

    # Analyze a specific ticker
    ticker = "AADBX"
    result = analyzer.analyze_ticker(ticker)

    if result:
        print(f"\n--- Analysis for {result.ticker} ---")
        print(f"Classification: {result.bagger_type.value}")
        print(f"Start Price: ${result.start_price:.2f} ({result.start_date})")
        print(f"Max Price: ${result.max_price:.2f} ({result.max_date}) - {result.max_return_multiple:.1f}x")
        print(f"Final Price: ${result.final_price:.2f} ({result.final_date}) - {result.final_return_multiple:.1f}x")
        print(f"Total Trading Days: {result.total_days}")
        print(f"Days to Peak: {result.days_to_peak}")

        if result.bagger_type in [BaggerType.FALLEN_HUNDRED_BAGGER, BaggerType.FALLEN_MULTIBAGGER]:
            peak_to_final = (result.final_return_multiple / result.max_return_multiple) * 100
            print(f"Peak to Final: {peak_to_final:.1f}% of peak value")
    else:
        print(f"Could not analyze {ticker} - insufficient data or ticker not found")


def batch_analyze_tickers(tickers: list[str], partitioned_data_dir: str = "stock_data_partitioned") -> list[BaggerResult]:
    """Analyze multiple tickers in batch.

    Args:
        tickers: List of ticker symbols to analyze
        partitioned_data_dir: Directory containing partitioned data

    Returns:
        List of BaggerResult objects (only successful analyses)
    """
    analyzer = TickerBaggerAnalyzer(partitioned_data_dir)
    results = []

    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0:
            print(f"Processed {i}/{len(tickers)} tickers...")

        result = analyzer.analyze_ticker(ticker)
        if result:
            results.append(result)

    return results


def save_results_to_parquet(results: list[BaggerResult], output_file: str = "bagger_analysis_results.parquet"):
    """Save analysis results to a parquet file.

    Args:
        results: List of BaggerResult objects
        output_file: Output parquet file path
    """
    if not results:
        print("No results to save")
        return

    # Convert results to DataFrame
    data = {
        "ticker": [r.ticker for r in results],
        "bagger_type": [r.bagger_type.value for r in results],
        "max_return_multiple": [r.max_return_multiple for r in results],
        "final_return_multiple": [r.final_return_multiple for r in results],
        "start_price": [r.start_price for r in results],
        "max_price": [r.max_price for r in results],
        "final_price": [r.final_price for r in results],
        "start_date": [r.start_date for r in results],
        "max_date": [r.max_date for r in results],
        "final_date": [r.final_date for r in results],
        "total_days": [r.total_days for r in results],
        "days_to_peak": [r.days_to_peak for r in results]
    }

    df = pl.DataFrame(data)
    df.write_parquet(output_file, compression='snappy')
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    print("=== Single Ticker Analysis Example ===")
    analyze_single_ticker_example()

    # # Example of how to get available tickers
    # analyzer = TickerBaggerAnalyzer()
    # available_tickers = analyzer.get_available_tickers()
    # print(f"\nFound {len(available_tickers)} available tickers")
    #
    # # Example batch analysis of first 10 tickers
    # if available_tickers:
    #     print(f"\n=== Batch Analysis Example (first 10 tickers) ===")
    #     sample_tickers = available_tickers[:10]
    #     results = batch_analyze_tickers(sample_tickers)
    #
    #     print(f"\nSuccessfully analyzed {len(results)} out of {len(sample_tickers)} tickers")
    #
    #     # Show summary by bagger type
    #     bagger_counts = {}
    #     for result in results:
    #         bagger_type = result.bagger_type.value
    #         bagger_counts[bagger_type] = bagger_counts.get(bagger_type, 0) + 1
    #
    #     print("\nBagger Type Summary:")
    #     for bagger_type, count in sorted(bagger_counts.items()):
    #         print(f"  {bagger_type}: {count}")
    #
    #     # Save results
    #     if results:
    #         save_results_to_parquet(results, "sample_bagger_analysis.parquet")