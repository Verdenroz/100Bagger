from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import polars as pl


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

    def analyze_ticker(self, ticker: str, min_days: int = 1) -> Optional[BaggerResult]:
        """Analyze a single ticker for bagger classification.

        Args:
            ticker: Stock ticker symbol to analyze
            min_days: Minimum number of trading days required for analysis (default: 1)

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


def save_results_to_csv(results: list[BaggerResult], output_file: str = "bagger_analysis_results.csv"):
    """Save analysis results to a CSV file.

    Args:
        results: List of BaggerResult objects
        output_file: Output CSV file path
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
    df.write_csv(output_file)
    print(f"Results saved to {output_file}")


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


def analyze_all_tickers(partitioned_data_dir: str = "stock_data_partitioned",
                        progress_interval: int = 1000) -> list[BaggerResult]:
    """Analyze all available tickers in the partitioned data.

    Args:
        partitioned_data_dir: Directory containing partitioned data
        progress_interval: How often to print progress updates

    Returns:
        List of BaggerResult objects (only successful analyses)
    """
    analyzer = TickerBaggerAnalyzer(partitioned_data_dir)

    # Get all available tickers
    print("Discovering available tickers...")
    all_tickers = analyzer.get_available_tickers()
    print(f"Found {len(all_tickers)} tickers to analyze")

    results = []
    failed_count = 0

    print("Starting batch analysis of all tickers...")
    for i, ticker in enumerate(all_tickers, 1):
        if i % progress_interval == 0:
            success_rate = ((i - failed_count) / i) * 100
            print(f"Processed {i:,}/{len(all_tickers):,} tickers... "
                  f"Success rate: {success_rate:.1f}% "
                  f"({len(results):,} successful, {failed_count:,} failed)")

        result = analyzer.analyze_ticker(ticker)
        if result:
            results.append(result)
        else:
            failed_count += 1

    print(f"\n✅ Analysis complete!")
    print(f"Successfully analyzed: {len(results):,} tickers")
    print(f"Failed to analyze: {failed_count:,} tickers")
    print(f"Success rate: {(len(results) / len(all_tickers)) * 100:.1f}%")

    return results


def print_summary_stats(results: list[BaggerResult]):
    """Print summary statistics of the analysis results.

    Args:
        results: List of BaggerResult objects
    """
    if not results:
        print("No results to summarize")
        return

    print(f"\n=== BAGGER ANALYSIS SUMMARY ===")
    print(f"Total analyzed tickers: {len(results):,}")

    # Count by bagger type
    bagger_counts = {}
    for result in results:
        bagger_type = result.bagger_type.value
        bagger_counts[bagger_type] = bagger_counts.get(bagger_type, 0) + 1

    print(f"\nBagger Type Distribution:")
    for bagger_type in [BaggerType.HUNDRED_BAGGER.value, BaggerType.MULTIBAGGER.value,
                        BaggerType.FALLEN_HUNDRED_BAGGER.value, BaggerType.FALLEN_MULTIBAGGER.value,
                        BaggerType.NO_BAGGER.value]:
        count = bagger_counts.get(bagger_type, 0)
        percentage = (count / len(results)) * 100
        print(f"  {bagger_type}: {count:,} ({percentage:.1f}%)")

    # Top performers
    hundred_baggers = [r for r in results if r.bagger_type == BaggerType.HUNDRED_BAGGER]
    multibaggers = [r for r in results if r.bagger_type == BaggerType.MULTIBAGGER]
    fallen_hundred = [r for r in results if r.bagger_type == BaggerType.FALLEN_HUNDRED_BAGGER]

    if hundred_baggers:
        top_current = sorted(hundred_baggers, key=lambda x: x.final_return_multiple, reverse=True)[:5]
        print(f"\nTop 5 Current 100-Baggers (by final return):")
        for r in top_current:
            print(f"  {r.ticker}: {r.final_return_multiple:.1f}x (peak: {r.max_return_multiple:.1f}x)")

    if multibaggers:
        top_multis = sorted(multibaggers, key=lambda x: x.final_return_multiple, reverse=True)[:5]
        print(f"\nTop 5 Current Multibaggers (by final return):")
        for r in top_multis:
            print(f"  {r.ticker}: {r.final_return_multiple:.1f}x (peak: {r.max_return_multiple:.1f}x)")

    if fallen_hundred:
        top_fallen = sorted(fallen_hundred, key=lambda x: x.max_return_multiple, reverse=True)[:5]
        print(f"\nTop 5 Fallen 100-Baggers (by peak return):")
        for r in top_fallen:
            drawdown = ((r.max_return_multiple - r.final_return_multiple) / r.max_return_multiple) * 100
            print(f"  {r.ticker}: peaked at {r.max_return_multiple:.1f}x, now {r.final_return_multiple:.1f}x ({drawdown:.1f}% drawdown)")


if __name__ == "__main__":
    print("=== COMPREHENSIVE BAGGER ANALYSIS ===")

    # Analyze all tickers
    results = analyze_all_tickers()

    if results:
        # Print summary statistics
        print_summary_stats(results)

        # Save to both CSV and Parquet
        print(f"\nSaving results...")
        save_results_to_csv(results, "bagger_analysis_results.csv")
        save_results_to_parquet(results, "bagger_analysis_results.parquet")

        print(f"\n✅ Analysis complete! Results saved to:")
        print(f"  - bagger_analysis_results.csv")
        print(f"  - bagger_analysis_results.parquet")
        print(f"\nYou can now analyze the results using:")
        print(f"  df = pl.read_csv('bagger_analysis_results.csv')")
        print(f"  # or")
        print(f"  df = pl.read_parquet('bagger_analysis_results.parquet')")
    else:
        print("❌ No successful analyses. Check your data directory and ticker files.")