from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict
from typing import Optional, Tuple

import polars as pl


class BaggerType(Enum):
    """Enumeration of different bagger types."""
    HUNDRED_BAGGER = "100-bagger"
    MULTIBAGGER = "multibagger"
    FALLEN_HUNDRED_BAGGER = "fallen_100-bagger"
    FALLEN_MULTIBAGGER = "fallen_multibagger"
    NO_BAGGER = "no_bagger"


@dataclass
class BaggerTransition:
    """Represents a transition between bagger states."""
    from_status: BaggerType
    to_status: BaggerType
    date: str
    price: float
    return_multiple: float
    days_from_start: int


@dataclass
class BaggerMilestone:
    """Represents when a stock hits specific return multiples."""
    multiple: float  # e.g., 10.0 for 10x, 100.0 for 100x
    date: str
    price: float
    days_from_start: int
    maintained_for_days: Optional[int] = None  # How long it stayed above this level


@dataclass
class BaggerResult:
    """Result with full time-series analysis."""
    ticker: str

    # Basic info
    start_price: float
    final_price: float
    start_date: str
    final_date: str
    total_days: int

    # Current status
    current_bagger_type: BaggerType
    current_return_multiple: float

    # Peak information
    max_return_multiple: float
    max_price: float
    max_date: str
    days_to_peak: int

    # Milestone tracking
    milestones: List[BaggerMilestone]

    # Transition history
    transitions: List[BaggerTransition]

    # Time spent in each status
    time_in_status: Dict[BaggerType, int]  # days spent in each status

    # Advanced metrics
    first_10x_date: Optional[str] = None
    first_100x_date: Optional[str] = None
    last_10x_date: Optional[str] = None  # Last time it was above 10x
    last_100x_date: Optional[str] = None  # Last time it was above 100x

    # Drawdown analysis
    max_drawdown_from_peak: float = 0.0
    max_drawdown_date: Optional[str] = None

    # Current streak info
    current_streak_type: BaggerType = BaggerType.NO_BAGGER
    current_streak_days: int = 0
    current_streak_start_date: Optional[str] = None


class TickerBaggerAnalyzer:
    """Analyzer that tracks full bagger journey over time."""

    def __init__(self, partitioned_data_dir: str = "stock_data_partitioned"):
        """Initialize the analyzer.

        Args:
            partitioned_data_dir: Directory containing partitioned parquet files
        """
        self.partitioned_data_dir = Path(partitioned_data_dir)

    def analyze_ticker(self, ticker: str, min_days: int = 252, debug: bool = False) -> Optional[BaggerResult]:
        """Analyze a single ticker with comprehensive time-series bagger tracking.

        Args:
            ticker: Stock ticker symbol to analyze
            min_days: Minimum number of trading days required (default: 252 = ~1 year)
            debug: Print debug information

        Returns:
            BaggerResult if ticker can be analyzed, None otherwise
        """
        try:
            # Load ticker data
            ticker_data = self._load_ticker_data(ticker)
            if ticker_data is None:
                if debug:
                    print(f"DEBUG: {ticker} - No data file found")
                return None

            if len(ticker_data) < min_days:
                if debug:
                    print(f"DEBUG: {ticker} - Insufficient data: {len(ticker_data)} < {min_days} days")
                return None

            # Sort by date and clean data
            df = (ticker_data
                  .sort("date")
                  .with_columns([
                pl.col("date").str.to_date(),
                pl.col("adjusted_close").alias("price")
            ])
                  .filter(pl.col("price").is_not_null() & (pl.col("price") > 0)))

            if len(df) < min_days:
                if debug:
                    print(f"DEBUG: {ticker} - After filtering: {len(df)} < {min_days} days")
                return None

            # Calculate return multiples
            start_price = df["price"][0]
            if start_price <= 0:
                if debug:
                    print(f"DEBUG: {ticker} - Invalid start price: {start_price}")
                return None

            df = df.with_columns([
                (pl.col("price") / start_price).alias("return_multiple")
            ])

            # Add historical peak for fallen bagger classification
            df = df.with_columns([
                pl.col("return_multiple").cum_max().alias("peak_so_far")
            ])

            # Perform comprehensive analysis
            return self._perform_comprehensive_analysis(df, ticker)

        except Exception as e:
            if debug:
                print(f"ERROR analyzing {ticker}: {e}")
            return None

    def _perform_comprehensive_analysis(self, df: pl.DataFrame, ticker: str) -> BaggerResult:
        """Perform comprehensive time-series analysis of bagger status."""

        # Basic metrics
        start_price = df["price"][0]
        final_price = df["price"][-1]
        start_date = str(df["date"][0])
        final_date = str(df["date"][-1])
        total_days = len(df)

        # Find peak
        max_return_idx = df["return_multiple"].arg_max()
        max_return_multiple = df["return_multiple"][max_return_idx]
        max_price = df["price"][max_return_idx]
        max_date = str(df["date"][max_return_idx])
        days_to_peak = max_return_idx + 1

        current_return_multiple = df["return_multiple"][-1]
        current_peak = df["peak_so_far"][-1]
        current_bagger_type = self._classify_bagger(current_return_multiple, current_peak)

        # Track milestones
        milestones = self._find_milestones(df)

        # Track transitions with corrected logic
        transitions = self._track_transitions_fixed(df)

        # Calculate time in each status
        time_in_status = self._calculate_time_in_status_fixed(df)

        # Advanced metrics
        first_10x_date = self._find_first_milestone_date(df, 10.0)
        first_100x_date = self._find_first_milestone_date(df, 100.0)
        last_10x_date = self._find_last_milestone_date(df, 10.0)
        last_100x_date = self._find_last_milestone_date(df, 100.0)

        # Drawdown analysis
        max_drawdown_from_peak, max_drawdown_date = self._calculate_max_drawdown(df)

        # Current streak
        current_streak_type, current_streak_days, current_streak_start_date = self._analyze_current_streak_fixed(df)

        return BaggerResult(
            ticker=ticker,
            start_price=start_price,
            final_price=final_price,
            start_date=start_date,
            final_date=final_date,
            total_days=total_days,
            current_bagger_type=current_bagger_type,
            current_return_multiple=current_return_multiple,
            max_return_multiple=max_return_multiple,
            max_price=max_price,
            max_date=max_date,
            days_to_peak=days_to_peak,
            milestones=milestones,
            transitions=transitions,
            time_in_status=time_in_status,
            first_10x_date=first_10x_date,
            first_100x_date=first_100x_date,
            last_10x_date=last_10x_date,
            last_100x_date=last_100x_date,
            max_drawdown_from_peak=max_drawdown_from_peak,
            max_drawdown_date=max_drawdown_date,
            current_streak_type=current_streak_type,
            current_streak_days=current_streak_days,
            current_streak_start_date=current_streak_start_date
        )

    def _find_milestones(self, df: pl.DataFrame) -> List[BaggerMilestone]:
        """Find when stock hit specific return multiples - optimized version."""
        milestones = []
        milestone_levels = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

        for level in milestone_levels:
            # Find first time it hit this level using polars operations
            above_level = df.with_row_index("idx").filter(pl.col("return_multiple") >= level)

            if len(above_level) > 0:
                first_row = above_level.row(0, named=True)
                first_idx = first_row["idx"]  # row index

                # Count consecutive days above this level from first hit
                maintained_days = 0
                remaining_data = df.slice(first_idx)
                for val in remaining_data["return_multiple"]:
                    if val >= level:
                        maintained_days += 1
                    else:
                        break

                milestone = BaggerMilestone(
                    multiple=level,
                    date=str(first_row["date"]),  # date
                    price=first_row["price"],  # price
                    days_from_start=first_idx + 1,
                    maintained_for_days=maintained_days
                )
                milestones.append(milestone)

        return milestones

    def _track_transitions_fixed(self, df: pl.DataFrame) -> List[BaggerTransition]:
        """Track transitions between bagger states over time - fixed version."""
        transitions = []

        if len(df) == 0:
            return transitions

        # Add bagger status column using vectorized operations
        df_with_status = df.with_columns([
            pl.struct(["return_multiple", "peak_so_far"])
            .map_elements(
                lambda x: self._classify_bagger(x["return_multiple"], x["peak_so_far"]).value,
                return_dtype=pl.Utf8
            )
            .alias("bagger_status")
        ])

        # Find state changes efficiently
        df_with_changes = df_with_status.with_columns([
            (pl.col("bagger_status") != pl.col("bagger_status").shift(1)).alias("status_changed")
        ])

        # Get rows where status changed
        change_rows = df_with_changes.filter(pl.col("status_changed") | (pl.int_range(pl.len()) == 0))

        prev_status = None
        for i, row in enumerate(change_rows.iter_rows(named=True)):
            current_status = BaggerType(row["bagger_status"])

            if i == 0:
                prev_status = current_status
                continue

            # Create transition
            transition = BaggerTransition(
                from_status=prev_status,
                to_status=current_status,
                date=str(row["date"]),
                price=row["price"],
                return_multiple=row["return_multiple"],
                days_from_start=row["__index_level_0__"] if "__index_level_0__" in row else i + 1
            )
            transitions.append(transition)
            prev_status = current_status

        return transitions

    def _calculate_time_in_status_fixed(self, df: pl.DataFrame) -> Dict[BaggerType, int]:
        """Calculate total days spent in each bagger status - fixed version."""
        time_in_status = {status: 0 for status in BaggerType}

        # Vectorized status calculation
        df_with_status = df.with_columns([
            pl.struct(["return_multiple", "peak_so_far"])
            .map_elements(
                lambda x: self._classify_bagger(x["return_multiple"], x["peak_so_far"]).value,
                return_dtype=pl.Utf8
            )
            .alias("bagger_status")
        ])

        # Count days in each status
        status_counts = df_with_status.group_by("bagger_status").agg(pl.len().alias("days"))

        for row in status_counts.iter_rows(named=True):
            status = BaggerType(row["bagger_status"])
            time_in_status[status] = row["days"]

        return time_in_status

    def _analyze_current_streak_fixed(self, df: pl.DataFrame) -> Tuple[BaggerType, int, Optional[str]]:
        """Analyze the current streak of bagger status - fixed version."""
        if len(df) == 0:
            return BaggerType.NO_BAGGER, 0, None

        # Get current status
        current_return = df["return_multiple"][-1]
        current_peak = df["peak_so_far"][-1]
        current_status = self._classify_bagger(current_return, current_peak)

        # Add status column and work backwards to find streak start
        df_with_status = df.with_columns([
            pl.struct(["return_multiple", "peak_so_far"])
            .map_elements(
                lambda x: self._classify_bagger(x["return_multiple"], x["peak_so_far"]).value,
                return_dtype=pl.Utf8
            )
            .alias("bagger_status")
        ])

        # Find where current streak started
        streak_days = 1
        streak_start_date = str(df["date"][-1])

        for i in range(len(df_with_status) - 2, -1, -1):
            status = BaggerType(df_with_status["bagger_status"][i])
            if status == current_status:
                streak_days += 1
                streak_start_date = str(df["date"][i])
            else:
                break

        return current_status, streak_days, streak_start_date

    def _classify_bagger(self, return_multiple: float, peak_so_far: float) -> BaggerType:
        """Classify bagger type based on current return and historical peak."""
        # Current status
        if return_multiple >= 100:
            return BaggerType.HUNDRED_BAGGER
        elif return_multiple >= 10:
            return BaggerType.MULTIBAGGER
        else:
            # Check if it's a fallen bagger
            if peak_so_far >= 100:
                return BaggerType.FALLEN_HUNDRED_BAGGER
            elif peak_so_far >= 10:
                return BaggerType.FALLEN_MULTIBAGGER
            else:
                return BaggerType.NO_BAGGER

    def _find_first_milestone_date(self, df: pl.DataFrame, milestone: float) -> Optional[str]:
        """Find the first date when stock hit a specific return multiple."""
        above_milestone = df.filter(pl.col("return_multiple") >= milestone)
        if len(above_milestone) > 0:
            return str(above_milestone["date"][0])
        return None

    def _find_last_milestone_date(self, df: pl.DataFrame, milestone: float) -> Optional[str]:
        """Find the last date when stock was above a specific return multiple."""
        above_milestone = df.filter(pl.col("return_multiple") >= milestone)
        if len(above_milestone) > 0:
            return str(above_milestone["date"][-1])
        return None

    def _calculate_max_drawdown(self, df: pl.DataFrame) -> Tuple[float, Optional[str]]:
        """Calculate maximum drawdown from peak and when it occurred."""
        if len(df) == 0:
            return 0.0, None

        # Calculate drawdown efficiently using polars
        df_with_drawdown = df.with_columns([
            ((pl.col("peak_so_far") - pl.col("return_multiple")) / pl.col("peak_so_far")).alias("drawdown")
        ])

        max_drawdown_idx = df_with_drawdown["drawdown"].arg_max()
        max_drawdown = df_with_drawdown["drawdown"][max_drawdown_idx]
        max_drawdown_date = str(df_with_drawdown["date"][max_drawdown_idx])

        return max_drawdown, max_drawdown_date

    def _load_ticker_data(self, ticker: str) -> Optional[pl.DataFrame]:
        """Load data for a specific ticker from partitioned parquet files."""
        ticker_dir = self.partitioned_data_dir / f"code={ticker}"
        parquet_file = ticker_dir / "data.parquet"

        if not parquet_file.exists():
            return None

        try:
            return pl.read_parquet(parquet_file)
        except Exception:
            return None

    def get_available_tickers(self) -> List[str]:
        """Get list of all available tickers in the partitioned data."""
        tickers = []
        for partition_dir in self.partitioned_data_dir.iterdir():
            if partition_dir.is_dir() and partition_dir.name.startswith("code="):
                ticker = partition_dir.name.replace("code=", "")
                tickers.append(ticker)
        return sorted(tickers)


def save_results_to_parquet(results: List[BaggerResult],
                            output_file: str = "bagger_analysis_milestones.parquet"):
    """Save results to parquet with flattened structure."""
    if not results:
        print("No results to save")
        return

    # Flatten the data structure
    flattened_data = []

    for result in results:
        # Create base record
        base_record = {
            "ticker": result.ticker,
            "start_price": result.start_price,
            "final_price": result.final_price,
            "start_date": result.start_date,
            "final_date": result.final_date,
            "total_days": result.total_days,
            "current_bagger_type": result.current_bagger_type.value,
            "current_return_multiple": result.current_return_multiple,
            "max_return_multiple": result.max_return_multiple,
            "max_price": result.max_price,
            "max_date": result.max_date,
            "days_to_peak": result.days_to_peak,
            "first_10x_date": result.first_10x_date,
            "first_100x_date": result.first_100x_date,
            "last_10x_date": result.last_10x_date,
            "last_100x_date": result.last_100x_date,
            "max_drawdown_from_peak": result.max_drawdown_from_peak,
            "max_drawdown_date": result.max_drawdown_date,
            "current_streak_type": result.current_streak_type.value,
            "current_streak_days": result.current_streak_days,
            "current_streak_start_date": result.current_streak_start_date,

            # Time in status
            "days_as_no_bagger": result.time_in_status.get(BaggerType.NO_BAGGER, 0),
            "days_as_multibagger": result.time_in_status.get(BaggerType.MULTIBAGGER, 0),
            "days_as_hundred_bagger": result.time_in_status.get(BaggerType.HUNDRED_BAGGER, 0),
            "days_as_fallen_multibagger": result.time_in_status.get(BaggerType.FALLEN_MULTIBAGGER, 0),
            "days_as_fallen_hundred_bagger": result.time_in_status.get(BaggerType.FALLEN_HUNDRED_BAGGER, 0),

            # Milestone counts
            "milestones_hit": len(result.milestones),
            "transitions_count": len(result.transitions),

            # Key milestone dates
            "first_2x_date": next((m.date for m in result.milestones if m.multiple == 2), None),
            "first_5x_date": next((m.date for m in result.milestones if m.multiple == 5), None),

            # How long stayed above key levels
            "days_above_10x": next((m.maintained_for_days for m in result.milestones if m.multiple == 10), 0),
            "days_above_100x": next((m.maintained_for_days for m in result.milestones if m.multiple == 100), 0),
        }

        flattened_data.append(base_record)

    df = pl.DataFrame(flattened_data)
    df.write_parquet(output_file, compression='snappy')
    print(f"Bagger analysis saved to {output_file}")


def analyze_all_tickers(
        partitioned_data_dir: str = "stock_data_partitioned",
        min_days: int = 252,
        progress_interval: int = 100,
        debug=False
) -> List[BaggerResult]:
    """Analyze all tickers with bagger tracking."""

    analyzer = TickerBaggerAnalyzer(partitioned_data_dir)

    print("Discovering available tickers...")
    all_tickers = analyzer.get_available_tickers()
    print(f"Found {len(all_tickers)} tickers to analyze")

    results = []
    failed_count = 0

    print(f"Starting analysis (minimum {min_days} days required)...")

    for i, ticker in enumerate(all_tickers, 1):
        if i % progress_interval == 0:
            success_rate = ((i - failed_count) / i) * 100
            print(f"Processed {i:,}/{len(all_tickers):,} tickers... "
                  f"Success rate: {success_rate:.1f}% "
                  f"({len(results):,} successful, {failed_count:,} failed)")

        result = analyzer.analyze_ticker(ticker, min_days=min_days, debug=debug)
        if result:
            results.append(result)
        else:
            failed_count += 1

    print(f"\n✅ Analysis complete!")
    print(f"Successfully analyzed: {len(results):,} tickers")
    print(f"Failed to analyze: {failed_count:,} tickers")
    print(f"Success rate: {(len(results) / len(all_tickers)) * 100:.1f}%")

    return results


def print_quick_summary(results: List[BaggerResult]):
    """Print a quick summary of the analysis results."""

    if not results:
        print("No results to summarize")
        return

    print(f"\n{'=' * 60}")
    print(f"BAGGER ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total Analyzed: {len(results):,} tickers")

    # Current status distribution including fallen baggers
    status_counts = defaultdict(int)
    for result in results:
        status_counts[result.current_bagger_type] += 1

    print(f"\nCurrent Status Distribution:")
    for status in BaggerType:
        count = status_counts[status]
        pct = (count / len(results)) * 100
        print(f"  {status.value}: {count:,} ({pct:.1f}%)")

    # Peak achievements
    peak_10x = sum(1 for r in results if r.max_return_multiple >= 10)
    peak_100x = sum(1 for r in results if r.max_return_multiple >= 100)

    print(f"\nPeak Achievements:")
    print(f"  Ever reached 10x+: {peak_10x:,} ({(peak_10x / len(results)) * 100:.1f}%)")
    print(f"  Ever reached 100x+: {peak_100x:,} ({(peak_100x / len(results)) * 100:.1f}%)")

    # Current vs fallen analysis
    current_10x = sum(1 for r in results if r.current_return_multiple >= 10)
    current_100x = sum(1 for r in results if r.current_return_multiple >= 100)
    fallen_multi = status_counts[BaggerType.FALLEN_MULTIBAGGER]
    fallen_hundred = status_counts[BaggerType.FALLEN_HUNDRED_BAGGER]

    print(f"\nCurrent vs Fallen:")
    print(f"  Currently 10x+: {current_10x:,}")
    print(f"  Currently 100x+: {current_100x:,}")
    print(f"  Fallen from 10x+: {fallen_multi:,}")
    print(f"  Fallen from 100x+: {fallen_hundred:,}")

    # Top current performers
    current_baggers = [r for r in results if r.current_return_multiple >= 10]
    if current_baggers:
        top_current = sorted(current_baggers, key=lambda x: x.current_return_multiple, reverse=True)[:5]
        print(f"\nTop 5 Current Performers:")
        for r in top_current:
            print(f"  {r.ticker}: {r.current_return_multiple:.1f}x")


def main():
    """Main function to run the bagger analysis."""

    print("🚀 Starting Bagger Analysis")
    print("This will analyze the full journey of each stock's bagger status over time")
    print("Including fallen bagger detection and improved performance")

    # Configuration
    partitioned_data_dir = "stock_data_partitioned"
    min_days = 252  # Require at least 1 year of data
    output_file = "bagger_analysis_milestones.parquet"

    # Run analysis
    results = analyze_all_tickers(
        partitioned_data_dir=partitioned_data_dir,
        min_days=min_days,
    )

    if results:
        # Print quick summary
        print_quick_summary(results)

        # Save results
        save_results_to_parquet(results, output_file)

        print(f"\n🎉 Analysis complete!")
        print(f"Results saved to: {output_file}")
        print(f"\nYou can now analyze the results using:")
        print(f"  df = pl.read_parquet('{output_file}')")

    else:
        print("❌ No successful analyses. Check your data directory.")


if __name__ == "__main__":
    main()
