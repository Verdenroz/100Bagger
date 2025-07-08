import json
from pathlib import Path
from typing import List, Dict

import polars as pl


class BaggerExplorer:
    """Tool for exploring and analyzing enhanced bagger analysis results."""

    def __init__(self, data_dir: str = "."):
        """Initialize the explorer with data directory."""
        self.data_dir = Path(data_dir)
        self.results_df = None
        self.milestones_df = None
        self.transitions_df = None
        self.report = None
        self.load_data()

    def load_data(self):
        """Load all available data files."""
        try:
            # Load main results
            results_file = self.data_dir / "bagger_analysis_milestones.parquet"
            if results_file.exists():
                self.results_df = pl.read_parquet(results_file)
                print(f"✅ Loaded {len(self.results_df)} ticker results")

            # Load milestones
            milestones_file = self.data_dir / "milestones.parquet"
            if milestones_file.exists():
                self.milestones_df = pl.read_parquet(milestones_file)
                print(f"✅ Loaded {len(self.milestones_df)} milestone events")

            # Load transitions
            transitions_file = self.data_dir / "transitions.parquet"
            if transitions_file.exists():
                self.transitions_df = pl.read_parquet(transitions_file)
                print(f"✅ Loaded {len(self.transitions_df)} transition events")

            # Load comprehensive report
            report_file = self.data_dir / "comprehensive_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    self.report = json.load(f)
                print(f"✅ Loaded comprehensive report")

        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def get_ticker_journey(self, ticker: str) -> Dict:
        """Get detailed journey for a specific ticker."""
        if self.results_df is None:
            return {"error": "No results data loaded"}

        # Get main result
        ticker_result = self.results_df.filter(pl.col("ticker") == ticker)
        if len(ticker_result) == 0:
            return {"error": f"Ticker {ticker} not found"}

        result = ticker_result.to_dicts()[0]

        # Get milestones
        milestones = []
        if self.milestones_df is not None:
            ticker_milestones = self.milestones_df.filter(pl.col("ticker") == ticker)
            milestones = ticker_milestones.sort("days_from_start").to_dicts()

        # Get transitions
        transitions = []
        if self.transitions_df is not None:
            ticker_transitions = self.transitions_df.filter(pl.col("ticker") == ticker)
            transitions = ticker_transitions.sort("days_from_start").to_dicts()

        return {
            "ticker": ticker,
            "summary": result,
            "milestones": milestones,
            "transitions": transitions
        }

    def print_ticker_story(self, ticker: str):
        """Print a narrative story of a ticker's bagger journey."""
        journey = self.get_ticker_journey(ticker)

        if "error" in journey:
            print(f"❌ {journey['error']}")
            return

        summary = journey["summary"]
        milestones = journey["milestones"]
        transitions = journey["transitions"]

        print(f"\n{'=' * 60}")
        print(f"THE JOURNEY OF {ticker}")
        print(f"{'=' * 60}")

        print(f"📅 Period: {summary['start_date']} to {summary['final_date']}")
        print(f"💰 Price Journey: ${summary['start_price']:.2f} → ${summary['final_price']:.2f}")
        print(f"📈 Return: {summary['current_return_multiple']:.1f}x")
        print(f"🏔️  Peak: {summary['max_return_multiple']:.1f}x on {summary['max_date']}")
        print(f"📊 Current Status: {summary['current_bagger_type']}")

        if summary['max_drawdown_from_peak'] > 0:
            print(f"📉 Max Drawdown: {summary['max_drawdown_from_peak']:.1%}")

        print(f"\n🎯 MILESTONES ACHIEVED:")
        for milestone in milestones:
            days_to_reach = milestone['days_from_start']
            years_to_reach = days_to_reach / 252
            maintained_years = milestone['maintained_for_days'] / 252

            print(f"  {milestone['multiple']}x: Reached on {milestone['date']} "
                  f"({years_to_reach:.1f} years from start)")
            print(f"       Maintained for {maintained_years:.1f} years")

        print(f"\n🔄 STATUS TRANSITIONS:")
        for i, transition in enumerate(transitions):
            days = transition['days_from_start']
            years = days / 252

            if i == 0:
                print(f"  Start: {transition['from_status']} (${transition['price']:.2f})")

            print(f"  {transition['date']} ({years:.1f}y): {transition['from_status']} → {transition['to_status']}")
            print(f"       Price: ${transition['price']:.2f}, Return: {transition['return_multiple']:.1f}x")

        # Time distribution
        print(f"\n⏰ TIME DISTRIBUTION:")
        total_days = summary['total_days']
        print(f"  No Bagger: {summary['days_as_no_bagger']} days ({summary['days_as_no_bagger'] / total_days:.1%})")
        print(f"  Multibagger: {summary['days_as_multibagger']} days ({summary['days_as_multibagger'] / total_days:.1%})")
        print(f"  100-Bagger: {summary['days_as_hundred_bagger']} days ({summary['days_as_hundred_bagger'] / total_days:.1%})")

        # Current streak
        if summary['current_streak_days'] > 0:
            streak_years = summary['current_streak_days'] / 252
            print(f"\n🏃 CURRENT STREAK:")
            print(f"  {summary['current_streak_type']} for {streak_years:.1f} years")
            print(f"  Since: {summary['current_streak_start_date']}")

    def find_interesting_stories(self) -> Dict[str, List[str]]:
        """Find tickers with interesting bagger journeys."""
        if self.results_df is None:
            return {}

        stories = {
            "comeback_kings": [],
            "fallen_angels": [],
            "steady_climbers": [],
            "volatile_journeys": [],
            "recent_breakthroughs": []
        }

        for row in self.results_df.iter_rows(named=True):
            ticker = row["ticker"]

            # Comeback kings: fell from 100x+ but currently still multibagger
            if (row["max_return_multiple"] >= 100 and
                    row["current_return_multiple"] >= 10 and
                    row["current_return_multiple"] < 100):
                stories["comeback_kings"].append(ticker)

            # Fallen angels: peaked very high but now below 10x
            elif (row["max_return_multiple"] >= 50 and
                  row["current_return_multiple"] < 10):
                stories["fallen_angels"].append(ticker)

            # Steady climbers: few transitions but high returns
            elif (row["transitions_count"] <= 3 and
                  row["current_return_multiple"] >= 50):
                stories["steady_climbers"].append(ticker)

            # Volatile journeys: many transitions
            elif row["transitions_count"] >= 10:
                stories["volatile_journeys"].append(ticker)

            # Recent breakthroughs: became multibagger in last part of journey
            elif (row["current_return_multiple"] >= 10 and
                  row["current_streak_days"] / row["total_days"] > 0.3):
                stories["recent_breakthroughs"].append(ticker)

        # Limit to top examples
        for category in stories:
            stories[category] = stories[category][:10]

        return stories

    def print_interesting_stories(self):
        """Print examples of interesting bagger stories."""
        stories = self.find_interesting_stories()

        print(f"\n{'=' * 60}")
        print(f"INTERESTING BAGGER STORIES")
        print(f"{'=' * 60}")

        for category, tickers in stories.items():
            if tickers:
                category_name = category.replace("_", " ").title()
                print(f"\n🎭 {category_name}:")
                for ticker in tickers[:5]:  # Show top 5
                    print(f"  {ticker}")

                if len(tickers) > 5:
                    print(f"  ... and {len(tickers) - 5} more")

    def analyze_milestone_patterns(self):
        """Analyze patterns in milestone achievement."""
        if self.milestones_df is None:
            print("❌ No milestone data available")
            return

        print(f"\n{'=' * 60}")
        print(f"MILESTONE ACHIEVEMENT PATTERNS")
        print(f"{'=' * 60}")

        # Group by milestone level
        milestone_stats = (self.milestones_df
                           .group_by("multiple")
                           .agg([
            pl.count().alias("count"),
            pl.col("days_from_start").mean().alias("avg_days_to_reach"),
            pl.col("maintained_for_days").mean().alias("avg_maintained_days"),
            pl.col("days_from_start").min().alias("fastest_days"),
            pl.col("days_from_start").max().alias("slowest_days")
        ])
                           .sort("multiple"))

        print(f"\nMilestone Achievement Statistics:")
        print(f"{'Multiple':<8} {'Count':<6} {'Avg Days':<10} {'Fastest':<8} {'Slowest':<8} {'Maintained':<10}")
        print("-" * 60)

        for row in milestone_stats.iter_rows(named=True):
            multiple = f"{row['multiple']}x"
            count = row['count']
            avg_days = int(row['avg_days_to_reach'])
            fastest = int(row['fastest_days'])
            slowest = int(row['slowest_days'])
            maintained = int(row['avg_maintained_days']) if row['avg_maintained_days'] else 0

            print(f"{multiple:<8} {count:<6} {avg_days:<10} {fastest:<8} {slowest:<8} {maintained:<10}")

        # Find fastest to each milestone
        print(f"\n🏃 Fastest to Each Milestone:")
        for multiple in [10, 50, 100, 500, 1000]:
            fastest_to_milestone = (self.milestones_df
                                    .filter(pl.col("multiple") == multiple)
                                    .sort("days_from_start")
                                    .head(3))

            if len(fastest_to_milestone) > 0:
                print(f"\n{multiple}x milestone:")
                for row in fastest_to_milestone.iter_rows(named=True):
                    days = row['days_from_start']
                    years = days / 252
                    print(f"  {row['ticker']}: {days} days ({years:.1f} years) - {row['date']}")

    def create_summary_dashboard(self):
        """Create a summary dashboard of key metrics."""
        if self.results_df is None:
            print("❌ No results data available")
            return

        print(f"\n{'=' * 80}")
        print(f"ENHANCED BAGGER ANALYSIS DASHBOARD")
        print(f"{'=' * 80}")

        total_tickers = len(self.results_df)

        # Current status breakdown
        status_counts = (self.results_df
                         .group_by("current_bagger_type")
                         .agg(pl.count().alias("count"))
                         .sort("count", descending=True))

        print(f"\n📊 CURRENT STATUS DISTRIBUTION ({total_tickers:,} total tickers):")
        for row in status_counts.iter_rows(named=True):
            count = row['count']
            pct = (count / total_tickers) * 100
            status = row['current_bagger_type']
            print(f"  {status:<20}: {count:>6,} ({pct:>5.1f}%)")

        # Peak vs current comparison
        peak_10x = len(self.results_df.filter(pl.col("max_return_multiple") >= 10))
        current_10x = len(self.results_df.filter(pl.col("current_return_multiple") >= 10))
        peak_100x = len(self.results_df.filter(pl.col("max_return_multiple") >= 100))
        current_100x = len(self.results_df.filter(pl.col("current_return_multiple") >= 100))

        print(f"\n🏔️  PEAK vs CURRENT ACHIEVEMENTS:")
        print(f"  10x+ Baggers   - Peak: {peak_10x:,} | Current: {current_10x:,} | Fallen: {peak_10x - current_10x:,}")
        print(f"  100x+ Baggers  - Peak: {peak_100x:,} | Current: {current_100x:,} | Fallen: {peak_100x - current_100x:,}")

        # Top performers
        top_current = (self.results_df
                       .filter(pl.col("current_return_multiple") >= 10)
                       .sort("current_return_multiple", descending=True)
                       .head(10))

        if len(top_current) > 0:
            print(f"\n🏆 TOP 10 CURRENT PERFORMERS:")
            for i, row in enumerate(top_current.iter_rows(named=True), 1):
                ticker = row['ticker']
                current = row['current_return_multiple']
                peak = row['max_return_multiple']
                print(f"  {i:2d}. {ticker:<8}: {current:>8.1f}x (peak: {peak:.1f}x)")

        # Most volatile journeys
        most_volatile = (self.results_df
                         .sort("transitions_count", descending=True)
                         .head(10))

        print(f"\n🎢 MOST VOLATILE JOURNEYS (by transitions):")
        for i, row in enumerate(most_volatile.iter_rows(named=True), 1):
            ticker = row['ticker']
            transitions = row['transitions_count']
            current = row['current_return_multiple']
            peak = row['max_return_multiple']
            print(f"  {i:2d}. {ticker:<8}: {transitions:>2} transitions, {current:.1f}x current ({peak:.1f}x peak)")

        # Recent activity
        if self.transitions_df is not None:
            # Find recent transitions (within last year of data)
            recent_transitions = (self.transitions_df
                                  .with_columns(pl.col("date").str.to_date().alias("transition_date"))
                                  .filter(pl.col("transition_date") >= pl.col("transition_date").max() - pl.duration(days=365))
                                  .group_by("to_status")
                                  .agg(pl.count().alias("count"))
                                  .sort("count", descending=True))

            if len(recent_transitions) > 0:
                print(f"\n📈 RECENT TRANSITIONS (last year of data):")
                for row in recent_transitions.iter_rows(named=True):
                    status = row['to_status']
                    count = row['count']
                    print(f"  Became {status:<20}: {count:>4} tickers")

        # Complexity distribution
        complexity_stats = (self.results_df
        .select([
            pl.col("transitions_count").mean().alias("avg_transitions"),
            pl.col("transitions_count").median().alias("median_transitions"),
            pl.col("transitions_count").max().alias("max_transitions"),
            pl.col("milestones_hit").mean().alias("avg_milestones"),
            pl.col("milestones_hit").median().alias("median_milestones"),
            pl.col("milestones_hit").max().alias("max_milestones")
        ]).to_dicts()[0])

        print(f"\n🧩 JOURNEY COMPLEXITY:")
        print(f"  Transitions - Avg: {complexity_stats['avg_transitions']:.1f}, "
              f"Median: {complexity_stats['median_transitions']:.0f}, "
              f"Max: {complexity_stats['max_transitions']:.0f}")
        print(f"  Milestones  - Avg: {complexity_stats['avg_milestones']:.1f}, "
              f"Median: {complexity_stats['median_milestones']:.0f}, "
              f"Max: {complexity_stats['max_milestones']:.0f}")


def explore_specific_patterns(explorer: BaggerExplorer):
    """Explore specific interesting patterns in the data."""

    if explorer.results_df is None:
        print("❌ No data available for pattern exploration")
        return

    print(f"\n{'=' * 80}")
    print(f"SPECIFIC PATTERN ANALYSIS")
    print(f"{'=' * 80}")

    # Pattern 1: The Phoenix - fell from 100x+ but came back to 50x+
    phoenix_stocks = (explorer.results_df
                      .filter((pl.col("max_return_multiple") >= 100) &
                              (pl.col("current_return_multiple") >= 50) &
                              (pl.col("current_return_multiple") < 100))
                      .sort("current_return_multiple", descending=True))

    if len(phoenix_stocks) > 0:
        print(f"\n🔥 THE PHOENIX PATTERN (fell from 100x+ but still 50x+):")
        print(f"Found {len(phoenix_stocks)} examples:")
        for row in phoenix_stocks.head(5).iter_rows(named=True):
            ticker = row['ticker']
            peak = row['max_return_multiple']
            current = row['current_return_multiple']
            drawdown = (1 - current / peak) * 100
            print(f"  {ticker}: peaked at {peak:.0f}x, now {current:.0f}x ({drawdown:.0f}% drawdown)")

    # Pattern 2: The Steady Eddie - minimal transitions but high returns
    steady_eddies = (explorer.results_df
                     .filter((pl.col("transitions_count") <= 2) &
                             (pl.col("current_return_multiple") >= 20))
                     .sort("current_return_multiple", descending=True))

    if len(steady_eddies) > 0:
        print(f"\n📈 THE STEADY EDDIE PATTERN (≤2 transitions, 20x+ returns):")
        print(f"Found {len(steady_eddies)} examples:")
        for row in steady_eddies.head(5).iter_rows(named=True):
            ticker = row['ticker']
            current = row['current_return_multiple']
            transitions = row['transitions_count']
            years = row['total_days'] / 252
            print(f"  {ticker}: {current:.0f}x return with {transitions} transitions over {years:.1f} years")

    # Pattern 3: The Roller Coaster - high number of transitions
    roller_coasters = (explorer.results_df
                       .filter(pl.col("transitions_count") >= 8)
                       .sort("transitions_count", descending=True))

    if len(roller_coasters) > 0:
        print(f"\n🎢 THE ROLLER COASTER PATTERN (8+ transitions):")
        print(f"Found {len(roller_coasters)} examples:")
        for row in roller_coasters.head(5).iter_rows(named=True):
            ticker = row['ticker']
            transitions = row['transitions_count']
            current = row['current_return_multiple']
            peak = row['max_return_multiple']
            print(f"  {ticker}: {transitions} transitions, {current:.1f}x current ({peak:.1f}x peak)")

    # Pattern 4: The Late Bloomer - became multibagger recently (current streak > 50% of total time)
    late_bloomers = (explorer.results_df
                     .filter((pl.col("current_return_multiple") >= 10) &
                             (pl.col("current_streak_days") / pl.col("total_days") > 0.5))
                     .sort("current_return_multiple", descending=True))

    if len(late_bloomers) > 0:
        print(f"\n🌱 THE LATE BLOOMER PATTERN (recent multibagger breakthrough):")
        print(f"Found {len(late_bloomers)} examples:")
        for row in late_bloomers.head(5).iter_rows(named=True):
            ticker = row['ticker']
            current = row['current_return_multiple']
            streak_pct = (row['current_streak_days'] / row['total_days']) * 100
            print(f"  {ticker}: {current:.1f}x, current streak is {streak_pct:.0f}% of total time")


def main():
    """Main function to demonstrate the enhanced analysis explorer."""

    print("🔍 Enhanced Bagger Analysis Explorer")
    print("Loading and analyzing enhanced bagger data...")

    # Initialize explorer
    explorer = BaggerExplorer()

    if explorer.results_df is None:
        print("❌ No data found. Please run the enhanced analysis first.")
        return

    # Create main dashboard
    explorer.create_summary_dashboard()

    # Analyze milestone patterns
    explorer.analyze_milestone_patterns()

    # Find interesting stories
    explorer.print_interesting_stories()

    # Explore specific patterns
    explore_specific_patterns(explorer)

    print(f"\n{'=' * 80}")
    print(f"EXPLORATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nTo explore specific tickers, use:")
    print(f"  explorer.print_ticker_story('TICKER_SYMBOL')")
    print(f"\nExample interesting tickers to explore:")

    # Find some examples
    if len(explorer.results_df) > 0:
        # Get a few examples from different categories
        top_current = explorer.results_df.sort("current_return_multiple", descending=True).head(1)
        most_volatile = explorer.results_df.sort("transitions_count", descending=True).head(1)

        if len(top_current) > 0:
            print(f"  Top performer: {top_current['ticker'][0]}")
        if len(most_volatile) > 0:
            print(f"  Most volatile: {most_volatile['ticker'][0]}")


if __name__ == "__main__":
    main()