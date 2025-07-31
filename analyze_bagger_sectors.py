import logging
from pathlib import Path
from typing import Optional

import polars as pl

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_bagger_data() -> Optional[pl.DataFrame]:
    """Load the fundamental bagger analysis results."""
    data_file = Path("fundamental_bagger_analysis_results.csv")

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return None

    try:
        df = pl.read_csv(data_file)
        logger.info(f"Loaded {len(df)} records from {data_file}")
        logger.info(f"Columns: {df.columns}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def analyze_bagger_type_distribution(df: pl.DataFrame) -> None:
    """Analyze the distribution of bagger types."""
    logger.info("\n" + "=" * 60)
    logger.info("BAGGER TYPE DISTRIBUTION")
    logger.info("=" * 60)

    bagger_counts = df.group_by("bagger_type").agg([
        pl.len().alias("count"),
        pl.col("max_return_multiple").mean().alias("avg_max_return"),
        pl.col("final_return_multiple").mean().alias("avg_final_return"),
        pl.col("total_days").mean().alias("avg_days"),
        pl.col("days_to_peak").mean().alias("avg_days_to_peak")
    ]).sort("count", descending=True)

    print("\nBagger Type Summary:")
    print(bagger_counts)

    # Calculate percentages
    total = len(df)
    bagger_pct = bagger_counts.with_columns([
        (pl.col("count") / total * 100).round(2).alias("percentage")
    ])

    print(f"\nBagger Type Distribution (Total: {total:,}):")
    print(bagger_pct.select(["bagger_type", "count", "percentage"]))


def analyze_classification_distribution(df: pl.DataFrame, classification_col: str, title: str, top_n: int = 20) -> None:
    """Generic function to analyze any classification column."""
    logger.info(f"\n{title}")
    logger.info("=" * len(title))

    # Overall distribution
    valid_data = df.filter(pl.col(classification_col).is_not_null())
    total_valid = len(valid_data)

    if total_valid == 0:
        print(f"No data available for {classification_col}")
        return

    print(f"\nTotal records with {classification_col}: {total_valid:,}")

    # Top classifications by count
    classification_counts = valid_data.group_by(classification_col).agg([
        pl.len().alias("count"),
        pl.col("max_return_multiple").mean().alias("avg_max_return"),
        pl.col("final_return_multiple").mean().alias("avg_final_return"),
        pl.col("total_days").mean().alias("avg_total_days"),
        pl.col("days_to_peak").mean().alias("avg_days_to_peak")
    ]).sort("count", descending=True)

    print(f"\nTop {top_n} {classification_col} by Count:")
    print(classification_counts.head(top_n))

    # Top performing classifications (min 3 baggers for statistical significance)
    top_performers = classification_counts.filter(pl.col("count") >= 3).sort("avg_max_return", descending=True)

    print(f"\nTop 10 {classification_col} by Average Max Return (min 3 baggers):")
    print(top_performers.head(10))

    # Distribution by bagger type
    print(f"\n{classification_col} Distribution by Bagger Type:")
    type_distribution = valid_data.group_by(["bagger_type", classification_col]).agg([
        pl.len().alias("count")
    ]).sort(["bagger_type", "count"], descending=[False, True])

    # Show top 5 for each bagger type
    for bagger_type in df["bagger_type"].unique().sort():
        if bagger_type is not None:
            type_data = type_distribution.filter(pl.col("bagger_type") == bagger_type).head(5)
            if len(type_data) > 0:
                print(f"\nTop 5 {classification_col} for {bagger_type}:")
                print(type_data.select([classification_col, "count"]))


def generate_data_quality_report(df: pl.DataFrame) -> None:
    """Generate a data quality report for all classification columns."""
    logger.info("\n" + "=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    total_records = len(df)
    classification_cols = ["sector", "industry", "gic_sector", "gic_group", "gic_industry", "gic_sub_industry"]

    print(f"Total Records: {total_records:,}")
    print("\nData Completeness by Classification:")

    quality_data = []
    for col in classification_cols:
        non_null_count = len(df.filter(pl.col(col).is_not_null()))
        completeness_pct = (non_null_count / total_records) * 100
        unique_count = df.select(pl.col(col).n_unique()).item()

        quality_data.append({
            "classification": col,
            "total_records": total_records,
            "non_null_records": non_null_count,
            "completeness_pct": round(completeness_pct, 2),
            "unique_values": unique_count
        })

    quality_df = pl.DataFrame(quality_data)
    print(quality_df)


def save_bagger_type_analysis_results(df: pl.DataFrame) -> None:
    """Save analysis results focused on bagger types."""
    logger.info("\nSaving bagger type analysis results...")

    classifications = ["sector", "industry", "gic_sector", "gic_group", "gic_industry", "gic_sub_industry"]

    for classification in classifications:
        # Create detailed analysis for each classification by bagger type
        valid_data = df.filter(pl.col(classification).is_not_null())

        if len(valid_data) > 0:
            # By bagger type analysis
            by_type_analysis = valid_data.group_by(["bagger_type", classification]).agg([
                pl.len().alias("count"),
                pl.col("max_return_multiple").mean().alias("avg_max_return"),
                pl.col("max_return_multiple").median().alias("median_max_return"),
                pl.col("final_return_multiple").mean().alias("avg_final_return"),
                pl.col("final_return_multiple").median().alias("median_final_return"),
                pl.col("total_days").mean().alias("avg_total_days"),
                pl.col("days_to_peak").mean().alias("avg_days_to_peak")
            ]).sort(["bagger_type", "count"], descending=[False, True])

            filename_by_type = f"{classification}_by_bagger_type_analysis.csv"
            by_type_analysis.write_csv(filename_by_type)
            logger.info(f"Saved {classification} by bagger type analysis to {filename_by_type}")


def generate_executive_summary(df: pl.DataFrame) -> None:
    """Generate an executive summary of key findings."""
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTIVE SUMMARY")
    logger.info("=" * 60)

    total_baggers = len(df)

    print(f"Total Baggers Analyzed: {total_baggers:,}")

    # Data coverage summary
    classifications = ["sector", "industry", "gic_sector", "gic_group", "gic_industry", "gic_sub_industry"]
    print("\nData Coverage:")
    for classification in classifications:
        coverage = len(df.filter(pl.col(classification).is_not_null()))
        pct = (coverage / total_baggers) * 100
        print(f"  {classification}: {coverage:,} ({pct:.1f}%)")

    # Key performance metrics
    print("\n--- Overall Performance Metrics ---")
    overall_stats = df.select([
        pl.col("max_return_multiple").mean().alias("avg_max_return"),
        pl.col("max_return_multiple").median().alias("median_max_return"),
        pl.col("final_return_multiple").mean().alias("avg_final_return"),
        pl.col("total_days").mean().alias("avg_total_days"),
        pl.col("days_to_peak").mean().alias("avg_days_to_peak")
    ])
    print(overall_stats)

    # Top performing classifications
    print("\n--- Top Performing Classifications (by avg max return, min 5 baggers) ---")

    for classification in ["sector", "gic_sector", "gic_industry"]:
        top_performer = (df
                         .filter(pl.col(classification).is_not_null())
                         .group_by(classification)
                         .agg([
            pl.len().alias("count"),
            pl.col("max_return_multiple").mean().alias("avg_max_return")
        ])
                         .filter(pl.col("count") >= 5)
                         .sort("avg_max_return", descending=True)
                         .head(1)
                         )

        if len(top_performer) > 0:
            name = top_performer.row(0)[0]
            count = top_performer.row(0)[1]
            avg_return = top_performer.row(0)[2]
            print(f"  Top {classification}: {name} ({count} baggers, {avg_return:.1f}x avg return)")


def main():
    """Main analysis function - focused on bagger type analysis."""
    logger.info("Starting bagger sector and industry analysis by bagger types")

    # Load data
    df = load_bagger_data()
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return

    # Bagger type distribution
    analyze_bagger_type_distribution(df)

    # Analyze each classification system by bagger type
    classifications_config = [
        ("sector", "TRADITIONAL SECTOR ANALYSIS BY BAGGER TYPE", 15),
        ("industry", "TRADITIONAL INDUSTRY ANALYSIS BY BAGGER TYPE", 25),
        ("gic_sector", "GICS SECTOR ANALYSIS BY BAGGER TYPE", 15),
        ("gic_group", "GICS INDUSTRY GROUP ANALYSIS BY BAGGER TYPE", 20),
        ("gic_industry", "GICS INDUSTRY ANALYSIS BY BAGGER TYPE", 25),
        ("gic_sub_industry", "GICS SUB-INDUSTRY ANALYSIS BY BAGGER TYPE", 30)
    ]

    for col, title, top_n in classifications_config:
        analyze_classification_distribution(df, col, title, top_n)

    # Data quality and summary focused on bagger types
    generate_data_quality_report(df)
    generate_executive_summary(df)

    # Save results focused on bagger type analysis
    save_bagger_type_analysis_results(df)

    logger.info("Bagger type analysis completed successfully!")


if __name__ == "__main__":
    main()
