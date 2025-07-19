import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load bagger analysis and fundamental data."""
    
    # Load bagger analysis results
    bagger_file = "bagger_analysis_results.csv"
    if not Path(bagger_file).exists():
        logger.error(f"Bagger analysis file not found: {bagger_file}")
        return None, None
    
    logger.info(f"Loading bagger analysis from {bagger_file}")
    bagger_df = pd.read_csv(bagger_file)
    logger.info(f"Loaded {len(bagger_df)} bagger analysis records")
    
    # Load fundamental data
    fundamental_file = "fundamentals_data.parquet"
    if not Path(fundamental_file).exists():
        logger.error(f"Fundamental data file not found: {fundamental_file}")
        return bagger_df, None
    
    logger.info(f"Loading fundamental data from {fundamental_file}")
    fundamental_df = pd.read_parquet(fundamental_file)
    logger.info(f"Loaded {len(fundamental_df)} fundamental data records")
    
    return bagger_df, fundamental_df

def prepare_fundamental_data(fundamental_df):
    """Prepare fundamental data for merging."""
    
    # Filter only successful extractions
    successful_df = fundamental_df[fundamental_df['status'] == 'success'].copy()
    logger.info(f"Using {len(successful_df)} successful fundamental extractions")
    
    # Select relevant columns for merging
    merge_columns = [
        'original_ticker',
        'type',
        'sector',
        'industry', 
        'gic_sector',
        'gic_group',
        'gic_industry',
        'gic_sub_industry'
    ]
    
    # Ensure all columns exist
    for col in merge_columns:
        if col not in successful_df.columns:
            successful_df[col] = None
            logger.warning(f"Column {col} not found in fundamental data, setting to None")
    
    # Rename original_ticker to ticker for merging
    merge_df = successful_df[merge_columns].copy()
    merge_df = merge_df.rename(columns={'original_ticker': 'ticker'})
    
    # Remove duplicates (keep first occurrence)
    initial_count = len(merge_df)
    merge_df = merge_df.drop_duplicates(subset=['ticker'], keep='first')
    if len(merge_df) < initial_count:
        logger.warning(f"Removed {initial_count - len(merge_df)} duplicate tickers from fundamental data")
    
    logger.info(f"Prepared {len(merge_df)} fundamental records for merging")
    
    return merge_df

def merge_data(bagger_df, fundamental_df):
    """Merge bagger analysis with fundamental data."""
    
    logger.info("Starting merge process...")
    
    # Prepare fundamental data
    prepared_fundamental = prepare_fundamental_data(fundamental_df)
    
    # Perform left join (keep all bagger records)
    merged_df = bagger_df.merge(
        prepared_fundamental,
        on='ticker',
        how='left',
        suffixes=('', '_fund')
    )
    
    logger.info(f"Merged dataset contains {len(merged_df)} records")
    
    # Reorder columns: ticker, bagger_type, type, fundamental fields, then original ordering
    fundamental_columns = ['type', 'sector', 'industry', 'gic_sector', 'gic_group', 'gic_industry', 'gic_sub_industry']
    base_columns = ['ticker', 'bagger_type']
    
    # Get remaining original columns (excluding ticker, bagger_type, and fundamental columns)
    remaining_columns = [col for col in merged_df.columns 
                        if col not in base_columns + fundamental_columns]
    
    # Create new column order
    new_column_order = base_columns + fundamental_columns + remaining_columns
    
    # Reorder the dataframe
    merged_df = merged_df[new_column_order]
    
    # Apply Common Stock logic - only populate sector/industry fields for Common Stock
    sector_industry_columns = ['sector', 'industry', 'gic_sector', 'gic_group', 'gic_industry', 'gic_sub_industry']
    
    for col in sector_industry_columns:
        if col in merged_df.columns:
            # Set to None where type is not 'Common Stock'
            merged_df.loc[merged_df['type'] != 'Common Stock', col] = None
    
    # Count successful matches
    successful_matches = merged_df['type'].notna().sum()
    common_stock_matches = (merged_df['type'] == 'Common Stock').sum()
    
    logger.info(f"Merge statistics:")
    logger.info(f"  Total records: {len(merged_df)}")
    logger.info(f"  Successful fundamental matches: {successful_matches}")
    logger.info(f"  Common Stock matches: {common_stock_matches}")
    logger.info(f"  ETF matches: {(merged_df['type'] == 'ETF').sum()}")
    logger.info(f"  Fund matches: {(merged_df['type'] == 'Fund').sum()}")
    logger.info(f"  Other type matches: {(merged_df['type'].notna() & ~merged_df['type'].isin(['Common Stock', 'ETF', 'Fund'])).sum()}")
    logger.info(f"  No fundamental data: {merged_df['type'].isna().sum()}")
    
    return merged_df

def analyze_merged_data(merged_df):
    """Analyze the merged dataset."""
    
    logger.info("\nDataset Analysis:")
    logger.info("=" * 50)
    
    # Type distribution
    type_counts = merged_df['type'].value_counts(dropna=False)
    logger.info(f"Security Type Distribution:")
    for security_type, count in type_counts.items():
        percentage = (count / len(merged_df)) * 100
        logger.info(f"  {security_type}: {count} ({percentage:.1f}%)")
    
    # Sector distribution for Common Stocks only
    common_stocks = merged_df[merged_df['type'] == 'Common Stock']
    if len(common_stocks) > 0:
        logger.info(f"\nSector Distribution (Common Stocks only):")
        sector_counts = common_stocks['sector'].value_counts(dropna=False).head(10)
        for sector, count in sector_counts.items():
            percentage = (count / len(common_stocks)) * 100
            logger.info(f"  {sector}: {count} ({percentage:.1f}%)")
    
    # Bagger type vs security type analysis
    logger.info(f"\nBagger Type vs Security Type:")
    crosstab = pd.crosstab(
        merged_df['bagger_type'], 
        merged_df['type'].fillna('No Data'), 
        margins=True
    )
    logger.info(f"\n{crosstab}")

def save_results(merged_df):
    """Save merged results to CSV and parquet."""
    
    output_base = "fundamental_bagger_analysis_results"
    
    # Save to CSV
    csv_file = f"{output_base}.csv"
    merged_df.to_csv(csv_file, index=False)
    logger.info(f"Saved merged data to {csv_file}")
    
    # Save to parquet
    parquet_file = f"{output_base}.parquet"
    merged_df.to_parquet(parquet_file, index=False)
    logger.info(f"Saved merged data to {parquet_file}")
    
    # Create summary file
    summary_file = f"{output_base}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Fundamental Bagger Analysis Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Records: {len(merged_df):,}\n")
        f.write(f"Columns: {len(merged_df.columns)}\n\n")
        
        f.write("Security Type Distribution:\n")
        type_counts = merged_df['type'].value_counts(dropna=False)
        for security_type, count in type_counts.items():
            percentage = (count / len(merged_df)) * 100
            f.write(f"  {security_type}: {count:,} ({percentage:.1f}%)\n")
        
        f.write("\nColumns with sector/industry data (Common Stock only):\n")
        f.write("  - sector\n")
        f.write("  - industry\n")
        f.write("  - gic_sector\n")
        f.write("  - gic_group\n")
        f.write("  - gic_industry\n")
        f.write("  - gic_sub_industry\n")
        f.write("\nNote: Sector/industry fields are null for non-Common Stock securities.\n")
    
    logger.info(f"Saved summary to {summary_file}")

def main():
    """Main function."""
    
    logger.info("Starting fundamental data and bagger analysis merger")
    
    # Load data
    bagger_df, fundamental_df = load_data()
    
    if bagger_df is None:
        logger.error("Failed to load bagger analysis data")
        return
    
    if fundamental_df is None:
        logger.error("Failed to load fundamental data")
        return
    
    # Merge data
    merged_df = merge_data(bagger_df, fundamental_df)
    
    # Analyze results
    analyze_merged_data(merged_df)
    
    # Save results
    save_results(merged_df)
    
    logger.info("Merger completed successfully!")

if __name__ == "__main__":
    main()