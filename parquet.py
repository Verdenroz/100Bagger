import traceback
from pathlib import Path
from typing import Optional

import polars as pl


class PartitionedParquetConverter:
    """Handles conversion of large CSV files to partitioned Parquet format."""

    def __init__(self, csv_file: str, output_dir: str = "stock_data_partitioned"):
        """Initialize the converter with input and output paths.

        Args:
            csv_file: Path to the input CSV file
            output_dir: Directory for partitioned output (default: "stock_data_partitioned")
        """
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.schema_overrides = {
            'code': pl.Utf8,
            'exchange_short_name': pl.Utf8,
            'date': pl.Utf8,
            'open': pl.Float64,
            'high': pl.Float64,
            'low': pl.Float64,
            'close': pl.Float64,
            'adjusted_close': pl.Float64,
            'volume': pl.Float64,
        }

    def convert_to_partitioned_parquet(self, batch_size: int = 5_000_000) -> bool:
        """Convert CSV to partitioned Parquet files by stock code.

        Args:
            batch_size: Number of rows to process per batch (default: 5M)

        Returns:
            True if conversion succeeded, False otherwise
        """
        print(f"Converting {self.csv_file} to partitioned format")

        try:
            Path(self.output_dir).mkdir(exist_ok=True)

            print("Using streaming batch processing...")
            lazy_df = pl.scan_csv(
                self.csv_file,
                schema_overrides=self.schema_overrides,
                infer_schema_length=0
            )

            # Get total row count for progress tracking
            total_rows = lazy_df.select(pl.len()).collect(engine="streaming").item()
            print(f"Total rows to process: {total_rows:,}")

            processed_rows = 0
            batch_num = 0

            while processed_rows < total_rows:
                batch_num += 1
                current_batch_size = min(batch_size, total_rows - processed_rows)

                print(f"Processing batch {batch_num}: rows {processed_rows:,} "
                      f"to {processed_rows + current_batch_size:,}")

                # Process current batch using streaming
                batch_df = (lazy_df
                            .slice(processed_rows, current_batch_size)
                            .collect(engine="streaming"))

                if len(batch_df) == 0:
                    break

                self._process_batch_partitions(batch_df, batch_num)

                processed_rows += len(batch_df)
                progress = (processed_rows / total_rows) * 100
                print(f"  Progress: {progress:.1f}% ({processed_rows:,}/{total_rows:,} rows)")

                # Clear memory
                del batch_df

            print("✅ Partitioned conversion completed!")
            return True

        except Exception as e:
            print(f"❌ Error during partitioned conversion: {e}")
            traceback.print_exc()
            return False

    def _process_batch_partitions(self, batch_df: pl.DataFrame, batch_num: int) -> None:
        """Process partitions for a single batch."""
        unique_codes = batch_df['code'].unique().to_list()
        print(f"  Found {len(unique_codes)} unique codes in batch")

        for code in unique_codes:
            code_data = batch_df.filter(pl.col('code') == code)
            partition_dir = Path(self.output_dir) / f"code={code}"
            partition_dir.mkdir(exist_ok=True)

            parquet_file = partition_dir / "data.parquet"

            if parquet_file.exists():
                # Append to existing file
                existing_data = pl.read_parquet(parquet_file)
                combined_data = pl.concat([existing_data, code_data])
                combined_data.write_parquet(parquet_file, compression='snappy')
            else:
                # Create new file
                code_data.write_parquet(parquet_file, compression='snappy')

    def verify_partitioned_data(self, sample_symbol: Optional[str] = None) -> None:
        """Verify the converted partitioned Parquet data.

        Args:
            sample_symbol: Optional symbol to show sample data for
        """
        print(f"\n--- Verifying Partitioned Data ---")

        try:
            df = pl.scan_parquet(f"{self.output_dir}/**/data.parquet")
            total_rows = df.select(pl.len()).collect().item()
            print(f"Total rows in Parquet: {total_rows:,}")

            # Count partitions
            output_path = Path(self.output_dir)
            partition_dirs = [d for d in output_path.iterdir()
                              if d.is_dir() and d.name.startswith('code=')]
            print(f"Number of partitions: {len(partition_dirs)}")

            # Show sample partition names
            sample_partitions = sorted([d.name for d in partition_dirs])
            if len(partition_dirs) <= 10:
                for partition in sample_partitions:
                    print(f"  - {partition}")
            else:
                for partition in sample_partitions[:5]:
                    print(f"  - {partition}")
                print(f"  ... and {len(partition_dirs) - 5} more partitions")

            # Show sample data if requested
            if sample_symbol:
                print(f"\nSample data for {sample_symbol}:")
                sample_data = (df
                               .filter(pl.col('code') == sample_symbol)
                               .head(5)
                               .collect())
                print(sample_data)

        except Exception as e:
            print(f"❌ Error during verification: {e}")


def main() -> None:
    """Main function to run the partitioned conversion process."""
    # Configuration
    csv_file = "stock_data.csv"
    output_dir = "stock_data_partitioned"

    converter = PartitionedParquetConverter(csv_file, output_dir)

    print("Converting CSV to partitioned Parquet format...")
    success = converter.convert_to_partitioned_parquet()

    if success:
        converter.verify_partitioned_data(sample_symbol="AADBX")
    else:
        print("\n❌ Conversion failed. Check system resources and file paths.")


if __name__ == "__main__":
    main()
