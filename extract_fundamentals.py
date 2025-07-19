import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

from eodhd.fundamentals_fetcher import FundamentalsFetcher
from eodhd.fundamentals_processor import FundamentalsProcessor

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fundamentals_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FundamentalsExtractor:
    """Main class for extracting fundamental data."""

    def __init__(self, api_token: str, max_concurrent: int = 50, batch_size: int = 1000):
        self.api_token = api_token
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.output_file = "fundamentals_data.parquet"
        self.progress_file = "extraction_progress.txt"
        self.summary_file = "fundamentals_summary.md"

        # State management
        self.processed_tickers = set()
        self.total_tickers = 0
        self.start_index = 0
        self.interrupted = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, finishing current batch...")
        self.interrupted = True

    def load_tickers(self, file_path: str) -> List[str]:
        """Load unique tickers from CSV file."""
        try:
            df = pd.read_csv(file_path)
            tickers = df['ticker'].tolist()
            logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
            return tickers
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return []

    def load_progress(self) -> int:
        """Load progress from previous run."""
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        start_index = int(content)
                        logger.info(f"Resuming from ticker index {start_index}")
                        return start_index
        except Exception as e:
            logger.error(f"Error loading progress: {e}")

        return 0

    def save_progress(self, index: int):
        """Save current progress."""
        try:
            with open(self.progress_file, 'w') as f:
                f.write(str(index))
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def progress_callback(self, completed: int, total: int):
        """Progress callback for batch processing."""
        global_completed = self.start_index + completed
        percentage = (global_completed / self.total_tickers) * 100
        logger.info(f"Progress: {global_completed}/{self.total_tickers} ({percentage:.1f}%) - Batch: {completed}/{total}")

    async def extract_batch(self, tickers: List[str], start_idx: int) -> bool:
        """
        Extract fundamental data for a batch of tickers.
        
        Returns:
            True if successful, False if interrupted
        """
        logger.info(f"Processing batch of {len(tickers)} tickers (starting at global index {start_idx})")

        async with FundamentalsFetcher(
                api_token=self.api_token,
                max_concurrent=self.max_concurrent,
                rate_limit_delay=0.1
        ) as fetcher:

            try:
                # Fetch data for the batch
                results = await fetcher.fetch_batch(tickers, self.progress_callback)

                if self.interrupted:
                    logger.info("Extraction interrupted, saving partial results...")
                    return False

                # Process and save results
                processor = FundamentalsProcessor()
                df = processor.process_batch(results)

                if not df.empty:
                    # Append to main file
                    processor.save_to_parquet(df, self.output_file, append=True)

                    # Update processed tickers
                    self.processed_tickers.update(tickers)

                    # Save progress
                    new_index = start_idx + len(tickers)
                    self.save_progress(new_index)

                    # Log statistics
                    stats = fetcher.get_stats()
                    logger.info(f"Batch completed - Success rate: {stats['success_rate']:.1f}%, "
                                f"Requests/sec: {stats['requests_per_second']:.1f}")

                    # Log any errors
                    if stats['errors']:
                        logger.warning(f"Batch had {len(stats['errors'])} errors")
                        for error in stats['errors'][:5]:  # Log first 5 errors
                            logger.warning(f"Error for {error['ticker']}: {error['error']}")

                return True

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                return False

    async def run(self, tickers_file: str, resume: bool = True):
        """
        Main extraction process.
        
        Args:
            tickers_file: Path to CSV file containing tickers
            resume: Whether to resume from previous progress
        """
        # Load tickers
        all_tickers = self.load_tickers(tickers_file)
        if not all_tickers:
            logger.error("No tickers loaded, exiting")
            return

        self.total_tickers = len(all_tickers)

        # Load progress if resuming
        if resume:
            self.start_index = self.load_progress()
        else:
            self.start_index = 0
            # Clear progress file
            if Path(self.progress_file).exists():
                Path(self.progress_file).unlink()

        # Get remaining tickers
        remaining_tickers = all_tickers[self.start_index:]
        logger.info(f"Processing {len(remaining_tickers)} remaining tickers "
                    f"(total: {self.total_tickers}, starting at: {self.start_index})")

        # Process in batches
        current_index = self.start_index

        for i in range(0, len(remaining_tickers), self.batch_size):
            if self.interrupted:
                logger.info("Extraction interrupted by user")
                break

            batch_tickers = remaining_tickers[i:i + self.batch_size]
            batch_start_idx = current_index + i

            logger.info(f"Starting batch {i // self.batch_size + 1} of "
                        f"{(len(remaining_tickers) + self.batch_size - 1) // self.batch_size}")

            success = await self.extract_batch(batch_tickers, batch_start_idx)

            if not success:
                logger.error("Batch processing failed or was interrupted")
                break

        # Generate final summary
        self.generate_summary()

        # Clean up progress file if completed
        if current_index + len(remaining_tickers) >= self.total_tickers and not self.interrupted:
            if Path(self.progress_file).exists():
                Path(self.progress_file).unlink()
            logger.info("Extraction completed successfully!")

    def generate_summary(self):
        """Generate summary report of extraction."""
        try:
            # Load data
            processor = FundamentalsProcessor()
            df = processor.load_from_parquet(self.output_file)

            if df is not None and not df.empty:
                # Create summary report
                processor.create_summary_report(df, self.summary_file)

                # Log key statistics
                stats = processor.get_summary_stats(df)
                logger.info(f"Summary: {stats['total_records']} total records, "
                            f"{stats['successful_extractions']} successful, "
                            f"{stats['failed_extractions']} failed")
                logger.info(f"Summary report saved to {self.summary_file}")
            else:
                logger.warning("No data found for summary generation")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract fundamental data for tickers")
    parser.add_argument("--tickers-file", default="unique_tickers.csv",
                        help="CSV file containing tickers (default: unique_tickers.csv)")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Maximum concurrent requests (default: 50)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for processing (default: 1000)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start from beginning, ignore previous progress")
    parser.add_argument("--api-token",
                        help="EODHD API token (defaults to EODHD_API_TOKEN env var)")

    args = parser.parse_args()

    # Get API token
    api_token = args.api_token or os.environ.get("EODHD_API_TOKEN")
    if not api_token:
        logger.error("API token not provided. Set EODHD_API_TOKEN environment variable or use --api-token")
        sys.exit(1)

    # Validate input file
    if not Path(args.tickers_file).exists():
        logger.error(f"Tickers file not found: {args.tickers_file}")
        sys.exit(1)

    # Create extractor
    extractor = FundamentalsExtractor(
        api_token=api_token,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size
    )

    # Run extraction
    logger.info(f"Starting fundamental data extraction")
    logger.info(f"Config: max_concurrent={args.max_concurrent}, batch_size={args.batch_size}")

    try:
        asyncio.run(extractor.run(args.tickers_file, resume=not args.no_resume))
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
