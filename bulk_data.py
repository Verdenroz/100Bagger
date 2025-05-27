import asyncio
import datetime
import os
from pathlib import Path
import sqlite3

import aiohttp
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_token = os.environ.get("EODHD_API_TOKEN")


class StockDataManager:
    def __init__(self, db_path="stock_data.db", csv_path="us_stocks_bulk_data.csv"):
        self.db_path = db_path
        self.csv_path = csv_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with proper schema."""
        print("Initializing SQLite database...")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS stock_data
                         (
                             code
                             TEXT,
                             exchange_short_name
                             TEXT,
                             date
                             TEXT,
                             open
                             REAL,
                             high
                             REAL,
                             low
                             REAL,
                             close
                             REAL,
                             adjusted_close
                             REAL,
                             volume
                             INTEGER,
                             PRIMARY
                             KEY
                         (
                             code,
                             date
                         )
                             )
                         """)

            # Create index for faster queries
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_date
                             ON stock_data(date)
                         """)

    def get_latest_date(self):
        """Get the latest date in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(date) FROM stock_data")
            result = cursor.fetchone()[0]
            return result

    def get_existing_dates(self):
        """Get all existing dates to avoid duplicates."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT date FROM stock_data ORDER BY date")
            return {row[0] for row in cursor.fetchall()}

    def save_batch_to_db(self, batch_data):
        """Save batch data to SQLite database efficiently."""
        if not batch_data:
            return 0

        # Prepare data for insertion
        records = []
        for entry in batch_data:
            records.append((
                entry.get('code', ''),
                entry.get('exchange_short_name', ''),
                entry.get('date', ''),
                entry.get('open'),
                entry.get('high'),
                entry.get('low'),
                entry.get('close'),
                entry.get('adjusted_close'),
                entry.get('volume')
            ))

        with sqlite3.connect(self.db_path) as conn:
            # Use INSERT OR REPLACE to handle duplicates
            conn.executemany("""
                INSERT OR REPLACE INTO stock_data 
                (code, exchange_short_name, date, open, high, low, close, adjusted_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

        return len(records)

    def import_existing_csv(self, csv_file_path, chunk_size=50000):
        """Import existing CSV data into the database."""
        if not Path(csv_file_path).exists():
            print(f"CSV file {csv_file_path} not found")
            return 0

        print(f"Importing existing CSV: {csv_file_path}")

        # Read CSV in chunks to manage memory
        total_imported = 0
        chunk_count = 0

        for chunk_df in pd.read_csv(csv_file_path, chunksize=chunk_size):
            chunk_count += 1
            print(f"Processing chunk {chunk_count} ({len(chunk_df)} records)...")

            # Convert DataFrame to list of dictionaries
            batch_data = chunk_df.to_dict('records')

            # Save to database
            imported = self.save_batch_to_db(batch_data)
            total_imported += imported

            print(f"Imported {imported} records from chunk {chunk_count}")

        print(f"CSV import completed. Total records imported: {total_imported}")
        return total_imported

    def export_to_csv_incremental(self, chunk_size=50000):
        """Export database to CSV in chunks to manage memory."""
        print("Exporting database to CSV...")

        with sqlite3.connect(self.db_path) as conn:
            # Get total count
            total_count = conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
            print(f"Total records to export: {total_count}")

            # Export in chunks
            offset = 0
            first_chunk = True

            while offset < total_count:
                df = pd.read_sql_query(
                    f"""SELECT * FROM stock_data 
                        ORDER BY date, code 
                        LIMIT {chunk_size} OFFSET {offset}""",
                    conn
                )

                if len(df) == 0:
                    break

                # Write to CSV
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                df.to_csv(self.csv_path, mode=mode, header=header, index=False)

                offset += chunk_size
                first_chunk = False
                print(f"Exported {min(offset, total_count)}/{total_count} records")

        print(f"CSV export completed: {self.csv_path}")


async def fetch_day_data(session, date_str, semaphore, existing_dates):
    """Fetch bulk data for a specific date using aiohttp."""
    # Skip if we already have this date
    if date_str in existing_dates:
        print(f"Skipping {date_str} (already exists)")
        return []

    url = f"https://eodhd.com/api/eod-bulk-last-day/US?api_token={api_token}&date={date_str}&fmt=json"
    print(f"Requesting data for {date_str} ...")

    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, list):
                            for entry in data:
                                entry['date'] = date_str
                            print(f"Data received successfully. Total entries: {len(data)}")
                            return data
                        else:
                            day_entries = data.get("data", [])
                            for entry in day_entries:
                                entry['date'] = date_str
                            return day_entries
                    except Exception as e:
                        print(f"Error parsing JSON for {date_str}: {e}")
                else:
                    print(f"Failed for {date_str}: HTTP {response.status}")
        except Exception as e:
            print(f"Request exception for {date_str}: {e}")
    return []


async def process_date_batch(session, date_batch, semaphore, data_manager, existing_dates):
    """Process a batch of dates and save to database."""
    print(f"\nProcessing batch of {len(date_batch)} dates...")

    # Fetch data for this batch of dates
    tasks = [fetch_day_data(session, date_str, semaphore, existing_dates) for date_str in date_batch]
    results = await asyncio.gather(*tasks)

    # Flatten the results for this batch
    batch_data = []
    for day_data in results:
        if day_data:
            batch_data.extend(day_data)

    # Save to database (much faster than CSV/JSON)
    records_saved = data_manager.save_batch_to_db(batch_data)
    print(f"Batch completed. Records saved to database: {records_saved}")

    return records_saved


async def main():
    start_date = datetime.datetime(1995, 1, 1)
    end_date = datetime.datetime.now()

    print(f"=== Configuration ===")
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"End date: {end_date.strftime('%Y-%m-%d')}")

    # Initialize data manager
    data_manager = StockDataManager()

    # Import existing CSV data if database is empty
    existing_dates = data_manager.get_existing_dates()
    if not existing_dates and Path("us_stocks_bulk_data.csv").exists():
        print("Found existing CSV file. Importing into database...")
        data_manager.import_existing_csv("us_stocks_bulk_data.csv")
        existing_dates = data_manager.get_existing_dates()

    # Check existing data
    latest_date = data_manager.get_latest_date()
    existing_dates = data_manager.get_existing_dates()

    print(f"Existing records in database: {len(existing_dates)} unique dates")
    if latest_date:
        print(f"Latest date in database: {latest_date}")
        # Start from the day after the latest date to avoid duplicates
        try:
            start_date = max(start_date, datetime.datetime.strptime(latest_date, '%Y-%m-%d') + datetime.timedelta(days=1))
            print(f"Adjusted start date: {start_date.strftime('%Y-%m-%d')}")
        except:
            pass

    # Generate business days using pandas
    dates = pd.bdate_range(start=start_date, end=end_date)
    date_strs = [date.strftime("%Y-%m-%d") for date in dates]

    # Filter out dates we already have
    date_strs = [d for d in date_strs if d not in existing_dates]

    print(f"Dates to process: {len(date_strs)}")
    if not date_strs:
        print("No new dates to process!")
        # Still export to CSV if requested
        response = input("Export existing data to CSV? (y/n): ")
        if response.lower() == 'y':
            data_manager.export_to_csv_incremental()
        return

    if date_strs:
        print(f"Date range: {date_strs[0]} to {date_strs[-1]}")

    # Process dates in batches
    batch_size = 50  # Increased batch size since DB is faster
    total_records = 0

    # Limit concurrent requests
    semaphore = asyncio.Semaphore(15)  # Slightly increased

    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),  # Add timeout
            connector=aiohttp.TCPConnector(limit=20)  # Connection pooling
    ) as session:
        for i in range(0, len(date_strs), batch_size):
            batch = date_strs[i:i + batch_size]

            try:
                batch_records = await process_date_batch(
                    session, batch, semaphore, data_manager, existing_dates
                )
                total_records += batch_records

                print(f"Progress: {min(i + batch_size, len(date_strs))}/{len(date_strs)} dates processed")
                print(f"Total new records this session: {total_records}")

                # Smaller delay since we're using database
                if i + batch_size < len(date_strs):
                    await asyncio.sleep(0.5)

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue

    print(f"\n=== Session Summary ===")
    print(f"New records added: {total_records}")
    print(f"Database file: {data_manager.db_path}")

    # Ask if user wants to export to CSV
    if total_records > 0:
        response = input("Export all data to CSV? This may take a while for large datasets (y/n): ")
        if response.lower() == 'y':
            data_manager.export_to_csv_incremental()

    # Show sample data
    try:
        with sqlite3.connect(data_manager.db_path) as conn:
            sample_df = pd.read_sql_query(
                "SELECT * FROM stock_data ORDER BY date DESC, code LIMIT 5",
                conn
            )
            print(f"\nSample of latest data:")
            print(sample_df)

            # Show total count
            total_count = conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
            print(f"Total records in database: {total_count}")
    except Exception as e:
        print(f"Error reading sample data: {e}")


if __name__ == "__main__":
    asyncio.run(main())