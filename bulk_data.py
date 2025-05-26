import asyncio
import datetime
import json
import os
from pathlib import Path

import aiohttp
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_token = os.environ.get("EODHD_API_TOKEN")


async def fetch_day_data(session, date_str, semaphore):
    """Fetch bulk data for a specific date using aiohttp."""
    url = f"https://eodhd.com/api/eod-bulk-last-day/US?api_token={api_token}&date={date_str}&fmt=json"
    print(f"Requesting data for {date_str} ...")

    async with semaphore:  # Limit concurrent requests
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        # Ensure each entry includes the date
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
    return []  # Return an empty list if there was an error


async def save_batch_data(batch_data, csv_file, json_file):
    """Save a batch of data to both CSV and JSON files (always append)."""
    if not batch_data:
        return

    # Convert to DataFrame for CSV saving
    df = pd.DataFrame(batch_data)

    # Check if CSV file exists to determine if we need headers
    csv_exists = Path(csv_file).exists()

    # Save to CSV (always append, include headers only if file doesn't exist)
    df.to_csv(csv_file, index=False, mode='a', header=not csv_exists)

    if csv_exists:
        print(f"Appended {len(batch_data)} records to existing CSV")
    else:
        print(f"Created new CSV file with {len(batch_data)} records")

    # Save to JSON (always append)
    json_exists = Path(json_file).exists()

    if json_exists:
        # Read existing JSON, append new data, and write back
        try:
            with open(json_file, "r") as f:
                existing_data = json.load(f)
            existing_data.extend(batch_data)
            with open(json_file, "w") as f:
                json.dump(existing_data, f)
            print(f"Appended {len(batch_data)} records to existing JSON")
        except Exception as e:
            print(f"Error appending to JSON file: {e}")
            # Fallback: create new file with just this batch
            with open(json_file, "w") as f:
                json.dump(batch_data, f)
            print(f"Created new JSON file due to error")
    else:
        # Create new JSON file
        with open(json_file, "w") as f:
            json.dump(batch_data, f)
        print(f"Created new JSON file with {len(batch_data)} records")


async def process_date_batch(session, date_batch, semaphore, csv_file, json_file):
    """Process a batch of dates and save the results incrementally."""
    print(f"\nProcessing batch of {len(date_batch)} dates...")

    # Fetch data for this batch of dates
    tasks = [fetch_day_data(session, date_str, semaphore) for date_str in date_batch]
    results = await asyncio.gather(*tasks)

    # Flatten the results for this batch
    batch_data = []
    for day_data in results:
        if day_data:
            batch_data.extend(day_data)

    # Save this batch immediately
    await save_batch_data(batch_data, csv_file, json_file)

    print(f"Batch completed. Total records in this batch: {len(batch_data)}")
    return len(batch_data)


async def main():
    start_date = datetime.datetime(1986, 7, 12)
    end_date = datetime.datetime.now()

    print(f"=== Configuration ===")
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"End date: {end_date.strftime('%Y-%m-%d')}")

    # Generate business days using pandas
    dates = pd.bdate_range(start=start_date, end=end_date)
    date_strs = [date.strftime("%Y-%m-%d") for date in dates]

    print(f"Total dates to process: {len(date_strs)}")
    if date_strs:
        print(f"Date range: {date_strs[0]} to {date_strs[-1]}")

    # Set up output files
    csv_output_file = "us_stocks_bulk_data.csv"
    json_output_file = "us_stocks_bulk_data.json"

    # Check if files already exist
    csv_exists = Path(csv_output_file).exists()
    json_exists = Path(json_output_file).exists()

    print(f"\n=== File Status ===")
    print(f"CSV file exists: {csv_exists}")
    print(f"JSON file exists: {json_exists}")

    if csv_exists:
        try:
            existing_df = pd.read_csv(csv_output_file)
            print(f"Existing CSV has {len(existing_df)} records")
            if 'date' in existing_df.columns:
                latest_date = existing_df['date'].max()
                print(f"Latest date in existing data: {latest_date}")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    # Process dates in batches to manage memory
    batch_size = 30  # Process 30 days at a time
    total_records = 0

    # Limit concurrent requests per batch
    semaphore = asyncio.Semaphore(10)

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(date_strs), batch_size):
            batch = date_strs[i:i + batch_size]

            try:
                batch_records = await process_date_batch(
                    session, batch, semaphore, csv_output_file, json_output_file
                )
                total_records += batch_records

                print(f"Progress: {min(i + batch_size, len(date_strs))}/{len(date_strs)} dates processed")
                print(f"Total records in this session: {total_records}")

                # Add a small delay between batches to be nice to the API
                if i + batch_size < len(date_strs):
                    await asyncio.sleep(1)

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue

    print(f"\n=== Session Summary ===")
    print(f"Records added this session: {total_records}")
    print(f"CSV file: {csv_output_file}")
    print(f"JSON file: {json_output_file}")

    # Show a sample of the final data
    if Path(csv_output_file).exists():
        try:
            sample_df = pd.read_csv(csv_output_file, nrows=5)
            print(f"\nSample data from CSV:")
            print(sample_df)

            # Show total count
            total_df = pd.read_csv(csv_output_file)
            print(f"Total records in CSV: {len(total_df)}")
        except Exception as e:
            print(f"Error reading sample data: {e}")


if __name__ == "__main__":
    asyncio.run(main())