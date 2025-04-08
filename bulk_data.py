import asyncio
import datetime
import json
import os

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


async def main():
    # Define the date range for the last 3 years
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=2)

    # Generate business days using pandas
    dates = pd.bdate_range(start=start_date, end=end_date)
    date_strs = [date.strftime("%Y-%m-%d") for date in dates]

    all_data = []
    # Limit to 10 concurrent requests
    semaphore = asyncio.Semaphore(10)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_day_data(session, date_str, semaphore) for date_str in date_strs]
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists
        for day_data in results:
            if day_data:
                all_data.extend(day_data)

    if all_data:
        # Convert the aggregated data into a DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        print("Sample data:")
        print(df.head())

        csv_output_file = "us_stocks_bulk_data.csv"
        df.to_csv(csv_output_file, index=False)
        print(f"Data saved to {csv_output_file}")

        # Save the full data to a JSON file
        json_output_file = "us_stocks_bulk_data.json"
        with open(json_output_file, "w") as f:
            json.dump(all_data, f)
        print(f"Data saved to {json_output_file}")
    else:
        print("No data was retrieved.")


if __name__ == "__main__":
    asyncio.run(main())
