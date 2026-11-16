
from datetime import datetime, timezone
import calendar
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Kyoto real-time Dst quicklook page
LIVE_URL = (
    "https://wdc.kugi.kyoto-u.ac.jp/"
    "dst_realtime/presentmonth/index.html"
)


def fetch_live_dst():
    """
    Download and parse the current month's real-time Dst data.

    Returns
    -------
    df : pandas.DataFrame
        timestamp : UTC timestamp
        dst_nt    : Dst value in nT

    update_time : datetime or None
        Kyoto page update time in UTC.
    """

    # --------------------------------------------------
    # 1. Download Kyoto page
    # --------------------------------------------------

    response = requests.get(
        LIVE_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    data_block = soup.find("pre", class_="data")

    if data_block is None:
        raise ValueError(
            "Could not find the Dst data block on Kyoto website."
        )

    page_text = data_block.get_text("\n")

    # --------------------------------------------------
    # 2. Extract Kyoto update timestamp
    # --------------------------------------------------

    page_text_all = soup.get_text(" ", strip=True)

    update_match = re.search(
        r"Updated at\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})UT",
        page_text_all,
        flags=re.IGNORECASE
    )

    update_time = None

    if update_match:

        update_time = datetime.strptime(
            update_match.group(1) + " " + update_match.group(2),
            "%Y-%m-%d %H:%M"
        ).replace(tzinfo=timezone.utc)

    # --------------------------------------------------
    # 3. Determine current month
    # --------------------------------------------------

    now = datetime.now(timezone.utc)

    year = now.year
    month = now.month

    days_in_month = calendar.monthrange(year, month)[1]

    # --------------------------------------------------
    # 4. Parse hourly Dst values
    # --------------------------------------------------

    records = []

    for line in page_text.splitlines():

        line = line.strip()

        match = re.match(
            r"^(\d{1,2})\s+(.*)$",
            line
        )

        if not match:
            continue

        day = int(match.group(1))

        if day < 1 or day > days_in_month:
            continue

        # Exclude future calendar dates.
        if day > now.day:
            continue

        values = match.group(2).split()

        # Kyoto columns are labelled 1 through 24.
        # Column 1 corresponds to 00:00 UTC.
        for hour, token in enumerate(values[:24], start=1):

            # Reject concatenated or malformed values.
            if not re.fullmatch(r"-?\d+", token):
                continue

            value = int(token)

            # Reject missing-data markers and implausible values.
            if not (-2000 < value < 2000):
                continue

            timestamp = datetime(
                year,
                month,
                day,
                hour - 1,
                tzinfo=timezone.utc
            )

            # Do not include future observations.
            if timestamp > now:
                continue

            # If Kyoto provides an update time, do not include
            # timestamps later than that update time.
            if update_time is not None and timestamp > update_time:
                continue

            records.append({
                "timestamp": timestamp,
                "dst_nt": value
            })

    # --------------------------------------------------
    # 5. Create DataFrame
    # --------------------------------------------------

    if not records:
        raise ValueError(
            "No valid Dst observations could be extracted."
        )

    df = pd.DataFrame(records)

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df, update_time


# ------------------------------------------------------
# 6. Run independently for testing
# ------------------------------------------------------

if __name__ == "__main__":

    try:

        dst, update_time = fetch_live_dst()

        print("\n===================================")
        print("KYOTO REAL-TIME DST MONITOR")
        print("===================================")

        print("\nSource: Kyoto WDC")
        print("Data type: REAL-TIME QUICKLOOK")

        if update_time is not None:
            print(
                "\nKyoto page updated at:",
                update_time.strftime("%Y-%m-%d %H:%M UTC")
            )
        else:
            print("\nKyoto update timestamp: Not available")

        print("\nLatest available Dst readings:")

        print(
            dst.tail(12).to_string(index=False)
        )

        print("\nLatest observation:")
        print(dst.iloc[-1]["timestamp"])
        print("Dst:", dst.iloc[-1]["dst_nt"], "nT")

        print("\nTotal valid observations:", len(dst))

        print("\nWarning:")
        print(
            "Real-time quicklook Dst may contain artificial "
            "noise and baseline shifts. Use provisional or "
            "final Dst for scientific analysis."
        )

        print("\n===================================")

    except Exception as error:

        print("\nLive Dst download failed:")
        print(error)
