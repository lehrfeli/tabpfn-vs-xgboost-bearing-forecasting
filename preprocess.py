"""
Preprocessing for the NASA IMS Bearing Dataset (Test 2).

Reads the raw ASCII vibration snapshot files (one file per timestamp,
tab-separated, no header, one column per bearing channel) and reduces
each snapshot to a handful of summary statistics per channel. This turns
~984 raw signal files into a single tabular dataframe suitable for
regression.
"""

import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

N_CHANNELS = 4  # bearings 1-4, one accelerometer channel each


def parse_timestamp(filepath: Path) -> datetime:
    """File names encode the snapshot time, e.g. '2004.02.12.10.32.39'."""
    filename = os.path.basename(filepath)
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def calc_stats(channel_data: np.ndarray) -> dict:
    """Reduce one channel's raw vibration signal to summary statistics."""
    rms = np.sqrt(np.mean(channel_data ** 2))
    peak = np.max(np.abs(channel_data))
    return {
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis(channel_data),
        "skew": skew(channel_data),
        "std": np.std(channel_data),
        "crest_factor": peak / rms,
    }


def load_snapshots(data_dir: Path) -> pd.DataFrame:
    """Build one row per snapshot file: timestamp + 6 stats x N_CHANNELS columns."""
    files = sorted(Path(data_dir).glob("*"))
    rows = defaultdict(list)

    for file in files:
        rows["timestamp"].append(parse_timestamp(file))
        snapshot = pd.read_csv(file, sep="\t", header=None)
        for channel in range(N_CHANNELS):
            stats = calc_stats(snapshot[channel].values)
            for stat_name, value in stats.items():
                rows[f"{stat_name}{channel + 1}"].append(value)

    return pd.DataFrame(rows)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target is the next snapshot's bearing-1 RMS (the channel that failed
    in this test - outer race failure). Shifting by -1 makes each row
    predict the following timestep.
    """
    df = df.copy()
    df["target"] = df["rms1"].shift(-1)
    # last row has a NaN target (nothing to shift in), and the two rows
    # before it reflect the sensor dropping off at end-of-life rather than
    # real degradation signal - drop all three
    df = df.iloc[:-3]
    return df


def load_and_preprocess(data_dir: Path) -> pd.DataFrame:
    df = load_snapshots(data_dir)
    df = add_target(df)
    return df


if __name__ == "__main__":
    df = load_and_preprocess(Path("data/NASA_bearing_datasest/2nd_test/2nd_test"))
    print(df.shape)
    print(df.head())
