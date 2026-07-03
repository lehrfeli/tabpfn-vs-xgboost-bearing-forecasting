"""
Feature engineering for the bearing-RMS forecasting pipeline.

Adds three groups of features on top of the raw per-snapshot statistics
produced by preprocess.py:
  1. A running time index (position in the sequence)
  2. Cyclic calendar features (hour/day/month encoded as sin/cos pairs)
  3. FFT-detected seasonal features from the target channel's RMS signal

This mirrors the "lightweight temporal featurization" idea behind
TabPFN-TS (Hoo et al., 2025): treat forecasting as tabular regression by
handing the model explicit temporal structure instead of raw sequence order.
"""

import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq

N_SEASONAL_PERIODS = 3  # how many dominant FFT periods to encode


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    return df


def add_cyclic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar values (hour, day of week, etc.) wrap around, so raw integers
    would tell the model 23:00 and 00:00 are far apart when they're really
    1 hour apart. Encoding each as a (sin, cos) pair on the unit circle
    fixes this: values close in time stay close numerically across the
    wrap boundary. Both sin and cos are needed since sin alone is
    ambiguous (e.g. 2am and 10am share a sin value).
    """
    df = df.copy()
    ts = df["timestamp"].dt

    # (raw value, length of one full cycle)
    cyclic_components = {
        "hour": (ts.hour, 24),
        "day": (ts.dayofweek, 7),
        "day_month": (ts.day, 31),
        "year": (ts.day_of_year, 365),
        "month_year": (ts.month, 12),
    }
    for name, (value, cycle_length) in cyclic_components.items():
        df[f"{name}_sin"] = np.sin(2 * np.pi * value / cycle_length)
        df[f"{name}_cos"] = np.cos(2 * np.pi * value / cycle_length)

    return df


def add_fft_seasonal_features(
    df: pd.DataFrame, signal_col: str = "rms1", k: int = N_SEASONAL_PERIODS
) -> pd.DataFrame:
    """
    FFT decomposes the signal into its frequency components, automatically
    detecting repeating cycles (e.g. ball-pass frequency, shaft rotation
    frequency, housing resonance) instead of requiring us to hand-pick
    seasonal periods up front. We take the top k strongest periods (after
    the DC component) and encode each as a sin/cos pair over time_idx, so
    every row carries where it falls within each dominant cycle.
    """
    df = df.copy()
    signal = df[signal_col].values
    n = len(signal)

    # fft output is symmetric for a real-valued signal - keep only the
    # positive-frequency half, the negative half is redundant
    fft_magnitudes = np.abs(fft(signal))[: n // 2]
    freqs = fftfreq(n)[: n // 2]

    # skip index 0 (DC component / mean), take the k strongest after that
    top_indices = np.argsort(fft_magnitudes)[::-1][1 : k + 1]
    top_periods = 1 / freqs[top_indices]

    for i, period in enumerate(top_periods, start=1):
        df[f"period{i}_sin"] = np.sin(2 * np.pi * df["time_idx"] / period)
        df[f"period{i}_cos"] = np.cos(2 * np.pi * df["time_idx"] / period)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_index(df)
    df = add_cyclic_calendar_features(df)
    df = add_fft_seasonal_features(df)
    return df


def split_features_target(df: pd.DataFrame):
    """Everything except the timestamp (not a model input) and target is a feature."""
    feature_columns = [c for c in df.columns if c not in ("timestamp", "target")]
    X = df[feature_columns]
    y = df["target"]
    return X, y
