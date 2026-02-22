#!/usr/bin/env python3
"""
Add a trend_score column to FX 10-second candle CSV files.

Input:
- CSV files (comma-separated) with at least a 'close' column.

Output:
- Same CSV structure + an extra column 'trend_score' in [-1.000, +1.000]
  where:
    +1.000 = strong uptrend
     0.000 = ranging / no clear trend
    -1.000 = strong downtrend

Methodology (the “4 steps”):
1) Rolling linear regression on CLOSE over a lookback window (e.g. 60 bars = 10 min):
       price ~ a + b * time_index
   -> b (slope) captures trend direction and magnitude

2) Convert slope into a robust “trend strength” by normalising with volatility:
       trend_raw = slope / rolling_std(price)
   -> this is a dimensionless measure of “drift vs noise”

3) Scale trend_raw into a consistent [-1, +1] score using data-driven calibration:
   - Compute a scaling factor = percentile(|trend_raw|) (e.g. 95th)
   - Strongest ~5% trends will saturate towards ±1

4) Final score:
       trend_score = clip(trend_raw / scale, -1, +1)
   and fill warm-up NaNs with 0.0 (no trend).
"""

from pathlib import Path
import numpy as np
import pandas as pd


def add_trend_score_linreg(
    df: pd.DataFrame,
    window: int = 60,            # e.g. 60 * 10s bars = 10 minutes lookback
    percentile: float = 95.0,    # used to scale the raw trend into [-1, +1]
    eps: float = 1e-12           # numerical stability guard (avoid div by zero)
) -> pd.DataFrame:
    """
    Add a 'trend_score' column based on rolling linear regression.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'close' column.
    window : int
        Rolling window length in bars (60 = 10 minutes for 10s bars).
    percentile : float
        Percentile of |trend_raw| used for scaling.
        Example: 95 means values stronger than 95% of history saturate near ±1.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    DataFrame
        Original df with an extra 'trend_score' column in [-1, 1],
        rounded to 3 decimals.
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    # Ensure close is numeric
    close = df["close"].astype(float)

    # -------------------------------------------------------------------------
    # STEP 1: Rolling linear regression to estimate trend slope
    # -------------------------------------------------------------------------
    # We fit: price ~ a + b * x, where x = 0..window-1.
    # The slope b captures direction:
    #   b > 0 => upward trend
    #   b < 0 => downward trend
    #   b ~ 0 => no trend / ranging
    x = np.arange(window, dtype=float)

    def slope_func(y: np.ndarray) -> float:
        # y is an array of the last 'window' closes.
        # np.polyfit(x, y, 1) returns [slope, intercept] for y ≈ slope*x + intercept.
        slope_b, intercept_a = np.polyfit(x, y, 1)
        return slope_b

    # Rolling slope: requires a full window, so first (window-1) values will be NaN.
    slope = close.rolling(window=window, min_periods=window).apply(
        slope_func, raw=True
    )

    # -------------------------------------------------------------------------
    # STEP 2: Normalise slope by volatility (noise) to get trend_raw
    # -------------------------------------------------------------------------
    # We compute rolling standard deviation of price in the same window.
    # This turns the slope into a dimensionless measure:
    #   trend_raw = slope / std
    # Intuition: "how strong is the drift relative to price wiggles?"
    roll_std = close.rolling(window=window, min_periods=window).std()

    trend_raw = slope / (roll_std + eps)

    # -------------------------------------------------------------------------
    # STEP 3: Convert trend_raw to a stable [-1, +1] scale using percentile scaling
    # -------------------------------------------------------------------------
    # If we divide by a fixed constant, trend_score might be tiny (as you saw).
    # Instead, we use a data-driven scale:
    #   scale = percentile(|trend_raw|)
    # So the strongest ~ (100 - percentile)% of observations saturate towards ±1.
    abs_raw = trend_raw.abs().dropna()
    if len(abs_raw) == 0:
        scale = 1.0  # fallback if data is too short or degenerate
    else:
        scale = np.percentile(abs_raw, percentile)
        if scale < eps:
            scale = 1.0  # guard against pathological cases

    # -------------------------------------------------------------------------
    # STEP 4: Clip to [-1, +1] and create final trend_score
    # -------------------------------------------------------------------------
    # trend_score is now a clean bounded number:
    #   +1 => strong uptrend
    #    0 => ranging / unclear
    #   -1 => strong downtrend
    trend_scaled = np.clip(trend_raw / scale, -1.0, 1.0)

    # Warm-up period (first window-1 rows) have NaN slope/std → set to 0.0 (no trend).
    df["trend_score"] = trend_scaled.fillna(0.0).round(3)

    return df


def process_file(
    in_path: Path,
    out_path: Path,
    window: int = 60,
    percentile: float = 95.0
) -> None:
    """
    Read a CSV, add trend_score, write to new CSV.
    """
    print(f"Processing {in_path.name} -> {out_path.name}")
    df = pd.read_csv(in_path)  # comma-separated by default
    df = add_trend_score_linreg(df, window=window, percentile=percentile)
    df.to_csv(out_path, index=False)


def main(
    input_dir: str,
    pattern: str = "*.csv",
    suffix: str = "-with-trend",
    window: int = 60,
    percentile: float = 95.0
) -> None:
    """
    Apply trend scoring to all matching CSV files in a directory.
    """
    input_dir_path = Path(input_dir)
    if not input_dir_path.is_dir():
        raise ValueError(f"{input_dir_path} is not a directory")

    for in_path in sorted(input_dir_path.glob(pattern)):
        if not in_path.is_file():
            continue

        out_name = f"{in_path.stem}{suffix}{in_path.suffix}"
        out_path = in_path.with_name(out_name)

        process_file(in_path, out_path, window=window, percentile=percentile)


if __name__ == "__main__":
    # Change this to the folder containing your CSVs
    data_folder = "D:/OneDrive/Job Application/FX"
    main(data_folder)
