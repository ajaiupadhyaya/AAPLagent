from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, cast

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when market data fails integrity checks."""


@dataclass(frozen=True)
class ValidationConfig:
    timestamp_col: str
    timezone: str = "America/New_York"
    expected_freq: str | None = None
    is_intraday: bool = False
    session_start: str = "09:30"
    session_end: str = "16:00"
    zscore_threshold: float = 6.0
    mad_threshold: float = 12.0


@dataclass(frozen=True)
class ValidationResult:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    timezone: str


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")


def _normalize_timestamps(series: pd.Series, timezone: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.isna().any():
        raise DataValidationError("Timestamp parsing failed for one or more rows.")

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone)
    else:
        ts = ts.dt.tz_convert(timezone)

    return ts.dt.tz_convert("UTC")


def _check_duplicate_timestamps(ts: pd.Series) -> list[pd.Timestamp]:
    duplicated = ts[ts.duplicated(keep=False)]
    if duplicated.empty:
        return []
    return sorted(pd.unique(duplicated))


def _infer_or_get_frequency(ts: pd.Series, expected_freq: str | None) -> str | None:
    if expected_freq:
        return expected_freq
    inferred = pd.infer_freq(ts.sort_values())
    return inferred


def _check_missing_timestamps(ts: pd.Series, frequency: str | None) -> list[pd.Timestamp]:
    if frequency is None:
        return []
    ts_sorted = ts.sort_values()
    expected = pd.date_range(start=ts_sorted.iloc[0], end=ts_sorted.iloc[-1], freq=frequency)
    missing = expected.difference(pd.DatetimeIndex(ts_sorted))
    return list(missing)


def _check_intraday_session_gaps(
    ts: pd.Series,
    frequency: str,
    session_start: str,
    session_end: str,
    timezone: str,
) -> dict[str, int]:
    local_ts = ts.dt.tz_convert(timezone)
    df = pd.DataFrame({"ts": local_ts})
    df["session_day"] = df["ts"].dt.floor("D")

    gaps: dict[str, int] = {}
    for day, day_frame in df.groupby("session_day", sort=True):
        day_ts = cast(pd.Timestamp, day)
        start = pd.Timestamp(f"{day_ts.date()} {session_start}", tz=timezone)
        end = pd.Timestamp(f"{day_ts.date()} {session_end}", tz=timezone)
        expected = pd.date_range(start=start, end=end, freq=frequency)
        observed = pd.DatetimeIndex(day_frame["ts"].sort_values())
        missing = expected.difference(observed)
        if len(missing) > 0:
            gaps[str(day_ts.date())] = int(len(missing))
    return gaps


def _split_adjustment_issue(df: pd.DataFrame) -> int:
    if "Adj Close" not in df.columns:
        return 0

    close_ret = df["Close"].pct_change()
    adj_ret = df["Adj Close"].pct_change()
    divergence = (close_ret - adj_ret).abs()

    potential = divergence[(divergence > 0.2) & (close_ret.abs() > 0.3)]
    return int(potential.count())


def _outlier_mask(series: pd.Series, zscore_threshold: float, mad_threshold: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 10:
        return pd.Series(False, index=series.index)

    std = float(x.std(ddof=0))
    if std == 0.0:
        zmask = pd.Series(False, index=x.index)
    else:
        zscores = (x - x.mean()) / std
        zmask = zscores.abs() > zscore_threshold

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad == 0.0:
        mmad = pd.Series(False, index=x.index)
    else:
        modified_z = 0.6745 * (x - med) / mad
        mmad = modified_z.abs() > mad_threshold

    flagged = zmask | mmad
    out = pd.Series(False, index=series.index)
    out.loc[flagged.index] = flagged
    return out


def validate_market_data(df: pd.DataFrame, config: ValidationConfig) -> ValidationResult:
    """Validate and normalize market data; raises DataValidationError on integrity failures."""

    _require_columns(df, [config.timestamp_col, "Open", "High", "Low", "Close", "Volume"])

    working = df.copy()
    working[config.timestamp_col] = _normalize_timestamps(working[config.timestamp_col], config.timezone)
    working = working.sort_values(config.timestamp_col).reset_index(drop=True)

    issues: list[str] = []

    dupes = _check_duplicate_timestamps(working[config.timestamp_col])
    if dupes:
        issues.append(f"Duplicate timestamps detected: {len(dupes)} unique duplicates.")

    frequency = _infer_or_get_frequency(working[config.timestamp_col], config.expected_freq)
    missing = _check_missing_timestamps(working[config.timestamp_col], frequency)
    if missing:
        issues.append(f"Missing timestamps detected: {len(missing)} gaps at frequency {frequency}.")

    split_issues = _split_adjustment_issue(working)
    if split_issues > 0:
        issues.append(
            "Potential split/dividend adjustment mismatch between Close and Adj Close "
            f"on {split_issues} rows."
        )

    if config.is_intraday:
        intraday_freq = frequency or config.expected_freq
        if intraday_freq is None:
            issues.append(
                "Intraday frequency could not be inferred; provide ValidationConfig.expected_freq."
            )
        else:
            gaps = _check_intraday_session_gaps(
                ts=working[config.timestamp_col],
                frequency=intraday_freq,
                session_start=config.session_start,
                session_end=config.session_end,
                timezone=config.timezone,
            )
            if gaps:
                total_missing = sum(gaps.values())
                sample = ", ".join([f"{k}:{v}" for k, v in list(gaps.items())[:5]])
                issues.append(
                    f"Intraday session gaps detected: {total_missing} missing bars across days ({sample})."
                )

    outlier_columns = ["Close", "Volume", "Open", "High", "Low"]
    for col in outlier_columns:
        outlier_count = int(
            _outlier_mask(
                working[col],
                zscore_threshold=config.zscore_threshold,
                mad_threshold=config.mad_threshold,
            ).sum()
        )
        if outlier_count > 0:
            issues.append(f"Outliers detected in {col}: {outlier_count} rows (z-score + MAD).")

    if issues:
        message = "Data validation failed:\n- " + "\n- ".join(issues)
        raise DataValidationError(message)

    return ValidationResult(
        rows=len(working),
        start=working[config.timestamp_col].iloc[0],
        end=working[config.timestamp_col].iloc[-1],
        timezone="UTC",
    )


def validate_market_data_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    timezone: str = "America/New_York",
    expected_freq: str | None = None,
    is_intraday: bool = False,
) -> pd.DataFrame:
    """Convenience API that validates and returns a normalized UTC-sorted copy."""

    config = ValidationConfig(
        timestamp_col=timestamp_col,
        timezone=timezone,
        expected_freq=expected_freq,
        is_intraday=is_intraday,
    )
    _ = validate_market_data(df, config)
    out = df.copy()
    out[timestamp_col] = _normalize_timestamps(out[timestamp_col], timezone)
    return out.sort_values(timestamp_col).reset_index(drop=True)
