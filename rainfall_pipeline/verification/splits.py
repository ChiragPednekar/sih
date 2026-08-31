"""Chronological train / validation / test splitting.

This is a time series. A random shuffle would put a 3 July row in training and
a 2 July row in test, and since rainfall is strongly autocorrelated in space and
time the model would effectively be reading the answer -- every skill number
that came out of it would be fiction.

There is no ``random_state`` parameter anywhere in this module, and there never
should be. The only supported split is by date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class Split:
    """A chronological three-way split.

    Attributes:
        train: Rows on or before ``train_end``, used to fit models.
        validation: Rows after ``train_end`` and on or before ``val_end``, used
            for probability calibration and hyperparameter choices.
        test: Rows after ``val_end``. Touched exactly once, at report time.
        train_end: The boundary date used.
        val_end: The boundary date used.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp

    def summary(self) -> dict:
        """Return row counts and date ranges for each split."""

        def _describe(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"n_rows": 0, "start": None, "end": None}
            return {
                "n_rows": int(len(df)),
                "start": str(df["date"].min().date()),
                "end": str(df["date"].max().date()),
            }

        return {
            "train": _describe(self.train),
            "validation": _describe(self.validation),
            "test": _describe(self.test),
        }


class SplitError(ValueError):
    """Raised when a chronological split cannot be made."""


def chronological_split(
    df: pd.DataFrame,
    *,
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    date_col: str = "date",
) -> Split:
    """Split ``df`` into train / validation / test by date.

    Args:
        df: Frame with a datetime ``date`` column.
        train_end: Last date in the training set (inclusive), ISO format. If
            None, falls back to ``train_fraction`` of the *unique dates*.
        val_end: Last date in the validation set (inclusive). If None, falls
            back to ``val_fraction``.
        train_fraction: Fraction of unique dates used for training when
            ``train_end`` is None.
        val_fraction: Fraction of unique dates used for validation when
            ``val_end`` is None.
        date_col: Name of the date column.

    Returns:
        The populated :class:`Split`.

    Raises:
        SplitError: If the date column is missing, the frame is empty, or the
            boundaries leave a split empty.

    Note:
        The fraction fallback splits on *unique dates*, not on rows, so that all
        grid cells for a given day land on the same side of the boundary. It is
        a convenience for getting a pipeline running; for the numbers that go in
        a report, set explicit boundaries in
        :data:`~rainfall_pipeline.config.thresholds.SPLIT` so the test set is a
        whole held-out monsoon season rather than an arbitrary cut.
    """
    if date_col not in df.columns:
        raise SplitError(f"Cannot split: no '{date_col}' column.")
    if df.empty:
        raise SplitError("Cannot split an empty frame.")
    if not (0 < train_fraction < 1) or not (0 < val_fraction < 1):
        raise SplitError("train_fraction and val_fraction must be in (0, 1).")
    if train_fraction + val_fraction >= 1:
        raise SplitError(
            f"train_fraction + val_fraction must leave room for a test set "
            f"(got {train_fraction} + {val_fraction})."
        )

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    dates = pd.Index(sorted(work[date_col].unique()))

    if train_end is None or val_end is None:
        n = len(dates)
        if n < 3:
            raise SplitError(
                f"Only {n} distinct date(s) present; a chronological three-way "
                f"split needs at least 3. Set explicit train_end/val_end, or "
                f"connect more data."
            )
        auto_train_end = dates[max(int(n * train_fraction) - 1, 0)]
        auto_val_end = dates[max(int(n * (train_fraction + val_fraction)) - 1, 0)]
        # Guarantee a non-empty validation split even on very short records.
        if auto_val_end <= auto_train_end:
            auto_val_end = dates[min(list(dates).index(auto_train_end) + 1, n - 2)]
        train_end = train_end or auto_train_end
        val_end = val_end or auto_val_end

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if val_end_ts < train_end_ts:
        raise SplitError(
            f"val_end ({val_end_ts.date()}) must not precede train_end ({train_end_ts.date()})."
        )

    train = work[work[date_col] <= train_end_ts]
    validation = work[(work[date_col] > train_end_ts) & (work[date_col] <= val_end_ts)]
    test = work[work[date_col] > val_end_ts]

    empty = [n for n, part in (("train", train), ("test", test)) if part.empty]
    if empty:
        raise SplitError(
            f"Split boundaries leave the {' and '.join(empty)} set(s) empty "
            f"(train_end={train_end_ts.date()}, val_end={val_end_ts.date()}, "
            f"data spans {dates[0].date()} to {dates[-1].date()})."
        )
    return Split(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        train_end=train_end_ts,
        val_end=val_end_ts,
    )


def resolve_split_bounds(configured_train_end, configured_val_end) -> Tuple[Optional[str], Optional[str]]:
    """Normalise configured split boundaries to ISO strings or None.

    Args:
        configured_train_end: Value from
            :data:`~rainfall_pipeline.config.thresholds.SPLIT`.
        configured_val_end: Value from the same config.

    Returns:
        ``(train_end, val_end)`` as ISO strings, or ``(None, None)`` if either
        is unset -- in which case the caller falls back to fractions.
    """
    if configured_train_end is None or configured_val_end is None:
        return None, None
    return str(configured_train_end), str(configured_val_end)
