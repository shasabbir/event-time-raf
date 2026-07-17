from __future__ import annotations

import pandas as pd

from event_timeraf.data import _event_coverage_days


def test_event_coverage_counts_union_of_source_ranges():
    events = pd.DataFrame(
        {
            "source_coverage_start": pd.to_datetime(
                ["2020-01-01", "2020-01-05", "2021-01-01"], utc=True
            ),
            "source_coverage_end": pd.to_datetime(
                ["2020-01-10", "2020-01-15", "2021-01-04"], utc=True
            ),
        }
    )
    result = _event_coverage_days(
        events,
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2021-01-03", tz="UTC"),
    )
    assert result == 16
