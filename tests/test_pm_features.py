"""
Uji untuk pm_features.py. Jalankan dari folder proyek:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pm_features import BASE, FEAT, ROLL, build_features, prepare  # noqa: E402


def make_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for dev in ["A", "B"]:
        for i, d in enumerate(dates):
            rows.append({"date": d.strftime("%m/%d/%Y"), "device": dev, "failure": 0,
                         **{m: float(i + (100 if dev == "B" else 0)) for m in
                            [f"metric{k}" for k in range(1, 10)]}})
    return pd.DataFrame(rows)


class TestPmFeatures(unittest.TestCase):
    def test_metric8_dropped(self):
        self.assertNotIn("metric8", BASE)
        self.assertEqual(len(FEAT), 3 * len(BASE))

    def test_no_future_leakage(self):
        """Mengubah nilai masa depan tidak boleh mengubah fitur hari ini."""
        df = prepare(make_df())
        f1 = build_features(df)
        df2 = df.copy()
        mask = (df2["device"] == "A") & (df2.index > df2.index[4])
        df2.loc[mask, "metric1"] = 9999.0
        f2 = build_features(df2)
        same_before = df.index[(df["device"] == "A")][:5]
        pd.testing.assert_frame_equal(f1.loc[same_before], f2.loc[same_before])

    def test_devices_do_not_mix(self):
        """Rolling dihitung per perangkat: perangkat B tidak boleh memengaruhi A."""
        df = prepare(make_df())
        f = build_features(df)
        a = f.loc[df["device"] == "A", "metric1_mean7"]
        self.assertTrue((a < np.log1p(50)).all())

    def test_rolling_window_length(self):
        df = prepare(make_df())
        f = build_features(df)
        a_idx = df.index[df["device"] == "A"]
        vals = np.log1p(df.loc[a_idx, "metric1"].values)
        expected = pd.Series(vals).rolling(ROLL, min_periods=1).mean().values
        np.testing.assert_allclose(f.loc[a_idx, "metric1_mean7"].values, expected)

    def test_negative_values_clipped(self):
        df = make_df()
        df.loc[0, "metric1"] = -5.0
        f = build_features(prepare(df))
        self.assertTrue(np.isfinite(f.values).all())
        self.assertEqual(f["metric1_log"].min(), 0.0)


if __name__ == "__main__":
    unittest.main()
