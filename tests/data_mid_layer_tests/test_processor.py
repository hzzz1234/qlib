# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np
import pandas as pd
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.data.dataset.processor import MinMaxNorm, ZScoreNorm, CSZScoreNorm, CSZFillna, FilterOutlier, ClipOutlier


class TestProcessor(TestAutoData):
    TEST_INST = "SH600519"

    def test_MinMaxNorm(self):
        def normalize(df):
            min_val = np.nanmin(df.values, axis=0)
            max_val = np.nanmax(df.values, axis=0)
            ignore = min_val == max_val
            for _i, _con in enumerate(ignore):
                if _con:
                    max_val[_i] = 1
                    min_val[_i] = 0
            df.loc(axis=1)[df.columns] = (df.values - min_val) / (max_val - min_val)
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        mmn = MinMaxNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        mmn.fit(df)
        mmn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_ZScoreNorm(self):
        def normalize(df):
            mean_train = np.nanmean(df.values, axis=0)
            std_train = np.nanstd(df.values, axis=0)
            ignore = std_train == 0
            for _i, _con in enumerate(ignore):
                if _con:
                    std_train[_i] = 1
                    mean_train[_i] = 0
            df.loc(axis=1)[df.columns] = (df.values - mean_train) / std_train
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        zsn = ZScoreNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        zsn.fit(df)
        zsn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_CSZFillna(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[97:99])[228:238]
        df = origin_df.copy()
        CSZFillna(fields_group=None).__call__(df)
        assert ~df[1:2].isna().all().all() and origin_df[1:2].isna().all().all()

    def test_CSZScoreNorm(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[10:12])[50:60]
        df = origin_df.copy()
        CSZScoreNorm(fields_group=None).__call__(df)
        # If we use the formula directly on the original data, we cannot get the correct result,
        # because the original data is processed by `groupby`, so we use the method of slicing,
        # taking the 2nd group of data from the original data, to calculate and compare.
        assert (df[2:4] == ((origin_df[2:4] - origin_df[2:4].mean()).div(origin_df[2:4].std()))).all().all()

    def test_FilterOutlier(self):
        """Test FilterOutlier processor"""
        # Create test data with known outliers
        test_data = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 10.0, 3.0, 4.0],  # 10.0 is outlier
                "col2": [1.5, 2.5, 3.5, 4.5, 5.5],
                "col3": [-1.0, 0.0, 1.0, 2.0, 100.0],  # 100.0 is outlier
            },
            index=pd.MultiIndex.from_tuples(
                [("2021-01-01", "stock1"), ("2021-01-02", "stock1"), ("2021-01-03", "stock1"),
                 ("2021-01-04", "stock1"), ("2021-01-05", "stock1")],
                names=["datetime", "instrument"]
            ),
        )

        # Test with both upper and lower bounds
        filter_proc = FilterOutlier(fields_group=None, lower=0.0, upper=20.0)
        result = filter_proc(test_data.copy())
        # Row 0 (col3=-1.0) should be filtered out due to lower bound
        # Rows 1, 2, 3 should remain (all values within [0.0, 20.0])
        # Row 4 (col3=100.0) should be filtered out due to upper bound
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
        assert 100.0 not in result["col3"].values, "Outlier 100.0 should be filtered out"
        assert -1.0 not in result["col3"].values, "Outlier -1.0 should be filtered out"

        # Test with only upper bound
        filter_proc_upper = FilterOutlier(fields_group=None, upper=5.0)
        result_upper = filter_proc_upper(test_data.copy())
        # Row 0: [1.0, 1.5, -1.0] all <= 5.0 → keep
        # Row 1: [2.0, 2.5, 0.0] all <= 5.0 → keep
        # Row 2: [10.0, 3.5, 1.0] col1=10.0 > 5.0 → filter out
        # Row 3: [3.0, 4.5, 2.0] all <= 5.0 → keep
        # Row 4: [4.0, 5.5, 100.0] col2=5.5, col3=100.0 > 5.0 → filter out
        assert len(result_upper) == 3, f"Expected 3 rows after upper bound filtering, got {len(result_upper)}"

        # Test with only lower bound
        filter_proc_lower = FilterOutlier(fields_group=None, lower=-0.5)
        result_lower = filter_proc_lower(test_data.copy())
        # Row 0 with col3=-1.0 should be filtered out (< -0.5)
        # Rows 1, 2, 3, 4 should remain
        assert len(result_lower) == 4, f"Expected 4 rows after lower bound filtering, got {len(result_lower)}"
        assert -1.0 not in result_lower["col3"].values, "Value -1.0 should be filtered out"

        # Test readonly property
        assert filter_proc.readonly() == True, "FilterOutlier should be readonly"

    def test_ClipOutlier(self):
        """Test ClipOutlier processor"""
        # Create test data with known outliers
        test_data = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 10.0, 3.0, 4.0],
                "col2": [1.5, 2.5, 3.5, 4.5, 5.5],
                "col3": [-5.0, 0.0, 1.0, 2.0, 100.0],
            },
            index=pd.MultiIndex.from_tuples(
                [("2021-01-01", "stock1"), ("2021-01-02", "stock1"), ("2021-01-03", "stock1"),
                 ("2021-01-04", "stock1"), ("2021-01-05", "stock1")],
                names=["datetime", "instrument"]
            ),
        )

        # Test with both upper and lower bounds
        clip_proc = ClipOutlier(fields_group=None, lower=0.0, upper=10.0)
        result = clip_proc(test_data.copy())
        # All rows should remain
        assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
        # Values should be clipped
        assert result["col1"].max() <= 10.0, "col1 max should be clipped to 10.0"
        assert result["col3"].min() >= 0.0, "col3 min should be clipped to 0.0"
        assert result["col3"].max() <= 10.0, "col3 max should be clipped to 10.0"
        assert result.loc[("2021-01-05", "stock1"), "col3"] == 10.0, "100.0 should be clipped to 10.0"
        assert result.loc[("2021-01-01", "stock1"), "col3"] == 0.0, "-5.0 should be clipped to 0.0"

        # Test with only upper bound
        clip_proc_upper = ClipOutlier(fields_group=None, upper=5.0)
        result_upper = clip_proc_upper(test_data.copy())
        assert result_upper["col1"].max() <= 5.0, "Values should be clipped to upper bound 5.0"
        assert result_upper["col3"].max() <= 5.0, "Values should be clipped to upper bound 5.0"
        # Lower values should not be affected
        assert result_upper.loc[("2021-01-01", "stock1"), "col3"] == -5.0, "Lower values should not be affected"

        # Test with only lower bound
        clip_proc_lower = ClipOutlier(fields_group=None, lower=-2.0)
        result_lower = clip_proc_lower(test_data.copy())
        assert result_lower["col3"].min() >= -2.0, "Values should be clipped to lower bound -2.0"
        assert result_lower.loc[("2021-01-01", "stock1"), "col3"] == -2.0, "-5.0 should be clipped to -2.0"
        # Upper values should not be affected
        assert result_lower.loc[("2021-01-05", "stock1"), "col3"] == 100.0, "Upper values should not be affected"


if __name__ == "__main__":
    unittest.main()
