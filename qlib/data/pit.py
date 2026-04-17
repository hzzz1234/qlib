# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Qlib follow the logic below to supporting point-in-time database

For each stock, the format of its data is <observe_time, feature>. Expression Engine support calculation on such format of data

To calculate the feature value f_t at a specific observe time t,  data with format <period_time, feature> will be used.
For example, the average earning of last 4 quarters (period_time) on 20190719 (observe_time)

The calculation of both <period_time, feature> and <observe_time, feature> data rely on expression engine. It consists of 2 phases.
1) calculation <period_time, feature> at each observation time t and it will collasped into a point (just like a normal feature)
2) concatenate all th collasped data, we will get data with format <observe_time, feature>.
Qlib will use the operator `P` to perform the collapse.
"""
import numpy as np
import pandas as pd
from qlib.data.ops import ElemOperator
from qlib.log import get_module_logger
from .data import Cal
from .ops import PFeature
from .base import ExpressionOps

class P(ElemOperator):
    # def _load_internal(self, instrument, start_index, end_index, freq):
    #     feature_data = self.feature.load(instrument, start_index, end_index, freq)
    #     _calendar = Cal.calendar(freq=freq)

    #     resample_data = np.empty(end_index - start_index + 1, dtype="float32")

    #     for cur_index in range(start_index, end_index + 1):
    #         cur_time = _calendar[cur_index]
    #         # To load expression accurately, more historical data are required
    #         start_ws, end_ws = self.feature.get_extended_window_size()
    #         if end_ws > 0:
    #             raise ValueError(
    #                 "PIT database does not support referring to future period (e.g. expressions like `Ref('$$roewa_q', -1)` are not supported"
    #             )

    #         # The calculated value will always the last element, so the end_offset is zero.
    #         try:
    #             s = self._load_feature(instrument, -start_ws, 0, cur_time)
    #             resample_data[cur_index - start_index] = s.iloc[-1] if len(s) > 0 else np.nan
    #         except FileNotFoundError:
    #             get_module_logger("base").warning(f"WARN: period data not found for {str(self)}")
    #             return pd.Series(dtype="float32", name=str(self))

    #     resample_series = pd.Series(
    #         resample_data, index=pd.RangeIndex(start_index, end_index + 1), dtype="float32", name=str(self)
    #     )
    #     return resample_series
    def _load_feature(self, instrument, start_index, end_index, freq):
        return self.feature.load(instrument, start_index, end_index, freq, None)

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        data_series = self._load_feature(instrument, 0, len(_calendar), freq)
        if data_series.empty:
            return data_series
         # concat data_series index and _calendar and remove duplicates and sort
        new_calendar_index = np.unique(np.concatenate([data_series.index.tolist(), _calendar]))
        data_series = data_series.reindex(new_calendar_index)
        # ffill to fill missing values
        data_series = data_series.ffill()
        data_series = data_series.dropna()
        # slice to the calendar range
        _, _, s_index, e_index = Cal.locate_index(data_series.index.min(), data_series.index.max(), freq)
        max_start_index = max(s_index, start_index)
        min_end_index = min(e_index, end_index)
        start_time = _calendar[max_start_index]
        end_time = _calendar[min_end_index]
        data_series = data_series.loc[start_time:end_time]
        # 如果有不存在日期，去掉该日期
        data_series = data_series[data_series.index.isin(_calendar)]
        data_series.index = pd.RangeIndex(max_start_index, min_end_index+1)
        return data_series

    # def _load_feature(self, instrument, start_index, end_index, cur_time):
    #     return self.feature.load(instrument, start_index, end_index, cur_time)

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0

class PRef(P):
    def __init__(self, feature, period):
        super().__init__(feature)
        self.period = period

    def __str__(self):
        # return f"{super().__str__()}[{self.period}]"
        return f"PRef({str(self.feature)},{self.period})"

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time, self.period)
    # def _load_internal(self, instrument, start_index, end_index, freq):
    #     return self.feature.load(instrument, start_index, end_index, freq, self.period)

#################### Operator which support factor ####################
class PreFactor(ExpressionOps):
    def __init__(self, feature, feature_factora, feature_factorb):
        self.feature = feature
        self.feature_factora = feature_factora
        self.feature_factorb = feature_factorb

    def __str__(self):
        return "PreFactor({},{},{})".format(self.feature, self.feature_factora, self.feature_factorb)

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        # load data
        series = self.feature.load(instrument, 0, len(_calendar), freq)
        if series.empty:
            return series

        if isinstance(self.feature_factora, PFeature) and isinstance(self.feature_factorb, PFeature):
            series_factora = self.feature_factora.load(instrument, 0, len(_calendar), freq, None)
            series_factorb = self.feature_factorb.load(instrument, 0, len(_calendar), freq, None)
            if len(series_factora)>0 and len(series_factora) == len(series_factorb):
                
                series_factora = series_factora[(series_factora.index>=_calendar[0]) & (series_factora.index<=_calendar[-1])]
                series_factora = series_factora.reindex(_calendar)
                series_factora.index = pd.RangeIndex(0, len(_calendar))
                series_factorb = series_factorb[(series_factorb.index>=_calendar[0]) & (series_factorb.index<=_calendar[-1])]
                series_factorb = series_factorb.reindex(_calendar)
                series_factorb.index = pd.RangeIndex(0, len(_calendar))

                # shift 1 pos to align the index
                series_factora = series_factora.shift(-1)
                series_factorb = series_factorb.shift(-1)

                # calculate the factor
                factor = (series - series_factorb) / series_factora / series
                factor.iloc[-1] = 1
                # Take out non-null values, 
                # multiply them in reverse order, 
                # and then put them back in the original index order.
                # last bfill to make sure the last value is 1
                factor = factor[factor.notnull()]
                factor = factor[::-1]
                factor = factor.cumprod()
                factor = factor[::-1]
                factor = factor.reindex(pd.RangeIndex(0, len(_calendar)))
                factor = factor.bfill()
                max_start_index = max(start_index, series.index[0])
                min_end_index = min(end_index, series.index[-1])
                return factor[max_start_index:min_end_index+1]
        
        return pd.Series(np.ones_like(series), index=series.index)

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0
