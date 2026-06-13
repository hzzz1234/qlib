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

logger = get_module_logger("data")
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

class FR(ExpressionOps):
    """Forward Ratio
    """
    def __init__(self, feature, feature_dr):
        self.feature = feature
        self.feature_dr = feature_dr
        
    def __str__(self):
        return "FR({},{})".format(self.feature, self.feature_dr)

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        data_series = self.feature.load(instrument, start_index, end_index, freq)
        if data_series.empty:
            return data_series

        if isinstance(self.feature_dr, PFeature):
            series_dr = self.feature_dr.load(instrument, start_index, end_index, freq, None)
            if series_dr.empty:
                return data_series
            #
            # Product of the series_dr
            factor = series_dr.cumprod()
            factor = factor / factor.iloc[-1]
            
            data_start_index = data_series.index[0]
            data_end_index = data_series.index[-1]

            data_start_datetime = _calendar[data_start_index]

            if factor.index.min() < data_start_datetime:
                filtered_index = factor.index[factor.index<data_start_datetime]
                factor = factor.reindex(filtered_index.union(_calendar[data_start_index:data_end_index+1]))
                factor.ffill(inplace=True)
                factor = factor[factor.index>=data_start_datetime]
                factor.index = pd.RangeIndex(data_start_index, data_end_index+1)
            else:
                factor = factor.reindex(_calendar[data_start_index:data_end_index+1])
                factor.ffill(inplace=True)
                factor.index = pd.RangeIndex(data_start_index, data_end_index+1)

            if len(factor) != len(data_series):
                assert "data_series and factor length do not match"
            data_series = factor * data_series
        return data_series

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0

class BR(ExpressionOps):
    """Backward Ratio
    """
    def __init__(self, feature, feature_dr):
        self.feature = feature
        self.feature_dr = feature_dr
        
    def __str__(self):
        return "BR({},{})".format(self.feature, self.feature_dr)

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        data_series = self.feature.load(instrument, start_index, end_index, freq)
        if data_series.empty:
            return data_series

        if isinstance(self.feature_dr, PFeature):
            series_dr = self.feature_dr.load(instrument, start_index, end_index, freq, None)
            if series_dr.empty:
                return data_series
            
            factor = series_dr.cumprod()

            data_start_index = data_series.index[0]
            data_end_index = data_series.index[-1]

            data_start_datetime = _calendar[data_start_index]

            if factor.index.min() < data_start_datetime:
                filtered_index = factor.index[factor.index<data_start_datetime]
                factor = factor.reindex(filtered_index.union(_calendar[data_start_index:data_end_index+1]))
                factor.ffill(inplace=True)
                factor = factor[factor.index>=data_start_datetime]
                factor.index = pd.RangeIndex(data_start_index, data_end_index+1)
            else:
                factor = factor.reindex(_calendar[data_start_index:data_end_index+1])
                factor.ffill(inplace=True)
                factor.index = pd.RangeIndex(data_start_index, data_end_index+1)

            if len(factor) != len(data_series):
                assert "data_series and factor length do not match"
            data_series = factor * data_series
        return data_series

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0


class FAdjust(ExpressionOps):
    """forward adjust
    """
    def __init__(self, feature, interest, allotPrice, allotNum, stockBonus, stockGift):
        self.feature = feature
        self.interest = interest
        self.allotPrice = allotPrice
        self.allotNum = allotNum
        self.stockBonus = stockBonus
        self.stockGift = stockGift
        
    def __str__(self):
        return "FAdjust({},{},{},{},{},{})".format(self.feature, self.interest, self.allotPrice, self.allotNum, self.stockBonus, self.stockGift)

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        data_series = self.feature.load(instrument, start_index, end_index, freq)
        if data_series.empty:
            return data_series
        # cached feature buffer may be read-only; in-place adjust needs a writable copy
        data_series = data_series.copy()

        if isinstance(self.interest, PFeature) and isinstance(self.allotPrice, PFeature) and isinstance(self.allotNum, PFeature) and isinstance(self.stockBonus, PFeature) and isinstance(self.stockGift, PFeature):
            interest = self.interest.load(instrument, start_index, end_index, freq, None)
            allotPrice = self.allotPrice.load(instrument, start_index, end_index, freq, None)
            allotNum = self.allotNum.load(instrument, start_index, end_index, freq, None)
            stockBonus = self.stockBonus.load(instrument, start_index, end_index, freq, None)
            stockGift = self.stockGift.load(instrument, start_index, end_index, freq, None)
            
            if interest.empty or allotPrice.empty or allotNum.empty or stockBonus.empty or stockGift.empty:
                # logger.warning("FAdjust: interest, allotPrice, allotNum, stockBonus, stockGift is empty")
                return data_series

            dividend_df = pd.concat([interest, allotPrice, allotNum, stockBonus, stockGift], axis=1)
            dividend_df.columns = ['interest','allotPrice','allotNum', 'stockBonus', 'stockGift']

            index_list = []
            if dividend_df.index.min() < _calendar[0]:
                less_datetime_indexes = dividend_df.index[dividend_df.index<_calendar[0]]
                
                start_index = -len(less_datetime_indexes)
                for i in range(len(less_datetime_indexes)):
                    index_list.append(start_index)
                    start_index += 1
                
            greater_datetime_indexes = dividend_df.index[dividend_df.index>=_calendar[0]]
            for greater_datetime_index in greater_datetime_indexes:
                _,_,s_index,_ = Cal.locate_index(greater_datetime_index,greater_datetime_index,freq)
                index_list.append(s_index)

            dividend_df['Index'] = index_list
            dividend_df = dividend_df.set_index('Index')

            for idx, row in dividend_df.iterrows():
                mask = data_series.index < idx
                if not mask.any():
                    continue
                interest = float(row["interest"])
                allot_num = float(row["allotNum"])
                allot_price = float(row["allotPrice"])
                stock_bonus = float(row["stockBonus"])
                stock_gift = float(row["stockGift"])
                factor = 1.0 + allot_num + stock_bonus + stock_gift
                data_series.loc[mask] = (
                    data_series.loc[mask] - interest + allot_num * allot_price
                ) / factor
            
        return data_series

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0

class BAdjust(ExpressionOps):
    """forward adjust
    """
    def __init__(self, feature, interest, allotPrice, allotNum, stockBonus, stockGift):
        self.feature = feature
        self.interest = interest
        self.allotPrice = allotPrice
        self.allotNum = allotNum
        self.stockBonus = stockBonus
        self.stockGift = stockGift
        
    def __str__(self):
        return "BAdjust({},{},{},{},{},{})".format(self.feature, self.interest, self.allotPrice, self.allotNum, self.stockBonus, self.stockGift)

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        data_series = self.feature.load(instrument, start_index, end_index, freq)
        if data_series.empty:
            return data_series
        # cached feature buffer may be read-only; in-place adjust needs a writable copy
        data_series = data_series.copy()

        if isinstance(self.interest, PFeature) and isinstance(self.allotPrice, PFeature) and isinstance(self.allotNum, PFeature) and isinstance(self.stockBonus, PFeature) and isinstance(self.stockGift, PFeature):
            interest = self.interest.load(instrument, start_index, end_index, freq, None)
            allotPrice = self.allotPrice.load(instrument, start_index, end_index, freq, None)
            allotNum = self.allotNum.load(instrument, start_index, end_index, freq, None)
            stockBonus = self.stockBonus.load(instrument, start_index, end_index, freq, None)
            stockGift = self.stockGift.load(instrument, start_index, end_index, freq, None)
            
            if interest.empty or allotPrice.empty or allotNum.empty or stockBonus.empty or stockGift.empty:
                # logger.warning("FAdjust: interest, allotPrice, allotNum, stockBonus, stockGift is empty")
                return data_series

            dividend_df = pd.concat([interest, allotPrice, allotNum, stockBonus, stockGift], axis=1)
            dividend_df.columns = ['interest','allotPrice','allotNum', 'stockBonus', 'stockGift']

            index_list = []
            if dividend_df.index.min() < _calendar[0]:
                less_datetime_indexes = dividend_df.index[dividend_df.index<_calendar[0]]
                
                start_index = -len(less_datetime_indexes)
                for i in range(len(less_datetime_indexes)):
                    index_list.append(start_index)
                    start_index += 1
                
            greater_datetime_indexes = dividend_df.index[dividend_df.index>=_calendar[0]]
            for greater_datetime_index in greater_datetime_indexes:
                _,_,s_index,_ = Cal.locate_index(greater_datetime_index,greater_datetime_index,freq)
                index_list.append(s_index)

            dividend_df['Index'] = index_list
            dividend_df = dividend_df.set_index('Index')

            for idx, row in dividend_df[::-1].iterrows():
                mask = data_series.index >= idx
                if not mask.any():
                    continue
                interest = float(row["interest"])
                allot_num = float(row["allotNum"])
                allot_price = float(row["allotPrice"])
                stock_bonus = float(row["stockBonus"])
                stock_gift = float(row["stockGift"])
                bias = interest - allot_num * allot_price
                data_series.loc[mask] = (
                    data_series.loc[mask] * (1 + stock_gift + stock_bonus + allot_num) + bias
                )
            
        return data_series

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0