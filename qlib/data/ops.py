# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from typing import Union, List, Type
from scipy.stats import percentileofscore
from .base import Expression, ExpressionOps, Feature, PFeature
from ..log import get_module_logger
from ..utils import get_callable_kwargs

try:
    from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
    from ._libs.expanding import expanding_slope, expanding_rsquare, expanding_resi
except ImportError:
    print(
        "#### Do not import qlib package in the repository directory in case of importing qlib from . without compiling #####"
    )
    raise
except ValueError:
    print("!!!!!!!! A error occurs when importing operators implemented based on Cython.!!!!!!!!")
    print("!!!!!!!! They will be disabled. Please Upgrade your numpy to enable them     !!!!!!!!")
    # We catch this error because some platform can't upgrade there package (e.g. Kaggle)
    # https://www.kaggle.com/general/293387
    # https://www.kaggle.com/product-feedback/98562


np.seterr(invalid="ignore")


#################### Element-Wise Operator ####################
class ElemOperator(ExpressionOps):
    """Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class ChangeInstrument(ElemOperator):
    """Change Instrument Operator
    In some case, one may want to change to another instrument when calculating, for example, to
    calculate beta of a stock with respect to a market index.
    This would require changing the calculation of features from the stock (original instrument) to
    the index (reference instrument)
    Parameters
    ----------
    instrument: new instrument for which the downstream operations should be performed upon.
                i.e., SH000300 (CSI300 index), or ^GPSC (SP500 index).

    feature: the feature to be calculated for the new instrument.
    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, instrument, feature):
        self.instrument = instrument
        self.feature = feature

    def __str__(self):
        return "{}('{}',{})".format(type(self).__name__, self.instrument, self.feature)

    def load(self, instrument, start_index, end_index, *args):
        # the first `instrument` is ignored
        return super().load(self.instrument, start_index, end_index, *args)

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(instrument, start_index, end_index, *args)


class NpElemOperator(ElemOperator):
    """Numpy Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    func : str
        numpy feature operation method

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature, func):
        self.func = func
        super(NpElemOperator, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return getattr(np, self.func)(series)


class Abs(NpElemOperator):
    """Feature Absolute Value

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with absolute output
    """

    def __init__(self, feature):
        super(Abs, self).__init__(feature, "abs")


class Sign(NpElemOperator):
    """Feature Sign

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(self, feature):
        super(Sign, self).__init__(feature, "sign")

    def _load_internal(self, instrument, start_index, end_index, *args):
        """
        To avoid error raised by bool type input, we transform the data into float32.
        """
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO:  More precision types should be configurable
        series = series.astype(np.float32)
        return getattr(np, self.func)(series)


class Log(NpElemOperator):
    """Feature Log

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with log
    """

    def __init__(self, feature):
        super(Log, self).__init__(feature, "log")


class Mask(NpElemOperator):
    """Feature Mask

    Parameters
    ----------
    feature : Expression
        feature instance
    instrument : str
        instrument mask

    Returns
    ----------
    Expression
        a feature instance with masked instrument
    """

    def __init__(self, feature, instrument):
        super(Mask, self).__init__(feature, "mask")
        self.instrument = instrument

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.instrument.lower())

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(self.instrument, start_index, end_index, *args)


class Not(NpElemOperator):
    """Not Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Feature:
        feature elementwise not output
    """

    def __init__(self, feature):
        super(Not, self).__init__(feature, "bitwise_not")


#################### Pair-Wise Operator ####################
class PairOperator(ExpressionOps):
    """Pair-wise operator

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
        feature instance or numeric value

    Returns
    ----------
    Feature:
        two features' operation output
    """

    def __init__(self, feature_left, feature_right):
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature_left, self.feature_right)

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        return max(ll, rl), max(lr, rr)


class NpPairOperator(PairOperator):
    """Numpy Pair-wise operator

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
        feature instance or numeric value
    func : str
        operator function

    Returns
    ----------
    Feature:
        two features' operation output
    """

    def __init__(self, feature_left, feature_right, func):
        self.func = func
        super(NpPairOperator, self).__init__(feature_left, feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, (Expression,)), self.feature_right, Expression]
        ), "at least one of two inputs is Expression instance"
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        check_length = isinstance(series_left, (np.ndarray, pd.Series)) and isinstance(
            series_right, (np.ndarray, pd.Series)
        )
        if check_length:
            warning_info = (
                f"Loading {instrument}: {str(self)}; np.{self.func}(series_left, series_right), "
                f"The length of series_left and series_right is different: ({len(series_left)}, {len(series_right)}), "
                f"series_left is {str(self.feature_left)}, series_right is {str(self.feature_right)}. Please check the data"
            )
        else:
            warning_info = (
                f"Loading {instrument}: {str(self)}; np.{self.func}(series_left, series_right), "
                f"series_left is {str(self.feature_left)}, series_right is {str(self.feature_right)}. Please check the data"
            )
        try:
            res = getattr(np, self.func)(series_left, series_right)
        except ValueError as e:
            get_module_logger("ops").debug(warning_info)
            raise ValueError(f"{str(e)}. \n\t{warning_info}") from e
        else:
            if check_length and len(series_left) != len(series_right):
                get_module_logger("ops").debug(warning_info)
        return res


class Power(NpPairOperator):
    """Power Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        The bases in feature_left raised to the exponents in feature_right
    """

    def __init__(self, feature_left, feature_right):
        super(Power, self).__init__(feature_left, feature_right, "power")

class SignedPower(NpPairOperator):
    """Signed Power Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        The signed power of feature_left raised to feature_right
    """

    def __init__(self, feature_left, feature_right):
        super(SignedPower, self).__init__(feature_left, feature_right, "signed_power")

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, (Expression,)), isinstance(self.feature_right, (Expression,))]
        ), "at least one of two inputs is Expression instance"
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        # 计算带符号的幂：sign(x) * |x|^y
        res = np.sign(series_left) * (np.abs(series_left) ** series_right)
        return res


class Sqrt(NpPairOperator):
    """Square Root Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        The square root of feature_left
    """

    def __init__(self, feature_left, feature_right):
        super(Sqrt, self).__init__(feature_left, feature_right, "sqrt")

class Add(NpPairOperator):
    """Add Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' sum
    """

    def __init__(self, feature_left, feature_right):
        super(Add, self).__init__(feature_left, feature_right, "add")


class Sub(NpPairOperator):
    """Subtract Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' subtraction
    """

    def __init__(self, feature_left, feature_right):
        super(Sub, self).__init__(feature_left, feature_right, "subtract")


class Mul(NpPairOperator):
    """Multiply Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' product
    """

    def __init__(self, feature_left, feature_right):
        super(Mul, self).__init__(feature_left, feature_right, "multiply")


class Div(NpPairOperator):
    """Division Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' division
    """

    def __init__(self, feature_left, feature_right):
        super(Div, self).__init__(feature_left, feature_right, "divide")


class Greater(NpPairOperator):
    """Greater Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        greater elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Greater, self).__init__(feature_left, feature_right, "maximum")


class Less(NpPairOperator):
    """Less Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        smaller elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Less, self).__init__(feature_left, feature_right, "minimum")


class Gt(NpPairOperator):
    """Greater Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left > right`
    """

    def __init__(self, feature_left, feature_right):
        super(Gt, self).__init__(feature_left, feature_right, "greater")


class Ge(NpPairOperator):
    """Greater Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left >= right`
    """

    def __init__(self, feature_left, feature_right):
        super(Ge, self).__init__(feature_left, feature_right, "greater_equal")


class Lt(NpPairOperator):
    """Less Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left < right`
    """

    def __init__(self, feature_left, feature_right):
        super(Lt, self).__init__(feature_left, feature_right, "less")


class Le(NpPairOperator):
    """Less Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left <= right`
    """

    def __init__(self, feature_left, feature_right):
        super(Le, self).__init__(feature_left, feature_right, "less_equal")


class Eq(NpPairOperator):
    """Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left == right`
    """

    def __init__(self, feature_left, feature_right):
        super(Eq, self).__init__(feature_left, feature_right, "equal")


class Ne(NpPairOperator):
    """Not Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left != right`
    """

    def __init__(self, feature_left, feature_right):
        super(Ne, self).__init__(feature_left, feature_right, "not_equal")


class And(NpPairOperator):
    """And Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row & output
    """

    def __init__(self, feature_left, feature_right):
        super(And, self).__init__(feature_left, feature_right, "bitwise_and")


class Or(NpPairOperator):
    """Or Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row | outputs
    """

    def __init__(self, feature_left, feature_right):
        super(Or, self).__init__(feature_left, feature_right, "bitwise_or")


class Cross(NpPairOperator):
    """Cross Operator

    Detect if feature A crosses above feature B (Golden Cross)
    Need to satisfy two conditions simultaneously:
    1. Current period: A_t > B_t
    2. Previous period: A_{t-1} <= B_{t-1}
    3. Current period: A_t <= B_t
    4. Previous period: A_{t-1} > B_{t-1}

    Parameters
    ----------
    feature_left : Expression
        feature instance (A)
    feature_right : Expression
        feature instance (B)

    Returns
    ----------
    Feature:
        bool series indicate A crosses above B
    """

    def __init__(self, feature_left, feature_right):
        super(Cross, self).__init__(feature_left, feature_right, "cross")

    def __str__(self):
        return "{}({}, {})".format(type(self).__name__, self.feature_left, self.feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, (Expression,)), isinstance(self.feature_right, (Expression,))]
        ), "at least one of two inputs is Expression instance"
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        
        # Get previous values (t-1)
        prev_left = series_left.shift(1)
        prev_right = series_right.shift(1)
        
        # Check both conditions and return the result for Golden Cross
        cond1 = series_left > series_right
        cond2 = prev_left <= prev_right

        # Check both conditions and return the result for Death Cross
        cond3 = series_left < series_right
        cond4 = prev_left >= prev_right

        # result Golden Cross value is 1 and Death Cross value is -1 other values are 0
        # 合并条件：Golden Cross 为 1，Death Cross 为 -1，其余为 0
        result = np.where(cond1 & cond2, 1,
                 np.where(cond3 & cond4, -1, 0))
        return pd.Series(result, index=series_left.index)

#################### Triple-wise Operator ####################
class If(ExpressionOps):
    """If Operator

    Parameters
    ----------
    condition : Expression
        feature instance with bool values as condition
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    """

    def __init__(self, condition, feature_left, feature_right):
        self.condition = condition
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "If({},{},{})".format(self.condition, self.feature_left, self.feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series_cond = self.condition.load(instrument, start_index, end_index, *args)
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        series = pd.Series(np.where(series_cond, series_left, series_right), index=series_cond.index)
        return series

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0

        if isinstance(self.condition, (Expression,)):
            c_br = self.condition.get_longest_back_rolling()
        else:
            c_br = 0
        return max(left_br, right_br, c_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0

        if isinstance(self.condition, (Expression,)):
            cl, cr = self.condition.get_extended_window_size()
        else:
            cl, cr = 0, 0
        return max(ll, rl, cl), max(lr, rr, cr)


#################### Rolling ####################
# NOTE: methods like `rolling.mean` are optimized with cython,
# and are super faster than `rolling.apply(np.mean)`


class Rolling(ExpressionOps):
    """Rolling Operator
    The meaning of rolling and expanding is the same in pandas.
    When the window is set to 0, the behaviour of the operator should follow `expanding`
    Otherwise, it follows `rolling`

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size
    func : str
        rolling method

    Returns
    ----------
    Expression
        rolling outputs
    """

    def __init__(self, feature, N, func):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        if isinstance(self.N, int) and self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(self.N, float) and 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
            # series.iloc[:self.N-1] = np.nan
        # series[isnull] = np.nan
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))  # (1 - N)**window == 1e-6
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self):
        if self.N == 0:
            # FIXME: How to make this accurate and efficiently? Or  should we
            # remove such support for N == 0?
            get_module_logger(self.__class__.__name__).warning("The Rolling(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        elif 0 < self.N < 1:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            size = int(np.log(1e-6) / np.log(1 - self.N))
            lft_etd = max(lft_etd + size - 1, lft_etd)
            return lft_etd, rght_etd
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N - 1, lft_etd)
            return lft_etd, rght_etd


class Ref(Rolling):
    """Feature Reference

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data

    Returns
    ----------
    Expression
        a feature instance with target reference
    """

    def __init__(self, feature, N):
        super(Ref, self).__init__(feature, N, "ref")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)  # copy
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        return self.feature.get_longest_back_rolling() + self.N

    def get_extended_window_size(self):
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning("The Ref(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N, lft_etd)
            rght_etd = max(rght_etd - self.N, rght_etd)
            return lft_etd, rght_etd

#################### FutureRolling ####################
class FutureRolling(ExpressionOps):
    """Future Rolling Window Operator

    Applies a rolling window calculation on future time periods.
    For each time point t, computes aggregation over the next N periods [t+1, t+N].

    Parameters
    ----------
    feature : Expression
        feature instance to apply the rolling window on
    N : int
        size of the forward-looking window
    func : callable
        aggregation function to apply on the window (e.g., lambda x: x.mean())

    Returns
    -------
    Expression
        a feature instance with aggregated future values

    Examples
    --------
    >>> # Calculate mean of next 5 days
    >>> FutureRolling($close, 5, lambda x: x.mean())
    >>> # Calculate max of next 10 days
    >>> FutureRolling($high, 10, lambda x: x.max())
    """

    def __init__(self, feature, N, func):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        
        if self.N == 0:
            # Window size of 0 returns the series as-is
            return series
        
        # Create result array
        result = np.empty(len(series))
        result[:] = np.nan
        
        # For each position, calculate the aggregation over the next N values
        for i in range(len(series)):
            # Get the next N values (excluding current position i)
            future_start = i + 1
            future_end = min(i + 1 + self.N, len(series))
            future_window = series.iloc[future_start:future_end]
            
            if len(future_window) >= self.N:
                # We have enough future data, use only future values
                result[i] = getattr(future_window, self.func)()
            elif len(future_window) > 0:
                # Not enough future data, use only future values
                result[i] = getattr(future_window, self.func)()
            else:
                # No future data at all, use only current value
                result[i] = getattr(np.array([series.iloc[i]]), self.func)()
            
        return pd.Series(result, index=series.index)

    def get_longest_back_rolling(self):
        # FutureRolling doesn't look backward, only forward
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        # Need to extend the right side (future) to get enough data
        # Since we shift(-1) and need N values, we need N additional future points
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        rght_etd = max(rght_etd + self.N, rght_etd)
        return lft_etd, rght_etd

class FutureMax(FutureRolling):
    def __init__(self, feature, N):
        super(FutureMax, self).__init__(feature, N, "max")

class FutureMin(FutureRolling):
    def __init__(self, feature, N):
        super(FutureMin, self).__init__(feature, N, "min")

class FutureMean(FutureRolling):
    def __init__(self, feature, N):
        super(FutureMean, self).__init__(feature, N, "mean")

class FutureMed(FutureRolling):
    def __init__(self, feature, N):
        super(FutureMed, self).__init__(feature, N, "median")   

class FutureStd(FutureRolling):
    def __init__(self, feature, N):
        super(FutureStd, self).__init__(feature, N, "std") 

class FutureVar(FutureRolling):
    def __init__(self, feature, N):
        super(FutureVar, self).__init__(feature, N, "var") 

class FutureSum(FutureRolling):
    def __init__(self, feature, N):
        super(FutureSum, self).__init__(feature, N, "sum") 

#################### TripleBarrier ####################
class TripleBarrier(ExpressionOps):
    """Triple Barrier Method Operator

    Labels each time point based on which barrier is hit first in the future:
    - Returns 1 if upper barrier is hit first (bullish signal)
    - Returns -1 if lower barrier is hit first (bearish signal)
    - Returns 0 if time barrier is hit first without hitting price barriers (neutral)

    This is a commonly used labeling method in quantitative finance for supervised learning,
    particularly in Lopez de Prado's "Advances in Financial Machine Learning".

    Parameters
    ----------
    feature : Expression
        price feature to monitor (typically close price)
    upper : float or Expression
        upper barrier threshold (can be a numeric value or an Expression for dynamic barriers)
    lower : float or Expression
        lower barrier threshold (can be a numeric value or an Expression for dynamic barriers)
    time_barrier : int
        maximum number of periods to wait before giving up
    use_percentage : bool, default=True
        if True, barriers represent percentage changes (e.g., 0.02 = 2%)
        if False, barriers represent absolute price changes

    Returns
    -------
    Expression
        Series with labels: 1 (upper barrier hit), -1 (lower barrier hit), 0 (time barrier hit)

    Examples
    --------
    >>> # Label with 2% up/down barriers, 5-day time limit
    >>> TripleBarrier($close, 0.02, 0.02, 5, use_percentage=True)
    >>> # Label with $3 absolute barriers, 10-day time limit
    >>> TripleBarrier($close, 3, 3, 10, use_percentage=False)
    >>> # Dynamic barriers using volatility (e.g., 2x ATR)
    >>> atr = Mean(($high - $low), 14)  # Simple ATR approximation
    >>> TripleBarrier($close, atr * 2, atr * 2, 5, use_percentage=False)
    """

    def __init__(self, feature, upper, lower, time_barrier, use_percentage):
        self.feature = feature
        self.upper = upper
        self.lower = lower
        self.time_barrier = time_barrier
        self.use_percentage = use_percentage

    def __str__(self):
        return "{}({},{},{},{},{})".format(
            type(self).__name__, 
            self.feature, 
            self.upper, 
            self.lower, 
            self.time_barrier,
            self.use_percentage
        )

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        
        # Load upper and lower barriers (they can be Expression objects or numeric values)
        if isinstance(self.upper, Expression):
            upper_series = self.upper.load(instrument, start_index, end_index, *args)
        else:
            upper_series = self.upper  # numeric value
            
        if isinstance(self.lower, Expression):
            lower_series = self.lower.load(instrument, start_index, end_index, *args)
        else:
            lower_series = self.lower  # numeric value
        
        # Initialize result array with zeros (time barrier hit by default)
        result = np.zeros(len(series))
        
        # For each time point, check future prices
        for i in range(len(series)):
            current_price = series.iloc[i]
            
            # Get barrier values for this time point
            if isinstance(upper_series, (pd.Series, np.ndarray)):
                upper_val = upper_series.iloc[i] if isinstance(upper_series, pd.Series) else upper_series[i]
            else:
                upper_val = upper_series
                
            if isinstance(lower_series, (pd.Series, np.ndarray)):
                lower_val = lower_series.iloc[i] if isinstance(lower_series, pd.Series) else lower_series[i]
            else:
                lower_val = lower_series
            
            # Get future window (up to time_barrier periods ahead)
            future_end = min(i + self.time_barrier + 1, len(series))
            future_prices = series.iloc[i+1:future_end]
            
            if len(future_prices) == 0:
                # No future data available
                result[i] = 0
                continue
            
            # Calculate barriers based on current price
            if self.use_percentage:
                upper_barrier = current_price * (1 + upper_val)
                lower_barrier = current_price * (1 - lower_val)
            else:
                upper_barrier = current_price + upper_val
                lower_barrier = current_price - lower_val
            
            # Check which barrier is hit first
            upper_hit = future_prices >= upper_barrier
            lower_hit = future_prices <= lower_barrier
            
            # Find first occurrence of each barrier
            upper_idx = np.where(upper_hit)[0]
            lower_idx = np.where(lower_hit)[0]
            
            if len(upper_idx) > 0 and len(lower_idx) > 0:
                # Both barriers were hit, check which came first
                if upper_idx[0] < lower_idx[0]:
                    result[i] = 1  # Upper barrier hit first
                else:
                    result[i] = -1  # Lower barrier hit first
            elif len(upper_idx) > 0:
                # Only upper barrier was hit
                result[i] = 1
            elif len(lower_idx) > 0:
                # Only lower barrier was hit
                result[i] = -1
            else:
                # Neither barrier was hit within time limit
                result[i] = 0
        
        return pd.Series(result, index=series.index)

    def get_longest_back_rolling(self):
        # TripleBarrier looks forward, not backward
        # But need to consider dependencies from upper/lower if they are expressions
        back_rolling = self.feature.get_longest_back_rolling()
        
        if isinstance(self.upper, Expression):
            back_rolling = max(back_rolling, self.upper.get_longest_back_rolling())
        if isinstance(self.lower, Expression):
            back_rolling = max(back_rolling, self.lower.get_longest_back_rolling())
            
        return back_rolling

    def get_extended_window_size(self):
        # Need to extend the right side (future) to check future barriers
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        
        # Also consider window sizes from upper/lower if they are expressions
        if isinstance(self.upper, Expression):
            upper_lft, upper_rght = self.upper.get_extended_window_size()
            lft_etd = max(lft_etd, upper_lft)
            rght_etd = max(rght_etd, upper_rght)
            
        if isinstance(self.lower, Expression):
            lower_lft, lower_rght = self.lower.get_extended_window_size()
            lft_etd = max(lft_etd, lower_lft)
            rght_etd = max(rght_etd, lower_rght)
        
        # Need future data for time_barrier
        rght_etd = max(rght_etd + self.time_barrier, rght_etd)
        return lft_etd, rght_etd


# Alias for Triple Barrier Method
TBM = TripleBarrier


class Mean(Rolling):
    """Rolling Mean (MA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling average
    """

    def __init__(self, feature, N):
        super(Mean, self).__init__(feature, N, "mean")


class Sum(Rolling):
    """Rolling Sum

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling sum
    """

    def __init__(self, feature, N):
        super(Sum, self).__init__(feature, N, "sum")


class Std(Rolling):
    """Rolling Std

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling std
    """

    def __init__(self, feature, N):
        super(Std, self).__init__(feature, N, "std")


class Var(Rolling):
    """Rolling Variance

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling variance
    """

    def __init__(self, feature, N):
        super(Var, self).__init__(feature, N, "var")


class Skew(Rolling):
    """Rolling Skewness

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling skewness
    """

    def __init__(self, feature, N):
        if N != 0 and N < 3:
            raise ValueError("The rolling window size of Skewness operation should >= 3")
        super(Skew, self).__init__(feature, N, "skew")


class Kurt(Rolling):
    """Rolling Kurtosis

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling kurtosis
    """

    def __init__(self, feature, N):
        if N != 0 and N < 4:
            raise ValueError("The rolling window size of Kurtosis operation should >= 5")
        super(Kurt, self).__init__(feature, N, "kurt")


class Max(Rolling):
    """Rolling Max

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max
    """

    def __init__(self, feature, N):
        super(Max, self).__init__(feature, N, "max")


class IdxMax(Rolling):
    """Rolling Max Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max index
    """

    def __init__(self, feature, N):
        super(IdxMax, self).__init__(feature, N, "idxmax")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class Min(Rolling):
    """Rolling Min

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min
    """

    def __init__(self, feature, N):
        super(Min, self).__init__(feature, N, "min")


class IdxMin(Rolling):
    """Rolling Min Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min index
    """

    def __init__(self, feature, N):
        super(IdxMin, self).__init__(feature, N, "idxmin")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class Quantile(Rolling):
    """Rolling Quantile

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling quantile
    """

    def __init__(self, feature, N, qscore):
        super(Quantile, self).__init__(feature, N, "quantile")
        self.qscore = qscore

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.qscore)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series


class Med(Rolling):
    """Rolling Median

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling median
    """

    def __init__(self, feature, N):
        super(Med, self).__init__(feature, N, "median")


class Mad(Rolling):
    """Rolling Mean Absolute Deviation

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling mean absolute deviation
    """

    def __init__(self, feature, N):
        super(Mad, self).__init__(feature, N, "mad")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO: implement in Cython

        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(mad, raw=True)
        return series


class Rank(Rolling):
    """Rolling Rank (Percentile)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling rank
    """

    def __init__(self, feature, N):
        super(Rank, self).__init__(feature, N, "rank")

    # for compatiblity of python 3.7, which doesn't support pandas 1.4.0+ which implements Rolling.rank
    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        rolling_or_expending = series.expanding(min_periods=1) if self.N == 0 else series.rolling(self.N, min_periods=1)
        if hasattr(rolling_or_expending, "rank"):
            return rolling_or_expending.rank(pct=True)

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100

        return rolling_or_expending.apply(rank, raw=True)


class Count(Rolling):
    """Rolling Count

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling count of number of non-NaN elements
    """

    def __init__(self, feature, N):
        super(Count, self).__init__(feature, N, "count")


class Delta(Rolling):
    """Rolling Delta

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with end minus start in rolling window
    """

    def __init__(self, feature, N):
        super(Delta, self).__init__(feature, N, "delta")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(self.N)
        return series


# TODO:
# support pair-wise rolling like `Slope(A, B, N)`
class Slope(Rolling):
    """Rolling Slope
    This operator calculate the slope between `idx` and `feature`.
    (e.g. [<feature_t1>, <feature_t2>, <feature_t3>] and [1, 2, 3])

    Usage Example:
    - "Slope($close, %d)/$close"

    # TODO:
    # Some users may want pair-wise rolling like `Slope(A, B, N)`

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression slope of given window
    """

    def __init__(self, feature, N):
        super(Slope, self).__init__(feature, N, "slope")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, self.N), index=series.index)
        return series


class Rsquare(Rolling):
    """Rolling R-value Square

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(Rsquare, self).__init__(feature, N, "rsquare")

    def _load_internal(self, instrument, start_index, end_index, *args):
        _series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series


class Resi(Rolling):
    """Rolling Regression Residuals

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression residuals of given window
    """

    def __init__(self, feature, N):
        super(Resi, self).__init__(feature, N, "resi")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series


class WMA(Rolling):
    """Rolling WMA

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with weighted moving average output
    """

    def __init__(self, feature, N):
        super(WMA, self).__init__(feature, N, "wma")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO: implement in Cython

        def weighted_mean(x):
            w = np.arange(len(x)) + 1
            w = w / w.sum()
            return np.nanmean(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(weighted_mean, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(weighted_mean, raw=True)
        return series


class EMA(Rolling):
    """Rolling Exponential Mean (EMA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int, float
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(EMA, self).__init__(feature, N, "ema")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def exp_weighted_mean(x):
            a = 1 - 2 / (1 + len(x))
            w = a ** np.arange(len(x))[::-1]
            w /= w.sum()
            return np.nansum(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(exp_weighted_mean, raw=True)
        elif 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = series.ewm(span=self.N, min_periods=1).mean()
        return series

class DecayLinear(Rolling):
    """
    Rolling Decay Linear
    """
    def __init__(self, feature, N):
        super(DecayLinear, self).__init__(feature, N, "decay_linear")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        weights = np.arange(self.N) + 1
        sum_weights = weights.sum()
        if self.N == 0:
            series = series
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: np.sum(weights*x)/sum_weights, raw=True)
        return series

#################### Pair-Wise Rolling ####################
class PairRolling(ExpressionOps):
    """Pair Rolling Operator

    Parameters
    ----------
    feature_left : Expression©
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling output of two input features
    """

    def __init__(self, feature_left, feature_right, N, func):
        # TODO: in what case will a const be passed into `__init__` as `feature_left` or `feature_right`
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature_left, self.feature_right, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, Expression), self.feature_right, Expression]
        ), "at least one of two inputs is Expression instance"

        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right

        if self.N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(self.N, min_periods=1), self.func)(series_right)
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if isinstance(self.feature_left, Expression):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, Expression):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, Expression):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0
        if isinstance(self.feature_right, Expression):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning(
                "The PairRolling(ATTR, 0) will not be accurately calculated"
            )
            return -np.inf, max(lr, rr)
        else:
            return max(ll, rl) + self.N - 1, max(lr, rr)


class Corr(PairRolling):
    """Rolling Correlation

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling correlation of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(Corr, self).__init__(feature_left, feature_right, N, "corr")

    def _load_internal(self, instrument, start_index, end_index, *args):
        res: pd.Series = super(Corr, self)._load_internal(instrument, start_index, end_index, *args)

        # NOTE: Load uses MemCache, so calling load again will not cause performance degradation
        series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res


class Cov(PairRolling):
    """Rolling Covariance

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(Cov, self).__init__(feature_left, feature_right, N, "cov")



#################### cross section operator ####################
class XSectionOperator(ElemOperator):
    """Base class for cross section operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with cross section operation of input feature
    """
    producer_instrument = {}

    def set_population(self, population):
        super(XSectionOperator, self).set_population(population)
        population_sorted = sorted(population)
        if str(self) not in self.producer_instrument:
            self.producer_instrument[str(self)] = population_sorted[
                len(self.producer_instrument) % len(population_sorted)
            ]

    def _process_df(self, df, **_) -> pd.DataFrame:
        raise NotImplementedError("This function must be implemented in your newly defined feature")
        
    def _load_internal(self, instrument, start_index, end_index, *args) -> pd.Series:
        from .cache import H  # pylint: disable=C0415

        cache_key = str(self), instrument, start_index, end_index, *args

        if cache_key not in H["fs"]:
            # get_module_logger(self.__class__.__name__).info(f"Acquiring lock {id(H['fs'].locks[str(self)])} for {str(self)} in {os.getpid()}")
            H["cs_rlock_dict"][str(self)].acquire()
            try:
                if cache_key not in H["fs"]:
                    # get_module_logger(self.__class__.__name__).info(f"calculating: {str(self)} for instrument {instrument}")
                    df, inst_ranges = self._load_all_instruments(start_index, end_index, *args)
                    df = self._process_df(df)

                    for inst in df.columns:
                        inst_st, inst_ed = inst_ranges.get(inst, (None, None))
                        inst_cache_key = str(self), inst, start_index, end_index, *args
                        inst_st = start_index if inst_st is None else max(inst_st, start_index)
                        inst_ed = end_index if inst_ed is None else min(inst_ed, end_index)

                        H["fs"][inst_cache_key] = df.loc[inst_st:inst_ed, inst]
                # else:
                #     get_module_logger(self.__class__.__name__).info(f"cache hit after waiting: {str(self)}")
            finally:
                # get_module_logger(self.__class__.__name__).info(f"Release lock {id(H['fs'].locks[str(self)])} for {str(self)} in {os.getpid()}")
                H["cs_rlock_dict"][str(self)].release()

        return H["fs"][cache_key]

    def _load_all_instruments(self, start_index, end_index, *args) -> pd.DataFrame:
        if isinstance(getattr(self, "population"), dict):

            def mask_data(series, spans):
                if bool(spans) and not series.empty:
                    mask = np.zeros(len(series), dtype=bool)
                    for begin, end in spans:
                        mask |= (series.index >= begin) & (series.index <= end)
                    series = series.copy()
                    series[~mask] = np.nan
                return series

            sub_features = [
                mask_data(self.feature.load(inst, start_index, end_index, *args).rename(inst), spans)
                for inst, spans in getattr(self, "population", {}).items()
            ]
        else:
            sub_features = [
                self.feature.load(inst, start_index, end_index, *args).rename(inst)
                for inst in getattr(self, "population", [])
            ]
        mydf = pd.concat([s for s in sub_features if not s.empty], axis=1, join="outer", sort=True)
        inst_ranges = {s.name: (s.index.min(), s.index.max()) for s in sub_features if not s.empty}

        return mydf, inst_ranges

    @property
    def require_cs_info(self):
        return True

class CSRank(XSectionOperator):
    """Cross section rank

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with cross section rank of input feature
    """
    def _process_df(self, df, **_) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

class CSBin(XSectionOperator):
    """Cross section bin

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with cross section bin of input feature
    """
    def __init__(self, feature, N_BINS=3):
        self.feature = feature
        self.N_BINS = N_BINS
        super().__init__(feature)

    def _process_df(self, df, **_) -> pd.DataFrame:
        def row_equal_width_cut(row):
            """
            对单行数据（Series）执行等寬分箱 (pd.cut)。
            """
            # 1. 对该行所有值进行 pd.cut 操作
            # bins=N_BINS (例如 3) 会在行内最大值和最小值之间创建 3 个等宽区间
            # labels=False 返回分桶的数字编号 (0, 1, 2)
            binned_labels = pd.cut(
                row,               # 输入的是该行的所有数值
                bins=self.N_BINS,       # 划分的组数
                labels=False,      # 返回数字标签 (0, 1, 2...)
                include_lowest=True # 确保包含最小值
            )
            
            # +1 是为了使标签从 1 开始 (1, 2, 3)
            return binned_labels + 1

        return df.apply(row_equal_width_cut, axis=1)

#################### Operator which only support data with time index ####################
# Convention
# - The name of the operators in this section will start with "T"


class TResample(ElemOperator):
    def __init__(self, feature, freq, func):
        """
        Resampling the data to target frequency.
        The resample function of pandas is used.

        - the timestamp will be at the start of the time span after resample.

        Parameters
        ----------
        feature : Expression
            An expression for calculating the feature
        freq : str
            It will be passed into the resample method for resampling basedn on given frequency
        func : method
            The method to get the resampled values
            Some expression are high frequently used
        """
        self.feature = feature
        self.freq = freq
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.freq)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        if series.empty:
            return series
        else:
            if self.func == "sum":
                return getattr(series.resample(self.freq), self.func)(min_count=1)
            else:
                return getattr(series.resample(self.freq), self.func)()

CSOpsList = [CSRank]
TOpsList = [TResample]
AdditionList = [DecayLinear, SignedPower]
FutureOpsList = [FutureRolling, FutureSum, FutureMed, FutureMean, FutureVar, FutureStd, FutureMax, FutureMin]
OpsList = [
    ChangeInstrument,
    Rolling,
    Ref,
    TripleBarrier,
    Max,
    Min,
    Sum,
    Mean,
    Std,
    Var,
    Skew,
    Kurt,
    Med,
    Mad,
    Slope,
    Rsquare,
    Resi,
    Rank,
    Quantile,
    Count,
    EMA,
    WMA,
    Corr,
    Cov,
    Delta,
    Abs,
    Sign,
    Log,
    Power,
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    And,
    Or,
    Cross,
    Not,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
    Mask,
    IdxMax,
    IdxMin,
    If,
    Feature,
    PFeature,
] + [TResample] + CSOpsList + AdditionList + FutureOpsList


class OpsWrapper:
    """Ops Wrapper"""

    def __init__(self):
        self._ops = {}

    def reset(self):
        self._ops = {}

    def register(self, ops_list: List[Union[Type[ExpressionOps], dict]]):
        """register operator

        Parameters
        ----------
        ops_list : List[Union[Type[ExpressionOps], dict]]
            - if type(ops_list) is List[Type[ExpressionOps]], each element of ops_list represents the operator class, which should be the subclass of `ExpressionOps`.
            - if type(ops_list) is List[dict], each element of ops_list represents the config of operator, which has the following format:

                .. code-block:: text

                    {
                        "class": class_name,
                        "module_path": path,
                    }

                Note: `class` should be the class name of operator, `module_path` should be a python module or path of file.
        """
        for _operator in ops_list:
            if isinstance(_operator, dict):
                _ops_class, _ = get_callable_kwargs(_operator)
            else:
                _ops_class = _operator

            if not issubclass(_ops_class, (Expression,)):
                raise TypeError("operator must be subclass of ExpressionOps, not {}".format(_ops_class))

            if _ops_class.__name__ in self._ops:
                get_module_logger(self.__class__.__name__).warning(
                    "The custom operator [{}] will override the qlib default definition".format(_ops_class.__name__)
                )
            self._ops[_ops_class.__name__] = _ops_class

    def __getattr__(self, key):
        if key not in self._ops:
            raise AttributeError("The operator [{0}] is not registered".format(key))
        return self._ops[key]


Operators = OpsWrapper()


def register_all_ops(C):
    """register all operator"""
    logger = get_module_logger("ops")

    from qlib.data.pit import P, PRef  # pylint: disable=C0415

    Operators.reset()
    Operators.register(OpsList + [P, PRef])

    if getattr(C, "custom_ops", None) is not None:
        Operators.register(C.custom_ops)
        logger.debug("register custom operator {}".format(C.custom_ops))
