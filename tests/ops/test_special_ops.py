import unittest
import numpy as np
import pandas as pd

from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.ops import ChangeInstrument, Cov, Feature, Ref, Var, TripleBarrier
from qlib.tests import TestOperatorData


class TestOperatorDataSetting(TestOperatorData):
    def test_setting(self):
        # All the query below passes
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close)"])

        # get market return for "SH600519"
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', Feature('close')/Ref(Feature('close'),1) -1)"])
        df = D.features(["SH600519"], ["ChangeInstrument('SH000300', $close/Ref($close,1) -1)"])
        # excess return
        df = D.features(
            ["SH600519"], ["($close/Ref($close,1) -1) - ChangeInstrument('SH000300', $close/Ref($close,1) -1)"]
        )
        print(df)

    def test_case2(self):
        def test_case(instruments, queries, note=None):
            if note:
                print(note)
            print(f"checking {instruments} with queries {queries}")
            df = D.features(instruments, queries)
            print(df)
            return df

        test_case(["SH600519"], ["ChangeInstrument('SH000300', $close)"], "get market index close")
        test_case(
            ["SH600519"],
            ["ChangeInstrument('SH000300', Feature('close')/Ref(Feature('close'),1) -1)"],
            "get market index return with Feature",
        )
        test_case(
            ["SH600519"],
            ["ChangeInstrument('SH000300', $close/Ref($close,1) -1)"],
            "get market index return with expression",
        )
        test_case(
            ["SH600519"],
            ["($close/Ref($close,1) -1) - ChangeInstrument('SH000300', $close/Ref($close,1) -1)"],
            "get excess return with expression with beta=1",
        )

        ret = "Feature('close') / Ref(Feature('close'), 1) - 1"
        benchmark = "SH000300"
        n_period = 252
        marketRet = f"ChangeInstrument('{benchmark}', Feature('close') / Ref(Feature('close'), 1) - 1)"
        marketVar = f"ChangeInstrument('{benchmark}', Var({marketRet}, {n_period}))"
        beta = f"Cov({ret}, {marketRet}, {n_period}) / {marketVar}"
        excess_return = f"{ret} - {beta}*({marketRet})"
        fields = [
            "Feature('close')",
            f"ChangeInstrument('{benchmark}', Feature('close'))",
            ret,
            marketRet,
            beta,
            excess_return,
        ]
        test_case(["SH600519"], fields[5:], "get market beta and excess_return with estimated beta")

        instrument = "sh600519"
        ret = Feature("close") / Ref(Feature("close"), 1) - 1
        benchmark = "sh000300"
        n_period = 252
        marketRet = ChangeInstrument(benchmark, Feature("close") / Ref(Feature("close"), 1) - 1)
        marketVar = ChangeInstrument(benchmark, Var(marketRet, n_period))
        beta = Cov(ret, marketRet, n_period) / marketVar
        fields = [
            Feature("close"),
            ChangeInstrument(benchmark, Feature("close")),
            ret,
            marketRet,
            beta,
            ret - beta * marketRet,
        ]
        names = ["close", "marketClose", "ret", "marketRet", f"beta_{n_period}", "excess_return"]
        data_loader_config = {"feature": (fields, names)}
        data_loader = QlibDataLoader(config=data_loader_config)
        df = data_loader.load(instruments=[instrument])  # , start_time=start_time)
        print(df)

        # test_case(["sh600519"],fields,
        # "get market beta and excess_return with estimated beta")


class TestTripleBarrier(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的价格序列用于测试
        self.dates = pd.date_range(start='2020-01-01', periods=10)
        self.prices = pd.Series([100, 101, 103, 102, 105, 104, 106, 108, 107, 109], index=self.dates)
        
        # 模拟Feature对象
        class MockFeature:
            def __init__(self, prices):
                self.prices = prices
            
            def load(self, instrument, start_index, end_index, *args):
                return self.prices
            
            def get_longest_back_rolling(self):
                return 0
            
            def get_extended_window_size(self):
                return 0, 0
        
        self.mock_feature = MockFeature(self.prices)
    
    def test_upper_barrier_hit(self):
        """测试上边界被触发的情况"""
        # 创建TripleBarrier算子，上边界为3%，下边界为3%，时间边界为3
        tb = TripleBarrier(self.mock_feature, 0.03, 0.03, 3, use_percentage=True)
        
        # 调用_load_internal方法
        result = tb._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，第2天价格是103，上涨了3%，触发上边界，应该标记为1
        self.assertEqual(result.iloc[0], 1)
    
    def test_lower_barrier_hit(self):
        """测试下边界被触发的情况"""
        # 修改价格序列，制造一个下跌触发的情况
        prices = pd.Series([100, 101, 98, 97, 105, 104, 106, 108, 107, 109], index=self.dates)
        self.mock_feature.prices = prices
        
        # 创建TripleBarrier算子，上边界为3%，下边界为2%，时间边界为3
        tb = TripleBarrier(self.mock_feature, 0.03, 0.02, 3, use_percentage=True)
        
        # 调用_load_internal方法
        result = tb._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，第2天价格是98，下跌了2%，触发下边界，应该标记为-1
        self.assertEqual(result.iloc[0], -1)
    
    def test_time_barrier_hit(self):
        """测试时间边界被触发的情况"""
        # 修改价格序列，制造一个在时间边界内没有触发价格边界的情况
        prices = pd.Series([100, 101, 101, 101, 105, 104, 106, 108, 107, 109], index=self.dates)
        self.mock_feature.prices = prices
        
        # 创建TripleBarrier算子，上边界为2%，下边界为2%，时间边界为3
        tb = TripleBarrier(self.mock_feature, 0.02, 0.02, 3, use_percentage=True)
        
        # 调用_load_internal方法
        result = tb._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，接下来3天的价格都在[98, 102]范围内，应该标记为0
        self.assertEqual(result.iloc[0], 0)
    
    def test_absolute_value_mode(self):
        """测试使用绝对值模式的情况"""
        # 创建TripleBarrier算子，上边界为3（绝对价值），下边界为3（绝对价值），时间边界为3
        tb = TripleBarrier(self.mock_feature, 3, 3, 3, use_percentage=False)
        
        # 调用_load_internal方法
        result = tb._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，第2天价格是103，达到上边界3，应该标记为1
        self.assertEqual(result.iloc[0], 1)
    
    def test_different_barrier_orders(self):
        """测试不同边界触发顺序的情况"""
        # 修改价格序列，先小幅下跌后大幅上涨
        prices = pd.Series([100, 99, 102, 101, 105, 104, 106, 108, 107, 109], index=self.dates)
        self.mock_feature.prices = prices
        
        # 创建TripleBarrier算子，上边界为2%，下边界为2%，时间边界为3
        tb = TripleBarrier(self.mock_feature, 0.02, 0.02, 3, use_percentage=True)
        
        # 调用_load_internal方法
        result = tb._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，第1天价格是99（-1%，未触发下边界），第2天价格是102（+2%，触发上边界）
        # 应该标记为1，因为第一个触发的是上边界
        self.assertEqual(result.iloc[0], 1)


class TestFutureRolling(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的价格序列用于测试
        self.dates = pd.date_range(start='2020-01-01', periods=10)
        self.prices = pd.Series([100, 101, 103, 102, 105, 104, 106, 108, 107, 109], index=self.dates)
        
        # 模拟Feature对象
        class MockFeature:
            def __init__(self, values):
                self.values = values
            
            def load(self, instrument, start_index, end_index, *args):
                return self.values
            
            def get_longest_back_rolling(self):
                return 0
            
            def get_extended_window_size(self):
                return 0, 0
        
        self.mock_feature = MockFeature(self.prices)
    
    def test_future_rolling_mean(self):
        """测试未来窗口均值计算"""
        # 导入FutureRolling类
        from qlib.data.ops import FutureRolling
        
        # 创建FutureRolling算子，窗口大小为3，计算均值
        fr = FutureRolling(self.mock_feature, 3, lambda x: x.mean())
        
        # 调用_load_internal方法
        result = fr._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的未来3天均值应该是(101+103+102)/3 ≈ 102.0
        self.assertAlmostEqual(result.iloc[0], 102.0, places=6)
        # 第6天的未来3天均值应该是(108+107+109)/3 = 108.0
        self.assertAlmostEqual(result.iloc[6], 108.0, places=6)
    
    def test_future_rolling_sum(self):
        """测试未来窗口总和计算"""
        from qlib.data.ops import FutureRolling
        
        # 创建FutureRolling算子，窗口大小为2，计算总和
        fr = FutureRolling(self.mock_feature, 2, lambda x: x.sum())
        
        # 调用_load_internal方法
        result = fr._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的未来2天总和应该是101+103=204
        self.assertEqual(result.iloc[0], 204)
        # 第8天的未来2天总和应该是107+109=216
        self.assertEqual(result.iloc[8], 216)
    
    def test_future_rolling_edge_cases(self):
        """测试边界情况"""
        from qlib.data.ops import FutureRolling
        
        # 窗口大小为0的情况
        fr_zero = FutureRolling(self.mock_feature, 0, lambda x: x.mean())
        result_zero = fr_zero._load_internal("test", 0, 10)
        # 窗口大小为0应该返回自身
        pd.testing.assert_series_equal(result_zero, self.prices)
        
        # 窗口大小超过可用数据的情况
        fr_large = FutureRolling(self.mock_feature, 10, lambda x: x.mean())
        result_large = fr_large._load_internal("test", 0, 10)
        # 对于最后几个数据点，窗口将只包含可用的数据
        self.assertEqual(result_large.iloc[9], 109)  # 最后一个点没有未来数据，返回自身


class TestTBM(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的价格序列用于测试
        self.dates = pd.date_range(start='2020-01-01', periods=10)
        self.prices = pd.Series([100, 101, 103, 102, 105, 104, 106, 108, 107, 109], index=self.dates)
        
        # 模拟Feature对象
        class MockFeature:
            def __init__(self, prices):
                self.prices = prices
            
            def load(self, instrument, start_index, end_index, *args):
                return self.prices
            
            def get_longest_back_rolling(self):
                return 0
            
            def get_extended_window_size(self):
                return 0, 0
        
        self.mock_feature = MockFeature(self.prices)
    
    def test_upper_barrier_hit(self):
        """测试上边界被触发的情况"""
        from qlib.data.ops import TBM
        
        # 创建TBM算子，上边界为3%，下边界为3%，时间边界为3
        tbm = TBM(self.mock_feature, 0.03, 0.03, 3, use_percentage=True)
        
        # 调用_load_internal方法
        result = tbm._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，第2天价格是103，上涨了3%，触发上边界，应该标记为1
        self.assertEqual(result.iloc[0], 1)
    
    def test_lower_barrier_hit(self):
        """测试下边界被触发的情况"""
        from qlib.data.ops import TBM
        
        # 修改价格序列，制造一个下跌触发的情况
        prices = pd.Series([100, 101, 98, 97, 105, 104, 106, 108, 107, 109], index=self.dates)
        self.mock_feature.prices = prices
        
        # 创建TBM算子，上边界为3%，下边界为2%，时间边界为3
        tbm = TBM(self.mock_feature, 0.03, 0.02, 3, use_percentage=True)
        
        # 调用_load_internal方法
        result = tbm._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，第2天价格是98，下跌了2%，触发下边界，应该标记为-1
        self.assertEqual(result.iloc[0], -1)
    
    def test_time_barrier_hit(self):
        """测试时间边界被触发的情况"""
        from qlib.data.ops import TBM
        
        # 修改价格序列，制造一个在时间边界内没有触发价格边界的情况
        prices = pd.Series([100, 101, 101, 101, 105, 104, 106, 108, 107, 109], index=self.dates)
        self.mock_feature.prices = prices
        
        # 创建TBM算子，上边界为2%，下边界为2%，时间边界为3
        tbm = TBM(self.mock_feature, 0.02, 0.02, 3, use_percentage=True)
        
        # 调用_load_internal方法
        result = tbm._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，接下来3天的价格都在[98, 102]范围内，应该标记为0
        self.assertEqual(result.iloc[0], 0)
    
    def test_absolute_value_mode(self):
        """测试使用绝对值模式的情况"""
        from qlib.data.ops import TBM
        
        # 创建TBM算子，上边界为3（绝对价值），下边界为3（绝对价值），时间边界为3
        tbm = TBM(self.mock_feature, 3, 3, 3, use_percentage=False)
        
        # 调用_load_internal方法
        result = tbm._load_internal("test", 0, 10)
        
        # 验证结果
        # 第0天的价格是100，第2天价格是103，达到上边界3，应该标记为1
        self.assertEqual(result.iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
