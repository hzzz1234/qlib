#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import warnings
from typing import Dict, List, Text, Tuple, Union

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel


class BuyThenSellTopkDropoutStrategy(BaseSignalStrategy):
    """
    基于 `TopkDropoutStrategy` 的“先买后卖”版本：

    - 订单顺序为 **buy -> sell**，因此不会用当期卖出的资金支持买入
    - 当现金不足时，允许 **跳过买单**（不强制卖出、不报错）
    """

    def __init__(
        self,
        *,
        topk: int,
        n_drop: int,
        method_sell: str = "bottom",
        method_buy: str = "top",
        hold_thresh: int = 1,
        only_tradable: bool = False,
        forbid_all_trade_at_limit: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit

    def _estimate_buy_required_cash(self, buy_amount: float, buy_price: float) -> float:
        trade_val = float(buy_amount) * float(buy_price)
        cost = max(trade_val * float(self.trade_exchange.open_cost), float(self.trade_exchange.min_cost))
        return trade_val + cost

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal 
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        if self.only_tradable:

            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:

            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        pos_for_buy: Position = copy.deepcopy(self.trade_position)
        pos_for_sell: Position = copy.deepcopy(self.trade_position)

        buy_order_list: List[Order] = []
        sell_order_list: List[Order] = []

        current_stock_list = pos_for_sell.get_stock_list()
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index

        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError("This type of input is not supported")

        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = candi
        else:
            raise NotImplementedError("This type of input is not supported")

        buy = today[: len(sell) + self.topk - len(last)]

        # ---- BUY first (cannot use proceeds from sells) ----
        cash = float(pos_for_buy.get_cash())
        value_per_stock = cash * float(self.get_risk_degree(trade_step)) / len(buy) if len(buy) > 0 else 0.0

        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue

            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            if buy_price is None or np.isnan(buy_price) or buy_price <= 1e-12:
                continue

            buy_amount = value_per_stock / float(buy_price)
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            if buy_amount <= 1e-12:
                continue

            # Cash check: if not enough, skip buying this stock entirely
            required_cash = self._estimate_buy_required_cash(buy_amount, float(buy_price))
            if float(pos_for_buy.get_cash()) + 1e-12 < required_cash:
                continue

            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)
            # Pre-simulate to update remaining cash; volume limits may still clip, which is OK
            self.trade_exchange.deal_order(buy_order, position=pos_for_buy)

        # ---- SELL after buy ----
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code not in sell:
                continue

            time_per_step = self.trade_calendar.get_freq()
            if pos_for_sell.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                continue

            sell_amount = pos_for_sell.get_stock_amount(code=code)
            if sell_amount <= 1e-12:
                continue

            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)

        return TradeDecisionWO(buy_order_list + sell_order_list, self)

