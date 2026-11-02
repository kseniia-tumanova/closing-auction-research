"""Test the core data parsing, sizing rules, and protection against look-ahead bias."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from auction_rules import maximum_order_size, recommend_order, submission_timing, volume_only_size
from prepare_data import DataQualityError, aggregate_trade, discover_pairs, read_auction


class RawDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.auction_path = self.root / "auction20250106.csv"
        self.trade_path = self.root / "trade20250106.csv"
        self.auction = pd.DataFrame([
            ["7203", 0, "B", 100, "0D15:25:00.000000000", 100, 1],
            ["7203", 0, "B", 150, "0D15:25:01.000000000", 50, 2],
            ["7203", 0, "B", 120, "0D15:26:00.000000000", -30, 1],
            ["7203", 0, "B", 70, "0D15:27:00.000000000", -50, 2],
            ["7203", 0, "B", 0, "0D15:30:00.000000000", -70, 1],
        ], columns=["sym", "price", "buySell", "volume", "time", "chg", "orderId"])
        self.trades = pd.DataFrame([
            ["7203", "0D15:30:00.100000000", 1000, 500, "0D15:24:59.000000000"],
            ["7203", "0D15:30:00.200000000", 1000, 200, "0D15:30:00.000000000"],
        ], columns=["sym", "time", "price", "qty", "exchTime"])

    def read_orders(self, events, chunk_size=2):
        events.to_csv(self.auction_path, sep="|", index=False)
        return read_auction(self.auction_path, "20250106", chunk_size)[0]

    def read_trades(self, events):
        events.to_csv(self.trade_path, sep="|", index=False)
        return aggregate_trade(self.trade_path, "20250106", chunk_size=1)[0]

    def test_order_reconstruction(self):
        orders = self.read_orders(self.auction, chunk_size=1)
        pd.testing.assert_frame_equal(
            orders, self.read_orders(self.auction, chunk_size=100)
        )
        self.assertEqual(orders.initial_size.tolist(), [100, 50])
        self.assertEqual(orders.residual_before_close.tolist(), [70, 0])
        self.assertEqual(orders.final_balance.tolist(), [0, 0])

    def test_invalid_order_lifecycles(self):
        cases = [
            ("negative_running_balance", {"volume": 40, "chg": -110}),
            ("repeat_positive", {"volume": 180, "chg": 30}),
            ("changing_price", {"price": 1}),
            ("changing_side", {"buySell": "S"}),
        ]
        for reason, changes in cases:
            with self.subTest(reason=reason):
                events = self.auction.iloc[:3].copy()
                for column, value in changes.items():
                    events.loc[2, column] = value
                with self.assertRaisesRegex(DataQualityError, reason):
                    self.read_orders(events)

    def test_closing_trade_identification_and_quality(self):
        # Both captures are after 15:30; only the exchange timestamp identifies the close.
        missing = self.trades.iloc[[-1]].assign(
            time="0D15:30:00.300000000", qty=np.nan
        )
        cases = [
            ("known", self.trades, 200, True, 1),
            ("partly_missing", pd.concat([self.trades, missing]), np.nan, False, 2),
            ("no_closing_print", self.trades.iloc[:1], 0, True, 0),
        ]
        for name, events, quantity, valid, prints in cases:
            with self.subTest(case=name):
                row = self.read_trades(events).loc["7203"]
                self.assertEqual(row.nonclosing_volume, 500)
                self.assertEqual(row.closing_valid, valid)
                self.assertEqual(row.closing_prints, prints)
                if pd.isna(quantity):
                    self.assertTrue(pd.isna(row.closing_volume))
                else:
                    self.assertEqual(row.closing_volume, quantity)

    def test_missing_daily_pair_fails(self):
        for present in [self.auction_path, self.trade_path]:
            with self.subTest(present=present.name):
                present.touch()
                with self.assertRaises(DataQualityError):
                    discover_pairs(self.root)
                present.unlink()


class RuleTests(unittest.TestCase):
    def setUp(self):
        days = pd.date_range("2025-01-01", periods=7)
        self.orders = pd.DataFrame([
            dict(date=day, sym="7203", side=side,
                 initial_size=size * multiple, residual_before_close=size * multiple,
                 second="15:29:59", boundary_timestamp=False, is_market=True)
            for day in days
            for side, multiple in [("B", 1), ("S", 2)]
            for size in [125, 225, 325, 425, 525]
        ])
        self.volumes = pd.DataFrame(dict(
            date=days, sym="7203", closing_volume=10_000.0, closing_valid=True
        ))
        self.kw = dict(stock="7203", date="2025-01-06", direction="B", lot_size=100)

    def size(self, **changes):
        return maximum_order_size(
            self.orders, self.volumes, **{**self.kw, **changes}
        )

    def test_caps_and_downward_lot_rounding(self):
        cases = [
            (10_000, 500, 400, "order", "ok"),
            (6_500, 325, 300, "liquidity", "ok"),
            (1_000, 50, 0, "liquidity", "below_one_lot"),
        ]
        for volume, liquidity_cap, size, binding, status in cases:
            with self.subTest(volume=volume):
                self.volumes["closing_volume"] = float(volume)
                result = self.size()
                self.assertEqual(result["order_cap"], 425)
                self.assertEqual(result["liquidity_cap"], liquidity_cap)
                self.assertEqual(result["maximum_size"], size)
                self.assertEqual(result["binding_constraint"], binding)
                self.assertEqual(result["status"], status)

    def test_only_prior_dates_affect_recommendation(self):
        target = pd.Timestamp(self.kw["date"])
        expected = recommend_order(
            self.orders.loc[self.orders.date.lt(target)],
            self.volumes.loc[self.volumes.date.lt(target)],
            **self.kw, safety_margin_seconds=1
        )
        self.assertEqual(expected["status"], "ok")
        self.assertEqual(expected["history_days"], 5)
        self.assertEqual(expected["reference_orders"], 25)
        self.assertEqual(expected["last_history_date"], "2025-01-05")
        self.assertEqual(
            recommend_order(self.orders, self.volumes, **self.kw, safety_margin_seconds=1),
            expected
        )
        fallback_kw = dict(
            stock=self.kw["stock"], date=target, lot_size=self.kw["lot_size"]
        )
        expected_fallback = volume_only_size(
            self.volumes.loc[self.volumes.date.lt(target)], **fallback_kw
        )
        self.assertEqual(
            volume_only_size(self.volumes, **fallback_kw), expected_fallback
        )
        future = self.orders.date.ge(target)
        self.orders.loc[future, ["initial_size", "residual_before_close"]] = 10**12
        self.orders.loc[future, "second"] = "15:29:55"
        self.volumes.loc[self.volumes.date.ge(target), "closing_volume"] = 10**15
        self.assertEqual(
            recommend_order(self.orders, self.volumes, **self.kw, safety_margin_seconds=1),
            expected
        )
        self.assertEqual(
            volume_only_size(self.volumes, **fallback_kw), expected_fallback
        )

    def test_reference_population_filters(self):
        large = self.orders.assign(initial_size=10**9, residual_before_close=10**9)
        self.orders = pd.concat([
            self.orders,
            large.assign(sym="1301"),
            large.assign(is_market=False),
            large.assign(boundary_timestamp=True),
            large.assign(second="15:24:59"),
            large.assign(second="15:30:00"),
        ], ignore_index=True)
        self.volumes = pd.concat([
            self.volumes, self.volumes.assign(sym="1301", closing_volume=0.0)
        ], ignore_index=True)
        for side, cap, size in [("B", 425, 400), ("S", 850, 500)]:
            with self.subTest(side=side):
                result = self.size(direction=side)
                self.assertEqual(result["order_cap"], cap)
                self.assertEqual(result["maximum_size"], size)
                self.assertEqual(result["reference_orders"], 25)
                self.assertEqual(result["history_days"], 5)
        included = self.size(include_boundary=True)
        self.assertEqual(included["reference_orders"], 50)
        self.assertGreater(included["order_cap"], 425)

    def test_minimum_historical_support(self):
        for date, days in [("2025-01-01", 0), ("2025-01-05", 4)]:
            with self.subTest(prior_days=days):
                result = recommend_order(
                    self.orders, self.volumes, **{**self.kw, "date": date},
                    safety_margin_seconds=1
                )
                self.assertEqual(result["history_days"], days)
                self.assertEqual(result["status"], "insufficient_days")
                self.assertEqual(result["maximum_size"], 0)
                self.assertIsNone(result["submission_time"])
        prior_buys = self.orders.loc[
            self.orders.date.lt(self.kw["date"]) & self.orders.side.eq("B")
        ]
        for count, status, size in [(19, "insufficient_orders", 0), (20, "ok", 400)]:
            with self.subTest(prior_orders=count):
                self.orders = prior_buys.iloc[:count].copy()
                result = self.size()
                self.assertEqual(result["history_days"], 5)
                self.assertEqual(result["reference_orders"], count)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["maximum_size"], size)

    def test_invalid_volume_history_abstains(self):
        for quantity, valid in [(np.nan, True), (-1, True), (np.inf, True), (10_000, False)]:
            with self.subTest(quantity=quantity, valid=valid):
                self.volumes.loc[0, ["closing_volume", "closing_valid"]] = [quantity, valid]
                result = recommend_order(
                    self.orders, self.volumes, **self.kw, safety_margin_seconds=1
                )
                self.assertEqual(result["status"], "invalid_volume_history")
                self.assertEqual(result["maximum_size"], 0)
                self.assertIsNone(result["submission_time"])

    def test_timing_safety_margin(self):
        cases = [(0.5, "15:29:59"), (1, "15:29:59"), (1.1, "15:29:58"),
                 (2, "15:29:58"), (3, "15:29:57"), (5, "15:29:55"), (5.1, None)]
        for margin, expected in cases:
            with self.subTest(margin=margin):
                result = recommend_order(
                    self.orders, self.volumes, **self.kw, safety_margin_seconds=margin
                )
                self.assertEqual(result["submission_time"], expected)
                self.assertEqual(result["maximum_size"], 400 if expected else 0)
                self.assertEqual(
                    result["status"], "ok" if expected else "margin_exceeds_window"
                )
        earlier = submission_timing(
            self.orders.assign(second="15:29:57"), self.kw["date"],
            safety_margin_seconds=1
        )
        self.assertEqual(earlier["submission_time"], "15:29:57")
        for margin in [0, -1, np.inf, np.nan]:
            with self.subTest(invalid_margin=margin), self.assertRaises(ValueError):
                submission_timing(
                    self.orders, self.kw["date"], safety_margin_seconds=margin
                )


class RealDataTests(unittest.TestCase):
    orders = volumes = forecasts = None

    def setUp(self):
        if self.orders is None or self.volumes is None:
            self.skipTest("Prepared order and volume tables are unavailable.")

    def test_real_future_invariance(self):
        dates = sorted(self.volumes.loc[self.volumes.sym.eq("7203"), "date"].unique())
        if len(dates) < 6:
            self.skipTest("Need at least six Toyota sessions.")
        target = pd.Timestamp(dates[max(5, len(dates) // 2)])
        second = self.orders.second.astype(str)
        # Retain all stocks' final-five-second orders because timing is pooled.
        orders = self.orders.loc[
            self.orders.sym.eq("7203") | (second.ge("15:29:55") & second.lt("15:30:00"))
        ].copy()
        kw = dict(stock="7203", date=target, direction="B", lot_size=100,
                  safety_margin_seconds=1)
        expected = recommend_order(orders, self.volumes, **kw)
        self.assertEqual(expected["status"], "ok")
        self.assertEqual(
            recommend_order(
                orders.loc[orders.date.lt(target)],
                self.volumes.loc[self.volumes.date.lt(target)], **kw
            ), expected
        )
        future = orders.date.ge(target)
        self.assertTrue(future.any())
        orders.loc[future, ["initial_size", "residual_before_close"]] = 10**12
        orders["second"] = orders.second.astype(str)
        orders.loc[future, "second"] = "15:29:55"
        volumes = self.volumes.copy()
        volumes.loc[volumes.date.ge(target), "closing_volume"] = 10**15
        self.assertEqual(recommend_order(orders, volumes, **kw), expected)

    def test_real_batch_matches_public_recommendation(self):
        if self.forecasts is None:
            self.skipTest("Prepared walk-forward table is unavailable.")
        f = self.forecasts
        eligible = f.loc[
            f.sym.eq("7203") & f.reference.eq("preclosing") & f.lookback.eq("expanding")
            & f.history_days.ge(5) & f.valid_days.eq(f.history_days)
            & f.order_count.ge(20) & f.q75.ge(100) & f.v25.ge(2_000)
        ].sort_values(["date", "side"])
        self.assertFalse(eligible.empty, "No supported Toyota forecast to compare.")
        row = eligible.iloc[len(eligible) // 2]
        result = recommend_order(
            self.orders, self.volumes, row.sym, row.date, row.side, 100,
            safety_margin_seconds=1
        )
        self.assertEqual(result["status"], "ok")
        np.testing.assert_allclose(
            [result["order_cap"], result["volume_estimate"]], [row.q75, row.v25]
        )
        self.assertEqual(result["reference_orders"], row.order_count)
        self.assertEqual(result["history_days"], row.history_days)
        expected_size = int(np.floor(min(row.q75, 0.05 * row.v25) / 100)) * 100
        self.assertEqual(result["maximum_size"], expected_size)
        self.assertEqual(pd.Timestamp(result["last_history_date"]), row.last_history_date)
        self.assertLess(row.last_history_date, row.date)


def run_tests(orders=None, volumes=None, forecasts=None, output_dir=None):
    if (orders is None) != (volumes is None):
        raise ValueError("Provide both orders and volumes, or neither.")

    cases = []
    for test_class in [RawDataTests, RuleTests]:
        cases.extend(unittest.defaultTestLoader.loadTestsFromTestCase(test_class))
    if orders is not None:
        for case in unittest.defaultTestLoader.loadTestsFromTestCase(RealDataTests):
            case.orders, case.volumes, case.forecasts = orders, volumes, forecasts
            cases.append(case)
    names = [case.id() for case in cases]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = dict(
            tests_run=result.testsRun, failures=len(result.failures),
            errors=len(result.errors), skipped=len(result.skipped),
            all_passed=result.wasSuccessful(), cases=names
        )
        (output_dir / "test_results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    if not result.wasSuccessful():
        raise RuntimeError("Tests failed; review the errors before using the results.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    prepared = args.output_dir / "prepared"
    order_path = prepared / "market_orders.pkl.gz"
    volume_path = prepared / "closing_volumes.csv.gz"
    forecast_path = prepared / "walk_forward.pkl.gz"

    if order_path.exists() != volume_path.exists():
        raise FileNotFoundError("Prepared order and volume tables must both be present.")
    if order_path.exists():
        orders = pd.read_pickle(order_path)
        volumes = pd.read_csv(volume_path, dtype={"sym": str}, parse_dates=["date"])
        forecasts = pd.read_pickle(forecast_path) if forecast_path.exists() else None
        run_tests(orders, volumes, forecasts, args.output_dir)
    else:
        run_tests(output_dir=args.output_dir)