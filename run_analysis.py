"""Run the complete analysis from the raw files and generate the final results."""

from pathlib import Path
import argparse
import json
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_data import DATA_DIR, CHUNK_SIZE, discover_pairs, prepare_data
from explore_data import save_walkthrough
from auction_rules import (reference_orders, recommend_order, maximum_order_size,
                           submission_timing, volume_only_size)
from validation import run_validation

ROOT = Path(__file__).resolve().parent


def describe(values):
    values = pd.Series(values).dropna()
    return dict(n=len(values), median=values.median(), p75=values.quantile(.75),
                p90=values.quantile(.90), mean=values.mean(), p99=values.quantile(.99),
                maximum=values.max(), p25=values.quantile(.25))


def empirical_findings(orders, volumes, out):
    """Full-sample descriptive findings, kept separate from historical forecasts."""
    rows = []
    for universe in ['all', '7203']:
        x = orders if universe == 'all' else orders.loc[orders.sym.eq('7203')]
        for cohort in ['market_additions', 'active', 'preclosing', 'late60_pooled', 'final5_pooled']:
            ref = (x[['date', 'sym', 'side', 'initial_size']].rename(columns={'initial_size': 'size'})
                   if cohort == 'market_additions' else reference_orders(x, cohort))
            for side in ['pooled', 'B', 'S']:
                if cohort.endswith('_pooled') and side != 'pooled':
                    continue
                z = ref if side == 'pooled' else ref.loc[ref.side.eq(side)]
                rows.append(dict(universe=universe, cohort=cohort, side=side, **describe(z['size'])))
    pd.DataFrame(rows).to_csv(out / 'order_size_distributions.csv', index=False)

    timing = []
    daily = []
    seconds = [f'15:29:{s:02d}' for s in range(55, 60)]
    for universe in ['all', '7203']:
        x = orders if universe == 'all' else orders.loc[orders.sym.eq('7203')]
        x = x.loc[x.is_final5_addition].copy()
        x['second'] = x.second.astype(str)
        for exclude in [False, True]:
            z = x.loc[~x.boundary_timestamp] if exclude else x
            counts = z.second.value_counts().reindex(seconds, fill_value=0)
            for second, n in counts.items():
                timing.append(dict(universe=universe, exclude_boundary=exclude,
                                   second=second, count=int(n)))
            d = z.groupby(['date', 'second'], observed=True).size().rename('count').reset_index()
            d['universe'] = universe
            d['exclude_boundary'] = exclude
            daily.append(d)
    timing = pd.DataFrame(timing)
    timing.to_csv(out / 'timing_counts.csv', index=False)
    pd.concat(daily, ignore_index=True).to_csv(out / 'timing_counts_by_day.csv', index=False)

    rows = []
    for universe in ['all', '7203']:
        x = volumes if universe == 'all' else volumes.loc[volumes.sym.eq('7203')]
        rows.append(dict(universe=universe, zero_print_days=int(x.closing_volume.eq(0).sum()),
                         unknown_days=int(x.closing_volume.isna().sum()), **describe(x.closing_volume)))
    pd.DataFrame(rows).to_csv(out / 'closing_volume_distributions.csv', index=False)
    return timing


def january_outputs(orders, volumes, out, safety_margin_seconds):
    """Generate January recommendations for both possible trade directions."""
    dates = sorted(d for d in volumes.date.unique()
                   if pd.Timestamp(d).year == 2025 and pd.Timestamp(d).month == 1)
    # Sizing is stock-specific; timing describes all stocks. Keeping these rows
    # avoids repeatedly scanning unrelated early-day orders for Toyota calls.
    relevant = orders.loc[orders.sym.eq('7203') | orders.is_final5_addition].copy()
    rows, fallback, sensitivity, scenarios = [], [], [], []

    for date in dates:
        for side in ['B', 'S']:
            result = recommend_order(relevant, volumes, '7203', date, side, 100,
                                     safety_margin_seconds=safety_margin_seconds)
            rows.append({k: v for k, v in result.items() if k != 'bucket_counts'})
            for ref in ['active', 'preclosing', 'late60_pooled']:
                for q in [.5, .75, .9]:
                    for alpha in [.01, .02, .05, .10]:
                        for vq in [.25, .5]:
                            value = maximum_order_size(relevant, volumes, '7203', date, side, 100,
                                                       reference=ref, order_quantile=q,
                                                       participation=alpha, volume_quantile=vq)
                            sensitivity.append(dict(date=pd.Timestamp(date), side=side, reference=ref,
                                                    order_quantile=q, participation=alpha,
                                                    volume_quantile=vq, maximum_size=value['maximum_size']))
        fallback.append(dict(date=pd.Timestamp(date).strftime('%Y-%m-%d'), ticker='7203',
                             **volume_only_size(volumes, '7203', date, 100)))
        for margin in [.5, 1, 2, 3, 5, 6]:
            for policy in ['historical_bucket', 'earliest']:
                t = submission_timing(relevant, date, safety_margin_seconds=margin, policy=policy)
                scenarios.append(dict(date=pd.Timestamp(date).strftime('%Y-%m-%d'),
                                      **{k: v for k, v in t.items() if k != 'bucket_counts'}))

    detail = pd.DataFrame(rows)
    detail.to_csv(out / 'tables' / 'Toyota_order_diagnostics.csv', index=False)
    wide = detail.pivot(index='date', columns='direction', values='maximum_size').rename(
        columns={'B': 'maximum_buy_size', 'S': 'maximum_sell_size'}).reset_index()
    buy = detail.loc[detail.direction.eq('B')].set_index('date')
    sell = detail.loc[detail.direction.eq('S')].set_index('date')
    wide['ticker'] = '7203'
    wide['history_sessions'] = wide.date.map(buy.history_days)
    wide['submission_time'] = [
        buy.loc[d, 'submission_time'] or sell.loc[d, 'submission_time'] for d in wide.date
    ]
    wide['status'] = [
        buy.loc[d, 'status'] if buy.loc[d, 'status'] == sell.loc[d, 'status']
        else f"B:{buy.loc[d, 'status']};S:{sell.loc[d, 'status']}" for d in wide.date
    ]
    wide['safety_margin_seconds'] = safety_margin_seconds
    wide = wide[['date', 'ticker', 'maximum_buy_size', 'maximum_sell_size',
                 'submission_time', 'history_sessions', 'status', 'safety_margin_seconds']]
    wide.to_csv(out / 'Toyota_January_2025.csv', index=False)
    pd.DataFrame(fallback).to_csv(out / 'tables' / 'Toyota_no_order_feed.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out / 'tables' / 'toyota_parameter_sensitivity.csv', index=False)
    pd.DataFrame(scenarios).to_csv(out / 'tables' / 'timing_margin_scenarios.csv', index=False)

    v = volumes.loc[volumes.sym.eq('7203'), ['date', 'closing_volume']].copy()
    v['date'] = v.date.dt.strftime('%Y-%m-%d')
    realized = detail.merge(v, on='date', how='left')
    realized['realized_participation'] = realized.maximum_size / realized.closing_volume.replace(0, np.nan)
    realized.to_csv(out / 'tables' / 'Toyota_realized_participation.csv', index=False)
    return wide


def make_figures(timing, recommendations, out):
    """Save two descriptive figures using tables from the current run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    seconds = [f'15:29:{s:02d}' for s in range(55, 60)]
    counts = timing.loc[timing.universe.eq('all')].pivot(
        index='second', columns='exclude_boundary', values='count'
    ).reindex(seconds)
    ordinary = counts[True]
    boundary = counts[False] - ordinary

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(seconds, ordinary, label='Other observed additions')
    ax.bar(seconds, boundary, bottom=ordinary,
           label='Exact 15:29:59.999999 boundary')
    ax.set_title('Historical market-order additions: final five seconds\n'
                 'All stocks; descriptive counts, not an optimal submission time',
                 fontsize=12)
    ax.set_xlabel('Tokyo time (one-second buckets)')
    ax.set_ylabel('Number of market-order additions')
    ax.yaxis.set_major_formatter('{x:,.0f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'timing_distribution.png', dpi=200)
    plt.close(fig)

    recommendations = recommendations.sort_values('date')
    dates = pd.to_datetime(recommendations.date)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dates, recommendations.maximum_buy_size, marker='o',
            markerfacecolor='none', label='BUY direction')
    ax.plot(dates, recommendations.maximum_sell_size, marker='x',
            linestyle='--', label='SELL direction')
    ax.set_title('Toyota (7203): January 2025 maximum order sizes\n'
                 'BUY and SELL are alternatives, not simultaneous trades',
                 fontsize=12)
    ax.set_xlabel('Tokyo session date')
    ax.set_ylabel('Maximum order size (shares)')
    ax.set_xticks(dates)
    ax.set_xticklabels(dates.dt.strftime('%b %d'), rotation=45, ha='right')
    ax.yaxis.set_major_formatter('{x:,.0f}')
    ax.axhline(0, color='0.7', linewidth=0.7, zorder=0)
    ax.margins(y=0.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'toyota_recommendations.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--dates', nargs='+', help='Optional session subset; research needs at least six dates')
    parser.add_argument('--safety-margin-seconds', type=float, default=1.0,
                        help='Illustrative operational assumption, not estimated latency (default: 1)')
    args = parser.parse_args()

    if args.chunk_size < 1 or not np.isfinite(args.safety_margin_seconds) or args.safety_margin_seconds <= 0:
        parser.error('Chunk size and safety margin must be positive and finite')
    pairs = discover_pairs(args.data_dir, args.dates)
    if len(pairs) < 6:
        parser.error('Full validation needs at least six sessions; use explore_data.py or prepare_data.py for fewer')

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = dict(data_dir=str(args.data_dir.resolve()), dates=[d for d, _, _ in pairs],
                  chunk_size=args.chunk_size, safety_margin_seconds=args.safety_margin_seconds,
                  python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                  baseline=dict(reference='preclosing', order_quantile=.75, participation=.05,
                                volume_quantile=.25, min_days=5, min_orders=20, lookback='expanding',
                                include_boundary=False, toyota_lot_size=100), status='running')
    start = time.perf_counter()
    (out / 'run_manifest.json').write_text(json.dumps(config, indent=2))

    # Always rebuild from the raw files; prepared tables are outputs.
    orders, volumes, audit = prepare_data(args.data_dir, out, selected_dates=args.dates,
                                         chunk_size=args.chunk_size)
    print('Raw reconstruction complete. Building educational walkthrough.', flush=True)
    example_date = '20250106' if '20250106' in config['dates'] else config['dates'][0]
    save_walkthrough(args.data_dir, out / 'tables', date=example_date, chunk_size=args.chunk_size)
    timing = empirical_findings(orders, volumes, out / 'tables')
    forecasts = run_validation(orders, volumes, out / 'tables')
    wide = january_outputs(orders, volumes, out, args.safety_margin_seconds)

    from tests import run_tests
    result = run_tests(orders, volumes, forecasts, out)
    make_figures(timing, wide, out / 'figures')

    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    config.update(status='complete', elapsed_seconds=round(time.perf_counter()-start, 2),
                  tests_run=result.testsRun, tests_passed=passed, tests_skipped=len(result.skipped))
    (out / 'run_manifest.json').write_text(json.dumps(config, indent=2))
    print(wide.to_string(index=False))
    print(f'Complete: {passed} tests passed; outputs saved to {out}', flush=True)


if __name__ == '__main__':
    main()