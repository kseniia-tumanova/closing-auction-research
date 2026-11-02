"""Test the sizing and liquidity assumptions with walk-forward historical analysis"""
from pathlib import Path

import numpy as np
import pandas as pd

from auction_rules import reference_orders
from prepare_data import save_pickle

PRIMARY_REFERENCES = ['active', 'preclosing', 'late60_pooled']
REFERENCES = PRIMARY_REFERENCES + ['active_pooled', 'final5_pooled']
KEY = ['date', 'sym', 'side']


def forecasts(orders, volumes):
    """Estimate three order quantiles and two volume quantiles using prior dates."""
    references = {name: reference_orders(orders, name) for name in REFERENCES}
    days = sorted(volumes.date.unique())
    frames = []
    for i, date in enumerate(days):
        if i == 0:
            continue
        for lookback in [None, 5, 10]:
            dates = days[:i] if lookback is None else days[max(0, i-lookback):i]
            history = volumes.loc[volumes.date.isin(dates)]
            grouped_volume = history.groupby('sym').closing_volume
            liquidity = grouped_volume.agg(history_days='size', valid_days='count')
            quantiles = grouped_volume.quantile([.25, .5]).unstack().rename(columns={.25: 'v25', .5: 'v50'})
            liquidity = liquidity.join(quantiles)
            for name, reference in references.items():
                if lookback is not None and name not in PRIMARY_REFERENCES:
                    continue
                x = reference.loc[reference.date.isin(dates)]
                grouped_size = x.groupby(['sym', 'side'], observed=True)['size']
                quantiles = grouped_size.quantile([.5, .75, .9]).unstack().rename(
                    columns={.5: 'q50', .75: 'q75', .9: 'q90'})
                f = grouped_size.agg(order_count='size').join(quantiles).reset_index()
                if name.endswith('_pooled'):
                    f = pd.concat([f.assign(side=side) for side in ['B', 'S']], ignore_index=True)
                f = f.merge(liquidity, on='sym', how='left')
                f['date'] = date
                f['reference'] = name
                f['lookback'] = 'expanding' if lookback is None else str(lookback)
                f['last_history_date'] = max(dates)
                frames.append(f)
        print('Forecasted', pd.Timestamp(date).date(), flush=True)
    f = pd.concat(frames, ignore_index=True)
    if not f.last_history_date.lt(f.date).all():
        raise AssertionError('A forecast used the target date or a future date')
    # Outcomes are attached only after historical estimates have been fixed.
    return f.merge(volumes[['date', 'sym', 'closing_volume']], on=['date', 'sym'], how='left')


def supported(f):
    return f.history_days.ge(5) & f.valid_days.eq(f.history_days) & f.order_count.ge(20)


def late_targets(orders):
    second = orders.second.astype(str)
    mask = orders.is_market & ~orders.boundary_timestamp & second.ge('15:29:55') & second.lt('15:30:00')
    return orders.loc[mask, KEY + ['initial_size']]


def compare_references(f, orders, volumes, out):
    """Compare the candidate historical reference groups against the same next-day market orders from the final 5 seconds"""
    target = late_targets(orders).rename(columns={'initial_size': 'observed_size'})
    f = f.loc[f.lookback.eq('expanding') & supported(f)].copy()
    z = target.merge(f, on=KEY, how='inner')
    common = f.loc[f.reference.isin(PRIMARY_REFERENCES)].groupby(KEY).reference.nunique()
    common = common.loc[common.eq(len(PRIMARY_REFERENCES))].index
    z['common_sample'] = pd.MultiIndex.from_frame(z[KEY]).isin(common)
    error = np.log(z.observed_size) - np.log(z.q75)
    z['log_pinball'] = np.maximum(.75*error, -.25*error)
    z['below_cap'] = z.observed_size.le(z.q75)
    z['capped_size'] = np.minimum(z.q75, .05*z.v25)
    z['below_capped'] = z.observed_size.le(z.capped_size)
    metrics, daily = [], []
    for sample in ['available', 'common']:
        selected = z if sample == 'available' else z.loc[z.common_sample & z.reference.isin(PRIMARY_REFERENCES)]
        # Many orders from one stock on one day must not masquerade as many
        # independent dates: aggregate groups first, then weight dates equally.
        stock_day = selected.groupby(['reference'] + KEY).agg(
            log_pinball=('log_pinball', 'mean'), coverage=('below_cap', 'mean'),
            capped_coverage=('below_capped', 'mean'), target_events=('observed_size', 'size')).reset_index()
        by_day = stock_day.groupby(['reference', 'date']).agg(
            log_pinball=('log_pinball', 'mean'), coverage=('coverage', 'mean'),
            capped_coverage=('capped_coverage', 'mean'), target_groups=('sym', 'size'),
            target_events=('target_events', 'sum')).reset_index()
        by_day['sample'] = sample
        daily.append(by_day)
        for reference, group in by_day.groupby('reference'):
            metrics.append(dict(reference=reference, sample=sample, days=len(group),
                                log_pinball=group.log_pinball.mean(), coverage=group.coverage.mean(),
                                capped_coverage=group.capped_coverage.mean(),
                                target_groups=group.target_groups.sum(), target_events=group.target_events.sum()))
    pd.DataFrame(metrics).to_csv(out / 'reference_comparison.csv', index=False)
    pd.concat(daily).to_csv(out / 'reference_comparison_by_day.csv', index=False)
    z.loc[z.sym.eq('7203')].to_csv(out / 'toyota_reference_targets.csv', index=False)
    eligible_days = sorted(volumes.date.unique())[5:]
    grid = pd.concat([volumes.loc[volumes.date.isin(eligible_days), ['date', 'sym']].assign(side=s)
                      for s in ['B', 'S']], ignore_index=True)
    support = []
    for reference in REFERENCES:
        x = grid.merge(f.loc[f.reference.eq(reference), KEY + ['q75', 'v25']], on=KEY, how='left')
        support.append(dict(reference=reference, stock_side_days=len(x), with_support=int(x.q75.notna().sum()),
                            positive_cap=int(np.minimum(x.q75, .05*x.v25).gt(0).sum())))
    pd.DataFrame(support).to_csv(out / 'reference_support.csv', index=False)


def parameter_sensitivity(f, out):
    """Test combinations of order-size percentiles, participation limits, and volume percentiles before lot-size rounding."""
    rows, daily = [], []
    base = f.loc[supported(f) & f.closing_volume.notna()]
    for (reference, window), group in base.groupby(['reference', 'lookback']):
        for quantile in [50, 75, 90]:
            for alpha in [.01, .02, .05, .10]:
                for volume_quantile in [25, 50]:
                    x = group[KEY + ['closing_volume']].copy()
                    x['size'] = np.minimum(group[f'q{quantile}'], alpha*group[f'v{volume_quantile}'])
                    x['liquidity_binds'] = alpha*group[f'v{volume_quantile}'] < group[f'q{quantile}']
                    x['participation'] = x['size'] / x.closing_volume.replace(0, np.nan)
                    positive = x.loc[x['size'].gt(0)]
                    live = positive.loc[positive.closing_volume.gt(0)]
                    # A positive hypothetical order on a no-print day is a
                    # separate failure case, never an invented finite ratio.
                    no_close = int(positive.closing_volume.eq(0).sum())
                    stable = positive.sort_values(['sym', 'side', 'date']).copy()
                    stable['log_size'] = np.log(stable['size'])
                    stable['log_change'] = stable.groupby(['sym', 'side']).log_size.diff().abs()
                    rows.append(dict(
                        reference=reference, lookback=window, order_quantile=quantile/100,
                        participation_limit=alpha, volume_quantile=volume_quantile/100,
                        supported=len(x), positive_orders=len(positive), positive_without_close=no_close,
                        liquidity_binds=float(x.liquidity_binds.mean()),
                        median_participation=live.participation.median(),
                        p95_participation=live.participation.quantile(.95),
                        p99_participation=live.participation.quantile(.99),
                        exceeds_policy_fraction=float(live.participation.gt(alpha).mean()),
                        median_abs_log_change=stable.log_change.median()))
                    if quantile == 75 and alpha == .05 and volume_quantile == 25:
                        by_day = live.groupby('date').agg(
                            n=('size', 'size'), median_participation=('participation', 'median'),
                            p95_participation=('participation', lambda s: s.quantile(.95)),
                            exceeds_policy_fraction=('participation', lambda s: s.gt(alpha).mean())).reset_index()
                        by_day['reference'] = reference
                        by_day['lookback'] = window
                        daily.append(by_day)
    pd.DataFrame(rows).to_csv(out / 'parameter_sensitivity.csv', index=False)
    pd.concat(daily).to_csv(out / 'baseline_risk_by_day.csv', index=False)


def no_feed_comparison(volumes, out):
    """Compare historical closing-volume (C) estimates with estimates based on
    the relationship between closing volume and prior-day non-closing volume (R).
    """
    rows = []
    for stock, x in volumes.sort_values('date').groupby('sym'):
        for i in range(5, len(x)):
            history, now = x.iloc[:i], x.iloc[i]
            if history.closing_volume.isna().any() or pd.isna(now.closing_volume):
                continue
            closing = history.closing_volume
            predictions = {'historical_p25': closing.quantile(.25), 'historical_median': closing.median()}
            valid = history.loc[history.nonclosing_valid & history.nonclosing_volume.gt(0)]
            last = history.iloc[-1]
            if len(valid) >= 5 and last.nonclosing_valid and last.nonclosing_volume > 0:
                ratio = valid.closing_volume / valid.nonclosing_volume
                predictions['ratio_p25_last_volume'] = ratio.quantile(.25)*last.nonclosing_volume
                predictions['ratio_median_last_volume'] = ratio.median()*last.nonclosing_volume
            for method, prediction in predictions.items():
                quantile = .25 if 'p25' in method else .5
                error = np.log1p(now.closing_volume) - np.log1p(prediction)
                rows.append(dict(date=now.date, sym=stock, method=method, quantile=quantile,
                                 prediction=prediction, actual=now.closing_volume,
                                 log_pinball=max(quantile*error, (quantile-1)*error),
                                 covered=float(now.closing_volume <= prediction)))
    z = pd.DataFrame(rows)
    common = z.groupby(['date', 'sym']).method.nunique()
    common = common.loc[common.eq(4)].index
    z = z.loc[pd.MultiIndex.from_frame(z[['date', 'sym']]).isin(common)]
    daily = z.groupby(['method', 'date']).agg(log_pinball=('log_pinball', 'mean'),
                                            coverage=('covered', 'mean'), n=('sym', 'size')).reset_index()
    daily.to_csv(out / 'no_feed_by_day.csv', index=False)
    daily.groupby('method').agg(log_pinball=('log_pinball', 'mean'), coverage=('coverage', 'mean'),
                               stock_days=('n', 'sum')).to_csv(out / 'no_feed_comparison.csv')
    z.loc[z.sym.eq('7203')].to_csv(out / 'toyota_volume_forecasts.csv', index=False)


def supplementary_sensitivity(orders, volumes, f, out):
    """Inspect cold start, side pooling, boundary exclusion and discrete-size ties."""
    base = f.loc[f.reference.eq('preclosing') & f.lookback.eq('expanding')].copy()
    rows = []
    for days in [1, 3, 5]:
        for count in [10, 20, 50]:
            z = base.loc[base.history_days.ge(days) & base.order_count.ge(count) & base.closing_volume.notna()]
            cap = np.minimum(z.q75, .05*z.v25)
            rows.append(dict(min_days=days, min_orders=count, supported_stock_side_days=len(z),
                             first_eligible_date=z.date.min(), positive_unrounded=int(cap.gt(0).sum())))
    pd.DataFrame(rows).to_csv(out / 'cold_start_sensitivity.csv', index=False)
    available = base.loc[supported(base)]
    alternatives = []
    references = [('pooled', reference_orders(orders, 'preclosing_pooled')),
                  ('include_boundary', reference_orders(orders, 'preclosing', True))]
    for name, reference in references:
        for day in sorted(volumes.date.unique())[5:]:
            x = reference.loc[reference.date.lt(day)]
            grouped = x.groupby(['sym', 'side'])['size']
            p = grouped.agg(n='size', q75=lambda s: s.quantile(.75)).reset_index()
            if name == 'pooled':
                p = pd.concat([p.assign(side=side) for side in ['B', 'S']], ignore_index=True)
            p['date'] = day
            p['alternative'] = name
            alternatives.append(p)
    a = pd.concat(alternatives, ignore_index=True)
    a = a.merge(available[KEY + ['q75', 'v25', 'closing_volume']], on=KEY,
                suffixes=('_alternative', '_baseline'))
    a = a.loc[a.n.ge(20)]
    a['base_cap'] = np.minimum(a.q75_baseline, .05*a.v25)
    a['alternative_cap'] = np.minimum(a.q75_alternative, .05*a.v25)
    a['relative_change'] = (a.alternative_cap / a.base_cap.replace(0, np.nan) - 1).abs()
    a.groupby('alternative').agg(
        common_groups=('sym', 'size'), median_abs_relative_cap_change=('relative_change', 'median'),
        p95_abs_relative_cap_change=('relative_change', lambda x: x.quantile(.95)),
        fraction_changed=('relative_change', lambda x: x.gt(1e-10).mean())
    ).to_csv(out / 'boundary_and_pooling_cap_sensitivity.csv')
    a.loc[a.sym.eq('7203')].to_csv(out / 'toyota_boundary_pooling.csv', index=False)
    target = late_targets(orders)
    common = pd.MultiIndex.from_frame(a.loc[a.alternative.eq('pooled'), KEY])
    pooled = a.loc[a.alternative.eq('pooled')].rename(columns={'q75_alternative': 'q75'})
    scores = []
    for label, frame in [('same_side', available), ('pooled', pooled)]:
        z = target.merge(frame[KEY + ['q75']], on=KEY)
        z = z.loc[pd.MultiIndex.from_frame(z[KEY]).isin(common)]
        error = np.log(z.initial_size) - np.log(z.q75)
        z['loss'] = np.maximum(.75*error, -.25*error)
        z['strict'] = z.initial_size.lt(z.q75)
        z['inclusive'] = z.initial_size.le(z.q75)
        stock_day = z.groupby(KEY)[['loss', 'strict', 'inclusive']].mean()
        daily = stock_day.groupby('date').mean()
        scores.append(dict(reference=label, log_pinball=daily.loss.mean(),
                           strict_coverage=daily.strict.mean(), inclusive_coverage=daily.inclusive.mean(),
                           stock_side_days=len(stock_day), target_events=len(z)))
    pd.DataFrame(scores).to_csv(out / 'preclosing_side_comparison.csv', index=False)
    calibration = []
    for quantile in [50, 75, 90]:
        z = target.merge(available[KEY + [f'q{quantile}']], on=KEY)
        z['strict'] = z.initial_size.lt(z[f'q{quantile}'])
        z['inclusive'] = z.initial_size.le(z[f'q{quantile}'])
        daily = z.groupby(KEY)[['strict', 'inclusive']].mean().groupby('date').mean()
        calibration.append(dict(quantile=quantile/100, strict_coverage=daily.strict.mean(),
                                inclusive_coverage=daily.inclusive.mean()))
    pd.DataFrame(calibration).to_csv(out / 'discrete_quantile_coverage.csv', index=False)


def run_validation(orders, volumes, output_dir):
    """Run walk-forward validation using only prior-day data.

    The same month was also used to develop the methodology, so these results are
    development evidence rather than a fully independent out-of-sample test"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    orders = orders.copy()
    for column in ['sym', 'side', 'second']:
        orders[column] = orders[column].astype(str)
    f = forecasts(orders, volumes)
    save_pickle(f, out.parent / 'prepared' / 'walk_forward.pkl.gz')
    compare_references(f, orders, volumes, out)
    print('Comparator validation complete', flush=True)
    parameter_sensitivity(f, out)
    print('Parameter and history-window sensitivity complete', flush=True)
    no_feed_comparison(volumes, out)
    supplementary_sensitivity(orders, volumes, f, out)
    print('No-feed, pooling and boundary sensitivity complete', flush=True)
    return f
