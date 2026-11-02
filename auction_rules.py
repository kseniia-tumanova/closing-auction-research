"""Functions for setting order-size and submission-time limits from historical data.

The sizing rules use reconstructed market orders and historical closing-auction
volumes from prior trading days only. The functions return limits and diagnostics;
they do not estimate price impact, execution probability, or realized fills.

Expected inputs are prepared by `prepare_data.py`.
"""
import numpy as np
import pandas as pd


def _positive_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0


def _target_date(date):
    target = pd.Timestamp(date)
    if pd.isna(target) or target.tzinfo is not None or target != target.normalize():
        raise ValueError('date must be a naive Tokyo session date, without a time')
    return target


def _volume_history(volumes, stock, target, lookback):
    history = volumes.loc[volumes.sym.eq(stock) & volumes.date.lt(target)].sort_values('date')
    if history.duplicated(['date', 'sym']).any():
        raise ValueError('Volume input must contain exactly one row per stock/date')
    if lookback is not None:
        sessions = sorted(volumes.loc[volumes.date.lt(target), 'date'].unique())[-lookback:]
        history = history.loc[history.date.isin(sessions)]
    return history


def _invalid_volumes(history):
    return (not np.isfinite(history.closing_volume).all()
            or history.closing_volume.lt(0).any()
            or ('closing_valid' in history and not history.closing_valid.all()))


def reference_orders(orders, reference='preclosing', include_boundary=False):
    """Select a historical cohort; pooling combines sides within stock only."""
    if isinstance(orders.second.dtype, pd.CategoricalDtype):
        orders = orders.assign(second=orders.second.astype(str))
    orders = orders.loc[orders.is_market]
    x = orders if include_boundary else orders.loc[~orders.boundary_timestamp]
    if reference == 'active':
        x = x.loc[x.residual_before_close > 0].copy()
        x['size'] = x.residual_before_close
    elif reference in ('preclosing', 'preclosing_pooled', 'late60_pooled', 'final5_pooled', 'active_pooled'):
        if reference == 'active_pooled':
            x = x.loc[x.residual_before_close > 0].copy()
            x['size'] = x.residual_before_close
        else:
            start = {'preclosing': '15:25:00', 'preclosing_pooled': '15:25:00', 'late60_pooled': '15:29:00', 'final5_pooled': '15:29:55'}[reference]
            x = x.loc[x.second.ge(start) & x.second.lt('15:30:00')].copy()
            x['size'] = x.initial_size
        if reference.endswith('_pooled'):
            x['side'] = 'pooled'
    else:
        raise ValueError('Unknown reference population')
    return x[['date', 'sym', 'side', 'size']]


def maximum_order_size(orders, volumes, stock, date, direction, lot_size, *,
                       reference='preclosing', order_quantile=.75, participation=.05,
                       volume_quantile=.25, min_days=5, min_orders=20,
                       lookback=None, include_boundary=False):
    """Calculate the maximum order size for one stock, side, and date.

    By default, the function looks at similar market orders from previous trading
    days and uses the 75th percentile as the individual-order limit. It also applies
    a second limit equal to 5% of a conservative estimate of historical closing-auction
    volume. The smaller of the two limits is used.

    At least five prior trading days and 20 comparable orders are required. If there
    is not enough history, the function returns zero instead of making a recommendation.
    The final size is rounded down to the stock's trading lot.
    """
    if direction not in ('B', 'S'):
        raise ValueError('direction must be B or S')
    if not _positive_integer(lot_size):
        raise ValueError('lot_size must be a positive integer from instrument metadata')
    if not (0 < order_quantile < 1 and 0 < volume_quantile < 1 and 0 < participation <= 1):
        raise ValueError('Invalid quantile or participation limit')
    if (not _positive_integer(min_days) or not _positive_integer(min_orders)
            or (lookback is not None and (not _positive_integer(lookback) or lookback < min_days))):
        raise ValueError('Invalid minimum support or lookback')
    target = _target_date(date)
    stock = str(stock)
    # Enforce the cutoff inside the public API, even if callers pass future rows.
    history = _volume_history(volumes, stock, target, lookback)

    result = dict(date=target.strftime('%Y-%m-%d'), ticker=stock, direction=direction,
                  maximum_size=0, status='insufficient_days', history_days=len(history),
                  reference_orders=0, last_history_date=None, order_cap=np.nan, volume_estimate=np.nan,
                  liquidity_cap=np.nan, binding_constraint=None)
    if len(history):
        result['last_history_date'] = history.date.max().strftime('%Y-%m-%d')
    if len(history) < min_days:
        return result
    if _invalid_volumes(history):
        result['status'] = 'invalid_volume_history'
        return result
    x = orders.loc[orders.sym.eq(stock) & orders.date.lt(target) & orders.date.isin(history.date)]
    x = reference_orders(x, reference, include_boundary)
    side = 'pooled' if reference.endswith('_pooled') else direction
    sizes = x.loc[x.side.eq(side), 'size']
    if not np.isfinite(sizes).all() or sizes.le(0).any():
        result['status'] = 'invalid_order_history'
        return result
    result['reference_orders'] = len(sizes)
    if len(sizes) < min_orders:
        result['status'] = 'insufficient_orders'
        return result
    order_cap = float(sizes.quantile(order_quantile))
    volume_estimate = float(history.closing_volume.quantile(volume_quantile))
    liquidity_cap = participation * volume_estimate
    cap = min(order_cap, liquidity_cap)
    result.update(order_cap=order_cap, volume_estimate=volume_estimate, liquidity_cap=liquidity_cap,
                  binding_constraint='order' if order_cap <= liquidity_cap else 'liquidity')
    # Round down; never round a risk limit upward to make an order tradable.
    result['maximum_size'] = int(np.floor(cap / lot_size)) * lot_size
    result['status'] = 'ok' if result['maximum_size'] > 0 else 'below_one_lot'
    return result


def historical_timing(orders, date):
    """Find the most common second for market-order additions in the final five seconds,
    using only previous trading days. If two seconds are tied, choose the earlier one."""
    target = _target_date(date)
    second = orders.second.astype(str)
    x = orders.loc[orders.is_market & orders.date.lt(target) & ~orders.boundary_timestamp
                   & second.ge('15:29:55') & second.lt('15:30:00')]
    counts = x.second.value_counts().reindex([f'15:29:{s}' for s in range(55, 60)], fill_value=0)
    return dict(typical_bucket=counts.idxmax() if len(x) else None,
                timing_observations=len(x), timing_history_days=x.date.nunique(),
                bucket_counts={k: int(v) for k, v in counts.items()})


def submission_timing(orders, date, *, safety_margin_seconds, policy='historical_bucket'):
    """Choose a submission time within the final five seconds.

    By default, use the most common historical submission second, but move earlier
    if needed to preserve the required safety margin before 15:30. The safety margin
    is supplied externally because the data do not contain outgoing latency or clock
    uncertainty measurements.

    If `policy='earliest'`, use 15:29:55. If the required safety margin is greater
    than five seconds, no valid submission time is available.
    """
    if not np.isfinite(safety_margin_seconds) or safety_margin_seconds <= 0:
        raise ValueError('Provide a finite positive dispatch-to-acceptance safety margin')
    if policy not in ('historical_bucket', 'earliest'):
        raise ValueError('policy must be historical_bucket or earliest')
    result = historical_timing(orders, date)
    result.update(submission_time=None, safety_margin_seconds=float(safety_margin_seconds),
                  timing_policy=policy, timing_status='ok')
    if safety_margin_seconds > 5:
        result['timing_status'] = 'margin_exceeds_window'
        return result
    if policy == 'historical_bucket' and result['typical_bucket'] is None:
        result['timing_status'] = 'no_timing_history'
        return result
    typical_second = int(result['typical_bucket'][-2:]) if policy == 'historical_bucket' else 55
    # Keep timing at whole-second precision; the data do not justify subsecond timing.
    second = min(typical_second, int(np.floor(60 - safety_margin_seconds)))
    result['submission_time'] = f'15:29:{second:02d}'
    return result


def volume_only_size(volumes, stock, date, lot_size, *, participation=.05,
                     volume_quantile=.25, min_days=5, lookback=None):
    """Estimate a maximum order size using historical auction volume when individual orders are unavailable"""
    if not _positive_integer(lot_size):
        raise ValueError('Provide a valid trading unit')
    if not (0 < participation <= 1 and 0 < volume_quantile < 1) or not _positive_integer(min_days):
        raise ValueError('Invalid policy parameters')
    if lookback is not None and (not _positive_integer(lookback) or lookback < min_days):
        raise ValueError('lookback cannot be less than min_days')
    x = _volume_history(volumes, str(stock), _target_date(date), lookback)
    if len(x) < min_days or _invalid_volumes(x):
        return dict(maximum_size=0, status='insufficient_or_invalid_history', history_days=len(x))
    estimate = float(x.closing_volume.quantile(volume_quantile))
    size = int(np.floor(participation * estimate / lot_size)) * lot_size
    return dict(maximum_size=size, status='ok' if size else 'below_one_lot',
                history_days=len(x), volume_estimate=estimate)


def recommend_order(orders, volumes, stock, date, direction, lot_size, *,
                    safety_margin_seconds, timing_policy='historical_bucket', **size_policy):
    """Combine quantity and timing; return zero if either decision is unavailable."""
    size = maximum_order_size(orders, volumes, stock, date, direction, lot_size, **size_policy)
    timing = submission_timing(orders, date, safety_margin_seconds=safety_margin_seconds,
                               policy=timing_policy)
    result = {**size, **timing, 'timezone': 'Asia/Tokyo'}
    if size['maximum_size'] == 0:
        result['submission_time'] = None
    elif timing['submission_time'] is None:
        result['maximum_size'] = 0
        result['status'] = timing['timing_status']
    return result
