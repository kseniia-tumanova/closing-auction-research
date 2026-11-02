"""Prepare the raw auction and trade data for analysis.

The script validates the input files, reconstructs individual orders from `chg`,
and aggregates trade data into daily closing-auction volumes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

DATA_DIR = Path('/Users/xeniat/Desktop/archive')
CHUNK_SIZE = 100_000
PRE = '0D15:25:00.000000000'
FINAL5 = '0D15:29:55.000000000'
CLOSE = '0D15:30:00.000000000'
BOUNDARY = '0D15:29:59.999999000' 
AUCTION_COLUMNS = ['sym', 'price', 'buySell', 'volume', 'time', 'chg', 'orderId']
TRADE_COLUMNS = ['sym', 'time', 'price', 'qty', 'exchTime']
TIME_PATTERN = r'0D(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{1,9}'


def save_csv(frame, path):
    """Save a table safely without leaving a partial output file"""
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    compression = 'gzip' if path.name.endswith('.gz') else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def save_pickle(frame, path):
    """Save a compressed table safely without leaving a partial output file"""
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    frame.to_pickle(temporary, compression='gzip')
    temporary.replace(path)


class DataQualityError(ValueError):
    """Raised when the raw data fail validation checks"""


def discover_pairs(data_dir, selected_dates=None):
    """Find matching auction and trade files and fail if any date is incomplete"""
    data_dir = Path(data_dir)
    found = {}
    for kind in ['auction', 'trade']:
        found[kind] = {p.stem[-8:]: p for p in data_dir.glob(f'{kind}*.csv')
                       if re.fullmatch(kind + r'\d{8}\.csv', p.name)}
    dates = sorted(set(found['auction']) | set(found['trade']))
    if selected_dates is not None:
        dates = sorted({pd.Timestamp(d).strftime('%Y%m%d') for d in selected_dates})
    if not dates:
        raise DataQualityError(f'No auctionYYYYMMDD.csv/tradeYYYYMMDD.csv pairs in {data_dir}')
    missing = [(d, k) for d in dates for k in found if d not in found[k]]
    if missing:
        raise DataQualityError(f'Unmatched or absent selected dates: {missing}')
    for d in dates:
        pd.to_datetime(d, format='%Y%m%d', errors='raise')
    return [(d, found['auction'][d], found['trade'][d]) for d in dates]


def read_chunks(path, columns, chunk_size=CHUNK_SIZE):
    """Read only the columns needed for reconstruction and the requested audit"""
    header = pd.read_csv(path, sep='|', nrows=0).columns.tolist()
    if header != columns:
        raise DataQualityError(f'{path.name}: expected {columns}, received {header}')
    yield from pd.read_csv(path, sep='|', usecols=columns, dtype=str,
                          chunksize=chunk_size, on_bad_lines='error')


def canonical_time(series):
    """Validate clock ranges and preserve all nine fractional digits without floats"""
    valid = series.str.fullmatch(TIME_PATTERN, na=False)
    if not valid.all():
        examples = series.loc[~valid].head(3).tolist()
        raise DataQualityError(f'Malformed timestamps ({int((~valid).sum())}): {examples}')
    # Padding permits equivalent six- and nine-digit encodings; lexicographic
    # comparison is then exact for local timestamps on the same session date.
    return series.str.pad(20, side='right', fillchar='0')


def _numeric(x, columns, integers=()):
    for c in columns:
        x[c] = pd.to_numeric(x[c], errors='coerce')
        valid = x[c].isna() | np.isfinite(x[c])
        if not valid.all():
            raise DataQualityError(f'Non-finite values in {c}')
        if c in integers and (x[c].dropna() % 1 != 0).any():
            raise DataQualityError(f'Non-integer shares or order identifier in {c}')


def reconstruct_orders(x, date):
    """Reconstruct individual orders from their `chg` history.

       We expect each order ID to appear first as one positive addition, followed only
       by reductions. If the same ID later gets another positive addition or changes
       price or side, the data no longer fit this order-lifecycle assumption, so the
       reconstruction stops instead of treating those rows as the same order.
    """
    key = ['sym', 'orderId']
    group = x.groupby(key, sort=False, observed=True)
    first = ~x.duplicated(key)
    running = group.chg.cumsum()
    level_delta = x.groupby(['sym', 'buySell', 'price'], sort=False).volume.diff().fillna(x.volume)
    checks = {
        'first_nonpositive': int((first & x.chg.le(0)).sum()),
        'repeat_positive': int((~first & x.chg.gt(0)).sum()),
        'zero_chg': int(x.chg.eq(0).sum()),
        'changing_price': int(group.price.nunique().gt(1).sum()),
        'changing_side': int(group.buySell.nunique().gt(1).sum()),
        'negative_running_balance': int(running.lt(0).sum()),
        'timestamp_decreases': int(x.time.lt(x.time.shift().fillna(x.time.iloc[0])).sum()),
        'level_delta_mismatch': int(level_delta.ne(x.chg).sum()),
        'after_close_rows': int(x.time.gt(CLOSE).sum()),
        'positive_exact_close': int((x.time.eq(CLOSE) & x.chg.gt(0)).sum()),
        'duplicate_rows': int(x.duplicated().sum()),
    }
    failures = {k: v for k, v in checks.items() if v}
    if failures:
        raise DataQualityError(f'Auction lifecycle ambiguity on {date}: {failures}')
    births = x.loc[first, key + ['buySell', 'price', 'time', 'chg']].rename(
        columns={'buySell': 'side', 'time': 'initial_time', 'chg': 'initial_size'}).copy()
    # volume is aggregate depth at a price level. chg belongs to the identified
    # order, and is therefore the quantity used for its individual balance.
    pre = x.loc[x.time.lt(CLOSE)].groupby(key, sort=False).chg.sum()
    births = births.join(pre.rename('residual_before_close'), on=key)
    births['residual_before_close'] = births.residual_before_close.fillna(0).astype('int64')
    births = births.join(group.chg.sum().rename('final_balance'), on=key)
    births['date'] = pd.Timestamp(date)
    births['is_market'] = births.price.eq(0)
    births['is_preclosing_addition'] = births.initial_time.ge(PRE) & births.initial_time.lt(CLOSE)
    births['is_final5_addition'] = births.initial_time.ge(FINAL5) & births.initial_time.lt(CLOSE)
    births['boundary_timestamp'] = births.initial_time.eq(BOUNDARY)
    births['second'] = births.initial_time.str[2:10]
    checks.update(unique_orders=len(births), nonzero_final_balances=int(births.final_balance.ne(0).sum()),
                  exact_close_removals=int(x.time.eq(CLOSE).sum()),
                  boundary_rows=int(x.time.eq(BOUNDARY).sum()),
                  active_before_close=int(births.residual_before_close.gt(0).sum()),
                  partially_reduced=int((births.residual_before_close.gt(0) &
                                         births.residual_before_close.lt(births.initial_size)).sum()))
    return births.reset_index(drop=True), checks


def read_auction(path, date, chunk_size=CHUNK_SIZE):
    frames = []
    for x in read_chunks(path, AUCTION_COLUMNS, chunk_size):
        if x.isna().any().any():
            raise DataQualityError(f'{path.name}: missing auction fields')
        x['time'] = canonical_time(x.time)
        _numeric(x, ['price', 'volume', 'chg', 'orderId'], ['volume', 'chg', 'orderId'])
        if x.isna().any().any() or not x.buySell.isin(['B', 'S']).all():
            raise DataQualityError(f'{path.name}: invalid numbers or side codes')
        if x.price.lt(0).any() or x.volume.lt(0).any() or x.orderId.lt(0).any():
            raise DataQualityError(f'{path.name}: negative price/depth/order identifier')
        x[['volume', 'chg', 'orderId']] = x[['volume', 'chg', 'orderId']].astype('int64')
        frames.append(x)
    if not frames:
        raise DataQualityError(f'{path.name}: empty file')
    x = pd.concat(frames, ignore_index=True)
    orders, checks = reconstruct_orders(x, date)
    audit = dict(date=date, kind='auction', file=path.name, rows=len(x), symbols=x.sym.nunique(),
                 missing_qty=0, malformed_timestamps=0, missing_other_fields=0,
                 invalid_side=0, negative_price_or_volume=0,
                 time_min=x.time.min(), time_max=x.time.max(), **checks)
    return orders, audit, x


def aggregate_trade(path, date, chunk_size=CHUNK_SIZE):
    totals, hashes, sample = [], [], []
    rows = missing = preclose = after = 0
    capture_min = exchange_min = None
    capture_max = exchange_max = None
    for x in read_chunks(path, TRADE_COLUMNS, chunk_size):
        if x.drop(columns='qty').isna().any().any():
            raise DataQualityError(f'{path.name}: missing trade identifier/price/time field')
        x['time'] = canonical_time(x.time)
        x['exchTime'] = canonical_time(x.exchTime)
        originally_missing = x.qty.isna()
        _numeric(x, ['price', 'qty'], ['qty'])
        x[['price', 'qty']] = x[['price', 'qty']].astype(float)
        if x.price.isna().any() or (x.qty.isna() & ~originally_missing).any():
            raise DataQualityError(f'{path.name}: malformed numeric value')
        if x.price.le(0).any() or x.qty.dropna().le(0).any():
            raise DataQualityError(f'{path.name}: nonpositive trade price/quantity')
        if x.time.lt(x.exchTime).any():
            raise DataQualityError(f'{path.name}: capture timestamp precedes exchange timestamp')
        rows += len(x)
        missing += int(x.qty.isna().sum())
        if x.qty.isna().any():
            sample.append(x.loc[x.qty.isna()].assign(date=date))
        preclose += int((x.exchTime.ge(PRE) & x.exchTime.lt(CLOSE)).sum())
        after += int(x.exchTime.gt(CLOSE).sum())
        capture_min = min(capture_min or x.time.min(), x.time.min())
        capture_max = max(capture_max or x.time.max(), x.time.max())
        exchange_min = min(exchange_min or x.exchTime.min(), x.exchTime.min())
        exchange_max = max(exchange_max or x.exchTime.max(), x.exchTime.max())
        # Hashes locate possible cross-chunk duplicates without retaining every
        # trade. Candidates are verified against all original fields below.
        hashes.append(pd.util.hash_pandas_object(x, index=False).to_numpy())
        close = x.exchTime.eq(CLOSE)
        nonclose = x.exchTime.lt(CLOSE)
        z = x[['sym']].copy()
        z['closing_volume'] = x.qty.where(close, 0).fillna(0)
        z['nonclosing_volume'] = x.qty.where(nonclose, 0).fillna(0)
        z['closing_prints'] = close.astype(int)
        z['missing_closing_qty'] = (close & x.qty.isna()).astype(int)
        z['missing_nonclosing_qty'] = (nonclose & x.qty.isna()).astype(int)
        part = z.groupby('sym', sort=False).sum()
        cp = x.loc[close].groupby('sym').price.agg(closing_price_min='min', closing_price_max='max')
        totals.append(part.join(cp))
    if not rows:
        raise DataQualityError(f'{path.name}: empty file')
    h = pd.Series(np.concatenate(hashes))
    candidates = set(h.loc[h.duplicated(keep=False)].tolist())
    duplicate_rows = 0
    duplicate_examples = pd.DataFrame(columns=TRADE_COLUMNS)
    if candidates:
        matches = []
        for x in read_chunks(path, TRADE_COLUMNS, chunk_size):
            x['time'] = canonical_time(x.time)
            x['exchTime'] = canonical_time(x.exchTime)
            _numeric(x, ['price', 'qty'], ['qty'])
            x[['price', 'qty']] = x[['price', 'qty']].astype(float)
            mask = pd.util.hash_pandas_object(x, index=False).isin(candidates)
            matches.append(x.loc[mask])
        matches = pd.concat(matches, ignore_index=True)
        duplicate_rows = int(matches.duplicated().sum())
        duplicate_examples = matches.loc[matches.duplicated(keep=False)].copy()
    d = pd.concat(totals).groupby(level=0).agg(
        closing_volume=('closing_volume', 'sum'), nonclosing_volume=('nonclosing_volume', 'sum'),
        closing_prints=('closing_prints', 'sum'), missing_closing_qty=('missing_closing_qty', 'sum'),
        missing_nonclosing_qty=('missing_nonclosing_qty', 'sum'),
        closing_price_min=('closing_price_min', 'min'), closing_price_max=('closing_price_max', 'max'))
    d['closing_valid'] = d.missing_closing_qty.eq(0)
    d['nonclosing_valid'] = d.missing_nonclosing_qty.eq(0)
    d.loc[~d.closing_valid, 'closing_volume'] = np.nan
    d.loc[~d.nonclosing_valid, 'nonclosing_volume'] = np.nan
    d['closing_price'] = d.closing_price_min.where(d.closing_price_min.eq(d.closing_price_max))
    d['duplicate_trade_rows'] = duplicate_examples.groupby('sym').size().reindex(d.index, fill_value=0)
    audit = dict(date=date, kind='trade', file=path.name, rows=rows, symbols=len(d),
                 missing_qty=missing, malformed_timestamps=0, missing_other_fields=0,
                 negative_price_or_volume=0, duplicate_rows=duplicate_rows,
                 time_min=capture_min, time_max=capture_max, exchange_time_min=exchange_min,
                 exchange_time_max=exchange_max, preclosing_trades=preclose, after_close_trades=after,
                 closing_prints=int(d.closing_prints.sum()))
    missing_examples = pd.concat(sample, ignore_index=True) if sample else pd.DataFrame(columns=TRADE_COLUMNS + ['date'])
    return d, audit, missing_examples, duplicate_examples


def coverage_diagnostic(orders, volumes):
    """Compare recorded price-eligible pre-close volume with the actual closing-auction volume"""
    x = orders.loc[orders.residual_before_close.gt(0)].merge(
        volumes[['sym', 'closing_volume', 'closing_price']], on='sym', how='left')
    eligible = x.is_market | (x.side.eq('B') & x.price.ge(x.closing_price)) | (
        x.side.eq('S') & x.price.le(x.closing_price))
    x['recorded_eligible'] = x.residual_before_close.where(eligible, 0)
    s = x.groupby(['date', 'sym', 'side'], observed=True).agg(
        recorded_eligible=('recorded_eligible', 'sum'), recorded_total=('residual_before_close', 'sum'))
    s = s.reset_index().merge(volumes[['sym', 'closing_volume', 'closing_price']], on='sym')
    s['eligible_shortfall'] = s.closing_volume - s.recorded_eligible
    s.loc[s.closing_price.isna(), ['recorded_eligible', 'eligible_shortfall']] = np.nan
    return s


def prepare_data(data_dir, output_dir, *, selected_dates=None, chunk_size=CHUNK_SIZE):
    """Prepare the raw files for the sizing analysis.

    Reconstruct the auction orders, aggregate daily trade volumes, and return the
    market-order history, stock-day volume table, and audit results. Full order
    tables are saved separately for inspection.

    A missing 15:30 print in an otherwise complete file is treated as zero observed
    closing volume, although the data cannot rule out silent feed truncation
    """
    out = Path(output_dir)
    tables = out / 'tables'
    prepared = out / 'prepared'
    tables.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    audits, markets, daily, coverage, manifest = [], [], [], [], []
    try:
        pairs = discover_pairs(data_dir, selected_dates)
        for date, ap, tp in pairs:
            print(f'Preparing {date}: auction lifecycle and chunked trades', flush=True)
            orders, aq, _events = read_auction(ap, date, chunk_size)
            del _events
            if aq['time_max'] != CLOSE or aq['nonzero_final_balances']:
                raise DataQualityError(f'{ap.name}: incomplete terminal cleanup; file completeness is uncertain')
            trades, tq, missing, duplicates = aggregate_trade(tp, date, chunk_size)
            if tq['exchange_time_max'] != CLOSE or tq['preclosing_trades'] or tq['after_close_trades']:
                raise DataQualityError(f'{tp.name}: unexpected session coverage or exchange-time regime')
            universe = pd.Index(sorted(set(orders.sym) | set(trades.index)), name='sym')
            v = trades.reindex(universe)
            absent = ~v.index.isin(trades.index)
            fill_zero = ['closing_volume', 'nonclosing_volume', 'closing_prints',
                         'missing_closing_qty', 'missing_nonclosing_qty', 'duplicate_trade_rows']
            v.loc[absent, fill_zero] = 0
            v.loc[absent, ['closing_valid', 'nonclosing_valid']] = True
            v['closing_status'] = np.select(
                [~v.closing_valid.astype(bool), v.closing_prints.gt(0)],
                ['unknown_quantity', 'observed_print'], default='no_print_observed_zero')
            v['in_auction'] = v.index.isin(set(orders.sym))
            v['in_trade'] = v.index.isin(trades.index)
            v['date'] = pd.Timestamp(date)
            v = v.reset_index()
            coverage.append(coverage_diagnostic(orders, v))
            # Compression keeps the complete reconstructed order tables portable;
            # these are outputs, never prerequisites for a fresh run.
            save_csv(orders, prepared / f'orders_{date}.csv.gz')
            market_columns = ['date', 'sym', 'side', 'initial_size', 'residual_before_close',
                              'is_market', 'is_preclosing_addition', 'is_final5_addition',
                              'boundary_timestamp', 'second']
            markets.append(orders.loc[orders.is_market, market_columns].copy())
            daily.append(v)
            audits.extend([aq, tq])
            if len(missing):
                missing.to_csv(tables / f'missing_trade_qty_{date}.csv', index=False)
            if len(duplicates):
                duplicates.to_csv(tables / f'duplicate_trade_rows_{date}.csv', index=False)
            for path in [ap, tp]:
                # Record exact source content, not only filenames or timestamps.
                digest = hashlib.sha256()
                with path.open('rb') as fh:
                    for block in iter(lambda: fh.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                manifest.append(dict(file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
            pd.DataFrame(audits).to_csv(out / 'diagnostics.csv', index=False)
    except Exception as exc:
        (out / 'preparation_failure.json').write_text(json.dumps({'error': str(exc)}, indent=2))
        raise
    orders = pd.concat(markets, ignore_index=True)
    for col in ['sym', 'side', 'second']:
        orders[col] = orders[col].astype('category')
    volumes = pd.concat(daily, ignore_index=True).sort_values(['date', 'sym']).reset_index(drop=True)
    for col in ['closing_valid', 'nonclosing_valid']:
        volumes[col] = volumes[col].astype(bool)
    if volumes.duplicated(['date', 'sym']).any():
        raise DataQualityError('Duplicate stock/date volume rows')
    save_pickle(orders, prepared / 'market_orders.pkl.gz')
    save_csv(volumes, prepared / 'closing_volumes.csv.gz')
    reread = pd.read_csv(prepared / 'closing_volumes.csv.gz', dtype={'sym': str}, parse_dates=['date'])
    pd.testing.assert_frame_equal(volumes, reread, check_dtype=False, check_exact=False)
    summary = dict(market_orders=len(orders), all_order_keys=int(pd.DataFrame(audits).unique_orders.sum()),
                   stock_days=len(volumes), symbols=volumes.sym.nunique(), sessions=volumes.date.nunique(),
                   book_side_days=sum(len(frame) for frame in coverage))
    (out / 'preparation_summary.json').write_text(json.dumps(summary, indent=2))
    book = pd.concat(coverage, ignore_index=True)
    save_csv(book, tables / 'book_coverage.csv.gz')
    book_saved = pd.read_csv(tables / 'book_coverage.csv.gz', dtype={'sym': str}, parse_dates=['date'])
    pd.testing.assert_frame_equal(book, book_saved, check_dtype=False, check_exact=False)
    (out / 'input_manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'preparation_failure.json').unlink(missing_ok=True)
    return orders, volumes, pd.DataFrame(audits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'output')
    parser.add_argument('--dates', nargs='+')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    prepare_data(args.data_dir, args.output_dir, selected_dates=args.dates, chunk_size=args.chunk_size)
