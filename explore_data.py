"""Inspect one raw trading session and show how the auction and trade data behave"""
from pathlib import Path
import argparse
import contextlib
import io

import pandas as pd

from prepare_data import (DATA_DIR, CHUNK_SIZE, AUCTION_COLUMNS, TRADE_COLUMNS,
                          PRE, FINAL5, CLOSE, BOUNDARY, canonical_time, read_chunks,
                          read_auction, coverage_diagnostic)

EXAMPLE_DATE = '20250106'
EXAMPLE_SYMBOL = '7203'


def walkthrough(data_dir=DATA_DIR, date=EXAMPLE_DATE, symbol=EXAMPLE_SYMBOL,
                chunk_size=CHUNK_SIZE):
    """Educational output; every example and number comes from the raw files."""
    date = pd.Timestamp(date).strftime('%Y%m%d')
    symbol = str(symbol)
    ap = Path(data_dir) / f'auction{date}.csv'
    tp = Path(data_dir) / f'trade{date}.csv'
    orders, audit, auction = read_auction(ap, date, chunk_size)
    selected, samples, symbols = [], [], set()
    n = 0
    ranges = {}
    for t in read_chunks(tp, TRADE_COLUMNS, chunk_size):
        for c in ['time', 'exchTime']:
            t[c] = canonical_time(t[c])
            lo, hi = ranges.get(c, (t[c].min(), t[c].max()))
            ranges[c] = (min(lo, t[c].min()), max(hi, t[c].max()))
        n += len(t)
        symbols.update(t.sym)
        if not samples:
            samples.append(t.head(5))
        selected.append(t.loc[t.sym.eq(symbol)])
    trades = pd.concat(selected, ignore_index=True)
    trades['qty'] = pd.to_numeric(trades.qty, errors='raise')
    trades['price'] = pd.to_numeric(trades.price, errors='raise')
    a = auction.loc[auction.sym.eq(symbol)].copy()
    o = orders.loc[orders.sym.eq(symbol)]
    print(f'RAW DATA WALKTHROUGH | {date} | {symbol} | Tokyo local time\n')
    print(f'Auction: {audit["rows"]:,} rows; {audit["symbols"]:,} symbols; columns {AUCTION_COLUMNS}')
    print(f'Time range: {audit["time_min"]} to {audit["time_max"]}')
    print(auction.head(5).to_string(index=False))
    print(f'\nTrade: {n:,} rows; {len(symbols):,} symbols; columns {TRADE_COLUMNS}')
    print(f'Time ranges: {ranges}')
    print(samples[0].to_string(index=False))
    print('\nSymbols are strings; letters are valid. Examples:', sorted(symbols)[:20])
    print('\nORDER LIFECYCLES')
    print('README: volume is total shares at a price level; chg is this order update.')
    print('One initial positive chg, followed by reductions; balance = cumulative chg.')
    print('A removal before an auction may be a cancellation or execution-related removal;')
    print('the supplied fields do not distinguish those causes. Exactly 15:30 is terminal cleanup.')
    # These raw IDs illustrate an entry/removal and multiple partial reductions.
    # For another date/symbol, fall back to the most frequently updated IDs.
    preferred = [7761, 16160, 374337]
    ids = [v for v in preferred if v in set(a.orderId)]
    ids += [v for v in a.orderId.value_counts().index if v not in ids][:max(0, 3-len(ids))]
    for order_id in ids:
        trace = a.loc[a.orderId.eq(order_id)].copy()
        trace['individual_running_balance'] = trace.chg.cumsum()
        print(f'\norderId={order_id} (key also includes date and symbol)')
        print(trace.to_string(index=False))
    mismatch = a.loc[a.chg.gt(0) & a.volume.ne(a.chg)].head(1)
    if len(mismatch):
        row = mismatch.iloc[0]
        print(f'\nConcrete distinction: order {row.orderId} added {row.chg:,} shares, while')
        print(f'the reported level volume was {row.volume:,}. volume != individual order size.')
    print('\nMARKET ORDERS: price == 0, as defined in the supplied README')
    print(a.loc[a.price.eq(0)].head(8).to_string(index=False))
    groups = [
        ('ALL PRE-CLOSING UPDATES [15:25,15:30)', a.time.ge(PRE) & a.time.lt(CLOSE)),
        ('PRE-CLOSING MARKET ADDITIONS', a.time.ge(PRE) & a.time.lt(CLOSE) & a.price.eq(0) & a.chg.gt(0)),
        ('FINAL FIVE SECONDS [15:29:55,15:30)', a.time.ge(FINAL5) & a.time.lt(CLOSE)),
        ('EXACT BOUNDARY RECORDS 15:29:59.999999', a.time.eq(BOUNDARY)),
    ]
    for label, mask in groups:
        z = a.loc[mask]
        print(f'\n{label}: {len(z):,} records (first 12 shown)')
        print(z.head(12).to_string(index=False))
    print('Boundary concentration is unusual, not proof that records are invalid or synthetic.')
    close = trades.loc[trades.exchTime.eq(CLOSE)]
    print('\nCLOSING TRADE(S): identify with exchange time, not delayed capture time')
    print(close.to_string(index=False))
    volume = close.qty.sum(min_count=1) if len(close) and close.qty.notna().all() else float('nan')
    print(f'Realized observed closing-auction volume: {volume:,.0f}')
    price = close.price.iloc[0] if len(close) and close.price.nunique() == 1 else float('nan')
    v = pd.DataFrame([dict(sym=symbol, closing_volume=volume, closing_price=price)])
    coverage = coverage_diagnostic(o, v)
    print('\nBOOK-COVERAGE DIAGNOSTIC (not a matching-engine reconstruction)')
    print('Eligible = market orders, buys at/above close price, sells at/below close price.')
    print('Each side is compared with the same executed volume; do not sum both sides.')
    print(coverage.to_string(index=False))
    print('A positive shortfall means the recorded eligible side cannot explain the print.')
    print('The extract lacks flags to identify the missing liquidity or transformation mechanism.')
    return {'coverage': coverage, 'order_examples': a.loc[a.orderId.isin(ids)],
            'closing_trades': close}


def save_walkthrough(data_dir, output_dir, date=EXAMPLE_DATE, symbol=EXAMPLE_SYMBOL,
                     chunk_size=CHUNK_SIZE):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        tables = walkthrough(data_dir, date, symbol, chunk_size)
    (out / 'raw_walkthrough.txt').write_text(buffer.getvalue(), encoding='utf-8')
    for name, table in tables.items():
        table.to_csv(out / f'example_{name}.csv', index=False)
    return buffer.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--date', default=EXAMPLE_DATE)
    parser.add_argument('--symbol', default=EXAMPLE_SYMBOL)
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    walkthrough(args.data_dir, args.date, args.symbol, args.chunk_size)
