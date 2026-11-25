from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.options import filter_option_quotes, filter_smile_quotes

root = Path(__file__).resolve().parents[1]
raw = pd.read_parquet(root / 'unpivoted_bid_offer_train_clean.parquet')
raw['date'] = pd.to_datetime(raw['date'])
raw['expiration'] = pd.to_datetime(raw['expiration'])
liquid = filter_option_quotes(raw)
clean = filter_smile_quotes(liquid)
entry = clean.loc[clean.time_to_expiry_days >= 30]
dates = pd.DatetimeIndex(sorted(raw.date.unique()))
tickers = sorted(raw.actual_underlying_ticker.unique())
index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'future'])
counts = pd.DataFrame(index=index)
for name, frame in [('raw', raw), ('liquid', liquid), ('filtered', clean), ('entry30', entry)]:
    series = frame.groupby(['date', 'actual_underlying_ticker']).option_ticker.nunique()
    series.index.names = ['date', 'future']
    counts[name] = series.reindex(index, fill_value=0)

summary = []
low = {}
reasons = {}
for ticker in tickers:
    original = raw.loc[raw.actual_underlying_ticker == ticker]
    daily = counts.xs(ticker, level='future')
    active = daily.loc[original.date.min():original.date.max()]
    row = dict(ticker=ticker, first=str(original.date.min().date()), last=str(original.date.max().date()),
               observed_days=int((daily.raw > 0).sum()), active_days=len(active))
    for name in counts.columns:
        row[name] = dict(mean=round(float(daily[name].mean()), 2), min=int(daily[name].min()),
                         max=int(daily[name].max()), zero_days=int((daily[name] == 0).sum()),
                         active_mean=round(float(active[name].mean()), 2), active_min=int(active[name].min()))
    summary.append(row)
    if ticker in ('COJ21','COK21'):
        low[ticker] = active.loc[active.filtered <= 4].reset_index().assign(date=lambda x:x.date.dt.strftime('%Y-%m-%d')).to_dict('records')
        bid, ask = original.option_bid, original.option_ask
        valid_market = np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask >= bid)
        live = (original.time_to_expiry_days > 0) & np.isfinite(original.time_to_expiry_days)
        mid = (bid + ask) / 2
        adequate = (mid >= .05) & (mid > 0) & ((ask-bid) <= .5*mid)
        reasons[ticker] = dict(rows=len(original), expired_or_invalid_ttm=int((~live).sum()),
            invalid_bid_ask_live=int((live & ~valid_market).sum()),
            low_premium_or_wide_spread=int((live & valid_market & ~adequate).sum()),
            rejected_smile=len(liquid.loc[liquid.actual_underlying_ticker == ticker])-len(clean.loc[clean.actual_underlying_ticker == ticker]),
            kept=len(clean.loc[clean.actual_underlying_ticker == ticker]))

out = root / 'output' / 'quote_availability'
out.mkdir(parents=True, exist_ok=True)
report = dict(first=str(dates.min().date()), last=str(dates.max().date()), days=len(dates),
              definition='Distinct option tickers; all dataset dates include zeros. Active window: first to last raw observation. Few = at most 4 filtered contracts.',
              summary=summary, low_days=low, exclusions=reasons)
(out / 'availability.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
lines = [report['definition'], f"Period: {report['first']} to {report['last']}; {len(dates)} market dates.",
         'ticker | raw mean/min/max | filtered mean/min/max | entry>=30d mean/min/max | filtered zeros | active filtered mean/min | raw first/last']
for row in summary:
    groups = ['{mean}/{min}/{max}'.format(**row[key]) for key in ('raw','filtered','entry30')]
    lines.append(f"{row['ticker']} | {' | '.join(groups)} | {row['filtered']['zero_days']} | {row['filtered']['active_mean']}/{row['filtered']['active_min']} | {row['first']} / {row['last']}")
for ticker in low:
    lines.append(f'\n{ticker}: few/zero days within observed history\n' + pd.DataFrame(low[ticker]).to_string(index=False))
    lines.append('Exclusions: '+str(reasons[ticker]))
(out / 'availability.txt').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
