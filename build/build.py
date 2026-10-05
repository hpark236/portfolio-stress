"""Build the data behind the site: copy-trading simulations, performance, and stress tests.

    .venv/bin/python build/ptr.py      # refresh Pelosi filings
    .venv/bin/python build/build.py    # prices, simulations, stress tests -> data/site.json
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
START = "2007-01-01"  # long enough for the 2008 window

# Tickers that changed or were delisted since the filing.
RENAME = {"FB": "META", "GOOG": "GOOGL"}
SKIP = {"WORK", "BFET"}  # Slack (acquired 2021), BF Enterprises (no price history)
BENCH = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "NANC": "Democrats in Congress ETF (NANC)"}
# Factors for the stress model: broad market, semiconductors, oil, long Treasuries, gold, US dollar, defence, bitcoin.
FACTORS = {"SPY": "US stocks", "SMH": "Semiconductors", "USO": "Oil", "TLT": "Long Treasuries", "GLD": "Gold", "UUP": "US dollar", "ITA": "Defence", "BTC-USD": "Bitcoin"}

# Historical windows (first and last trading day). Moves of the factors in these windows also
# drive the factor-model estimate for holdings that did not trade at the time.
WINDOWS = [
    ("2008 financial crisis", "2008-09-12", "2009-03-09", "Lehman failure to the market low."),
    ("2019 Abqaiq attack", "2019-09-13", "2019-09-17", "Drone strike on Saudi oil processing. Brent's largest one-day jump on record."),
    ("COVID crash", "2020-02-19", "2020-03-23", "Pandemic sell-off from the S&P 500 high to the low."),
    ("COVID recovery bull market", "2020-03-23", "2021-12-31", "Stimulus-driven bull market."),
    ("Russia invades Ukraine", "2022-02-16", "2022-03-08", "Build-up, invasion and the oil spike."),
    ("2022 bear market", "2022-01-03", "2022-10-12", "Rate hikes. Stocks and bonds fell together."),
    ("2023 AI bull market", "2023-01-03", "2023-12-29", "Mega-cap tech and semiconductors led."),
    ("April 2025 tariff shock", "2025-04-02", "2025-04-08", "Reciprocal tariff announcement to the low before the pause."),
    ("June 2025 Iran war, Hormuz threat", "2025-06-12", "2025-06-20", "Israel strikes Iran. Iran threatens to close the Strait of Hormuz."),
    ("June 2025 ceasefire, Hormuz stays open", "2025-06-20", "2025-06-25", "US strikes, a vote to close the strait, then a ceasefire. Oil gives back the war premium."),
]

# Hypothetical shocks with no clean precedent, stated as moves in the factors. Editable on the site.
HYPOTHETICAL = [
    ("Strait of Hormuz closed for a month", {"SPY": -0.12, "SMH": -0.18, "USO": 0.60, "TLT": 0.02, "GLD": 0.10, "UUP": 0.03, "ITA": 0.06, "BTC-USD": -0.15},
     "About a fifth of world oil supply stops. Oil roughly 1.5 to 2 times the 1990 Gulf War and 2022 spikes; stocks fall less than in 2020 because it is a supply shock, not a demand collapse."),
    ("Hormuz reopens after a closure", {"SPY": 0.05, "SMH": 0.08, "USO": -0.30, "TLT": 0.00, "GLD": -0.04, "UUP": -0.01, "ITA": -0.04, "BTC-USD": 0.06},
     "The war premium comes out of oil quickly, as in June 2025 and March 1991."),
    ("NATO-Russia escalation over Ukraine", {"SPY": -0.15, "SMH": -0.22, "USO": 0.30, "TLT": 0.05, "GLD": 0.12, "UUP": 0.04, "ITA": 0.10, "BTC-USD": -0.20},
     "A larger version of February 2022: energy shock, flight to dollars, gold and Treasuries, defence stocks up."),
    ("New pandemic", {"SPY": -0.30, "SMH": -0.30, "USO": -0.55, "TLT": 0.12, "GLD": 0.02, "UUP": 0.05, "ITA": -0.35, "BTC-USD": -0.40},
     "Close to February to March 2020, before any stimulus."),
    ("Rate shock", {"SPY": -0.12, "SMH": -0.20, "USO": 0.05, "TLT": -0.15, "GLD": -0.05, "UUP": 0.05, "ITA": -0.08, "BTC-USD": -0.25},
     "Inflation surprise and a jump in long yields, as in 2022."),
    ("12-month bull market", {"SPY": 0.25, "SMH": 0.45, "USO": 0.05, "TLT": -0.02, "GLD": 0.05, "UUP": -0.04, "ITA": 0.15, "BTC-USD": 0.60},
     "Roughly 2023: broad gains led by semiconductors."),
]


def prices(tickers):
    df = yf.download(sorted(set(tickers)), start=START, auto_adjust=True, progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def next_trading_day(px, d, after=False):
    d = pd.Timestamp(d)
    idx = px.index[px.index > d] if after else px.index[px.index >= d]
    return idx[0] if len(idx) else None


# ---------- copy-trading simulation ----------

def simulate(trades, px, when):
    """
    Replay the trades with positions sized by the midpoint of each reported dollar range.
    when = "traded": fill at the close on the trade date (what the member got).
    when = "filed": fill at the close of the first trading day after the filing (what a copier gets).
    Options are treated as buying the stock with the same dollar amount, as copy-trading apps do.
    Returns daily time-weighted returns of the invested positions and the current holdings.
    """
    shares = {}
    events = []
    for t in trades:
        day = next_trading_day(px, t["traded"]) if when == "traded" else next_trading_day(px, t["filed"], after=True)
        if day is not None:
            events.append((day, t))
    events.sort(key=lambda e: e[0])
    tickers = sorted({t["ticker"] for _, t in events})
    p = px[tickers].ffill()
    first = events[0][0]
    p = p[p.index >= first]
    values = []
    ei = 0
    prev_value = None
    rets = []
    for day, row in p.iterrows():
        # returns accrue on yesterday's holdings before today's trades
        value = sum(n * row[k] for k, n in shares.items() if n > 0 and not math.isnan(row[k]))
        if prev_value:
            rets.append((day, value / prev_value - 1))
        while ei < len(events) and events[ei][0] == day:
            _, t = events[ei]
            k = t["ticker"]
            price = row[k]
            if not math.isnan(price) and price > 0:
                usd = (t["amount_lo"] + t["amount_hi"]) / 2
                if t["type"] == "P":
                    shares[k] = shares.get(k, 0) + usd / price
                elif t["type"] == "S":
                    held = shares.get(k, 0)
                    if not t["partial"]:
                        shares[k] = 0
                    else:
                        shares[k] = max(0.0, held - usd / price)
            ei += 1
        prev_value = sum(n * row[k] for k, n in shares.items() if n > 0 and not math.isnan(row[k])) or None
    r = pd.Series({d: x for d, x in rets}).sort_index()
    last = p.ffill().iloc[-1]
    hold = {k: n * last[k] for k, n in shares.items() if n > 0}
    tot = sum(hold.values())
    return r, {k: v / tot for k, v in sorted(hold.items(), key=lambda kv: -kv[1])}


def lag_effect(trades, px):
    """Price move between her fill (trade date) and the copier's fill (day after filing), for buys and for sales."""
    out = {}
    for side, label in (("P", "buys"), ("S", "sales")):
        moves, usd = [], []
        for t in trades:
            if t["type"] != side:
                continue
            s = px[t["ticker"]].dropna()
            a, b = s[s.index >= t["traded"]], s[s.index > t["filed"]]
            if len(a) and len(b):
                moves.append(b.iloc[0] / a.iloc[0] - 1)
                usd.append((t["amount_lo"] + t["amount_hi"]) / 2)
        m, w = np.array(moves), np.array(usd)
        out[label] = {"n": len(m), "median": float(np.median(m)), "dollarWeighted": float((m * w).sum() / w.sum()), "higher": float((m > 0).mean())}
    return out


def stats(r, bench=None):
    r = r.dropna()
    if len(r) < 20:
        return None
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    out = {
        "from": r.index[0].date().isoformat(), "to": r.index[-1].date().isoformat(),
        "total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0.5 else None,
        "vol": float(r.std() * math.sqrt(252)), "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else None,
        "maxDrawdown": float(dd),
    }
    if bench is not None:
        j = pd.concat([r, bench], axis=1, join="inner").dropna()
        if len(j) > 20:
            out["beta"] = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())
    return out


def curve(r, step=5):
    eq = (1 + r.dropna()).cumprod()
    eq = pd.concat([eq.iloc[::step], eq.iloc[-1:]]) if len(eq) > step else eq
    return [[d.date().isoformat(), round(float(v), 4)] for d, v in eq.items()]


# ---------- stress tests ----------

def factor_betas(px, assets, years=3):
    """Weekly ridge regression of each asset on the factors over the last `years`."""
    end = px.index[-1]
    w = px[px.index >= end - pd.Timedelta(days=365 * years)].resample("W-FRI").last().pct_change().dropna(how="all")
    F = w[list(FACTORS)].dropna()
    out = {}
    for a in assets:
        if a not in w:
            continue
        j = pd.concat([w[a], F], axis=1, join="inner").dropna()
        if len(j) < 52:
            continue
        y = j.iloc[:, 0].values
        X = j.iloc[:, 1:].values
        mu, sd = X.mean(0), X.std(0)
        Z = (X - mu) / sd
        lam = 0.05 * len(y)
        b = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        beta = b / sd
        resid = y - y.mean() - Z @ b
        out[a] = {"beta": dict(zip(FACTORS, map(float, beta))), "r2": float(1 - resid.var() / y.var()), "weeks": len(y)}
    return out


def window_return(px, a, b, ticker):
    s = px[ticker].loc[a:b].dropna()
    if len(s) < 2 or s.index[0] > pd.Timestamp(a) + pd.Timedelta(days=5):
        return None
    return float(s.iloc[-1] / s.iloc[0] - 1)


def stress(px, weights, betas):
    results = []
    for name, a, b, note in WINDOWS:
        fac = {f: window_return(px, a, b, f) for f in FACTORS}
        replay, covered, model = 0.0, 0.0, 0.0
        rows = []
        for k, w in weights.items():
            actual = window_return(px, a, b, k)
            est = None
            if k in betas:  # factors that did not exist yet (bitcoin before 2010) count as unchanged
                est = sum(betas[k]["beta"][f] * (fac[f] or 0.0) for f in FACTORS)
            used = actual if actual is not None else est
            if used is not None:
                model += w * used
            if actual is not None:
                replay += w * actual
                covered += w
            rows.append({"ticker": k, "weight": w, "actual": actual, "estimate": est})
        results.append({
            "name": name, "from": a, "to": b, "note": note, "kind": "historical",
            "factors": fac,
            "replay": replay / covered if covered else None, "coverage": covered,
            "estimate": model, "holdings": rows,
        })
    for name, shock, note in HYPOTHETICAL:
        est = sum(w * sum(betas[k]["beta"][f] * shock[f] for f in FACTORS) for k, w in weights.items() if k in betas)
        cov = sum(w for k, w in weights.items() if k in betas)
        results.append({"name": name, "note": note, "kind": "hypothetical", "factors": shock,
                        "estimate": est / cov if cov else None, "coverage": cov,
                        "holdings": [{"ticker": k, "weight": w, "estimate": sum(betas[k]["beta"][f] * shock[f] for f in FACTORS) if k in betas else None} for k, w in weights.items()]})
    return results


def monte_carlo(px, weights, years=3, horizon=252, paths=4000, block=20, seed=7, zero_drift=False):
    """
    Block bootstrap of the current holdings' daily returns: keeps fat tails and volatility
    clustering. With zero_drift the sample's average return is removed, so the result shows risk
    without assuming the last three years' gains repeat.
    """
    cols = [k for k in weights if k in px]
    r = px[cols].pct_change()
    r = r[r.index >= r.index[-1] - pd.Timedelta(days=365 * years)].fillna(0)
    w = np.array([weights[k] for k in cols])
    w = w / w.sum()
    port = r.values @ w
    if zero_drift:
        port = port - port.mean()
    rng = np.random.default_rng(seed)
    n = len(port)
    total, mdd = [], []
    for _ in range(paths):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, horizon // block + 1)])[:horizon]
        eq = np.cumprod(1 + port[idx])
        total.append(eq[-1] - 1)
        mdd.append((eq / np.maximum.accumulate(eq) - 1).min())
    total, mdd = np.array(total), np.array(mdd)
    q = lambda a, p: float(np.percentile(a, p))
    var95 = q(total, 5)
    return {
        "paths": paths, "horizonDays": horizon, "sampleYears": years,
        "p5": var95, "p25": q(total, 25), "p50": q(total, 50), "p75": q(total, 75), "p95": q(total, 95),
        "expectedShortfall5": float(total[total <= var95].mean()),
        "probLoss": float((total < 0).mean()), "probLoss20": float((total < -0.2).mean()),
        "medianMaxDrawdown": q(mdd, 50), "badMaxDrawdown": q(mdd, 5),
    }


# ---------- AI and imported portfolios ----------

def load_snapshots():
    """data/holdings/<portfolio>/<YYYY-MM-DD>.json, each {name, source, holdings: [{ticker, weight}]}."""
    out = {}
    base = DATA / "holdings"
    if not base.exists():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        snaps = []
        for f in sorted(d.glob("*.json")):
            j = json.loads(f.read_text())
            j["asOf"] = f.stem
            tot = sum(h["weight"] for h in j["holdings"])
            j["holdings"] = [{**h, "weight": h["weight"] / tot} for h in j["holdings"]]
            snaps.append(j)
        if snaps:
            out[d.name] = snaps
    return out


def snapshot_returns(px, snaps):
    """Forward returns of a portfolio that holds each snapshot until the next one (daily rebalanced weights)."""
    parts = []
    for i, s in enumerate(snaps):
        a = pd.Timestamp(s["asOf"])
        b = pd.Timestamp(snaps[i + 1]["asOf"]) if i + 1 < len(snaps) else px.index[-1] + pd.Timedelta(days=1)
        cols = [h["ticker"] for h in s["holdings"] if h["ticker"] in px]
        if not cols:
            continue
        w = pd.Series({h["ticker"]: h["weight"] for h in s["holdings"] if h["ticker"] in px})
        r = px[cols].pct_change()
        r = r[(r.index > a) & (r.index <= b)]
        parts.append((r.fillna(0) * (w / w.sum())).sum(axis=1))
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def main():
    trades = [t for t in json.loads((DATA / "pelosi_trades.json").read_text()) if "error" not in t]
    for t in trades:
        if t.get("ticker"):
            t["ticker"] = RENAME.get(t["ticker"], t["ticker"])
    usable = [t for t in trades if t.get("ticker") and t["ticker"] not in SKIP and t["kind"] in ("ST", "OP", None) and t["type"] in ("P", "S")]
    snaps = load_snapshots()
    snap_tickers = {h["ticker"] for v in snaps.values() for s in v for h in s["holdings"]}
    px = prices({t["ticker"] for t in usable} | set(BENCH) | set(FACTORS) | snap_tickers)
    have = set(px.columns[px.notna().any()])
    usable = [t for t in usable if t["ticker"] in have]

    r_trade, w_trade = simulate(usable, px, "traded")
    r_copy, w_copy = simulate(usable, px, "filed")
    spy = px["SPY"].pct_change()

    lags = [(date.fromisoformat(t["filed"]) - date.fromisoformat(t["traded"])).days for t in usable]
    series = {
        "pelosi_trade_date": {"name": "Pelosi, filled on the trade date", "r": r_trade},
        "pelosi_copy": {"name": "Pelosi copier, filled after the filing", "r": r_copy},
    }
    for k, n in BENCH.items():
        series[k] = {"name": n, "r": px[k].pct_change()}
    portfolios = {}
    for key, s in snaps.items():
        r = snapshot_returns(px, s)
        portfolios[key] = {"name": s[-1].get("name", key), "source": s[-1].get("source", ""), "asOf": s[-1]["asOf"], "first": s[0]["asOf"],
                           "holdings": s[-1]["holdings"], "note": s[-1].get("note", ""), "snapshots": len(s)}
        if len(r):
            series[key] = {"name": portfolios[key]["name"], "r": r}

    # common window for a fair comparison of the two Pelosi versions and the benchmarks
    common_start = max(r_trade.index[0], r_copy.index[0])
    perf = {}
    for k, v in series.items():
        r = v["r"].dropna()
        perf[k] = {"name": v["name"], "all": stats(r, spy), "curve": curve(r[r.index >= common_start]) if k.startswith("pelosi") or k in BENCH else curve(r, 1),
                   "since": stats(r[r.index >= common_start], spy)}

    books = {"pelosi_copy": {"name": "Pelosi copier (current positions)", "weights": w_copy}}
    for k, p in portfolios.items():
        books[k] = {"name": p["name"], "weights": {h["ticker"]: h["weight"] for h in p["holdings"] if h["ticker"] in have}}
    books["SPY"] = {"name": "S&P 500 (SPY)", "weights": {"SPY": 1.0}}
    books["QQQ"] = {"name": "Nasdaq 100 (QQQ)", "weights": {"QQQ": 1.0}}

    all_assets = set().union(*[set(b["weights"]) for b in books.values()])
    betas = factor_betas(px, all_assets)
    for k, b in books.items():
        b["stress"] = stress(px, b["weights"], betas)
        b["monteCarlo"] = monte_carlo(px, b["weights"])
        b["monteCarloZeroDrift"] = monte_carlo(px, b["weights"], zero_drift=True)

    recent = sorted(usable, key=lambda t: t["filed"], reverse=True)[:40]
    site = {
        "builtAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "pricesTo": px.index[-1].date().isoformat(),
        "pelosi": {
            "filings": len({t["doc"] for t in trades}), "transactions": len(trades), "used": len(usable),
            "lagEffect": lag_effect(usable, px),
            "lagDays": {"median": float(np.median(lags)), "mean": float(np.mean(lags)), "max": int(max(lags)), "p90": float(np.percentile(lags, 90))},
            "recent": [{k: t[k] for k in ("filed", "traded", "type", "partial", "ticker", "kind", "amount_lo", "amount_hi", "detail")} for t in recent],
            "commonStart": common_start.date().isoformat(),
        },
        "performance": perf,
        "portfolios": portfolios,
        "books": books,
        "factors": FACTORS,
        "betas": betas,
        "hypothetical": [{"name": n, "shock": s, "note": d} for n, s, d in HYPOTHETICAL],
    }
    (DATA / "site.json").write_text(json.dumps(site, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
    print("prices to", site["pricesTo"], "| filings", site["pelosi"]["filings"], "| lag", site["pelosi"]["lagDays"])
    for k, v in perf.items():
        s = v["since"]
        if s:
            print(f"{v['name'][:42]:42s} total {s['total']:+.1%}  cagr {s['cagr'] or 0:+.1%}  maxDD {s['maxDrawdown']:.1%}  sharpe {s['sharpe'] or 0:.2f}")
    print("copy weights", {k: round(v, 3) for k, v in list(w_copy.items())[:10]})
    for s in books["pelosi_copy"]["stress"]:
        print(f"{s['name'][:40]:40s} replay {s.get('replay')}  est {s['estimate']:+.1%}  cov {s['coverage']:.0%}")
    print(books["pelosi_copy"]["monteCarlo"])


if __name__ == "__main__":
    main()
