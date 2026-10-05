# portfolio-stress

Tracks portfolios people copy or hand to AI, and stress-tests them.

**Live:** https://portfolio-stress.vercel.app

## What it does

1. **Pelosi, as a copier gets it.** Nancy Pelosi's periodic transaction reports are downloaded from the House Clerk and parsed from the PDFs (`build/ptr.py`): ticker, buy or sale, trade date, filing date, dollar range, share counts and option details. Her trades are replayed twice: filled on the trade date, and filled the trading day after the report was filed, which is the earliest anyone else could act. Reports come a median of 24 days after the trade.
2. **AI portfolios, forward only.** `build/ai_portfolio.py` asks a model for a one-month portfolio with a fixed prompt (`build/prompts/ai_portfolio.md`, which does not mention the stress tests) and saves it as a dated snapshot. Returns are counted only from that date, because asking a model to pick stocks "as of" a past date lets it use what it knows happened next. Autopilot's own Claude, Grok, ChatGPT and DeepSeek portfolios are not public. Holdings copied from its app can be added to `data/holdings/<name>/<date>.json`.
3. **Stress tests** on each portfolio's current holdings:
   - **Historical replay:** 2008, the 2019 Abqaiq attack, the COVID crash and recovery, the Ukraine invasion, the 2022 bear market, the 2023 AI rally, the April 2025 tariff shock, and the June 2025 Iran war and Hormuz threat and the ceasefire after it. Holdings that did not trade yet are estimated with the factor model.
   - **Factor model:** weekly ridge regression of each holding on US stocks, semiconductors, oil, long Treasuries, gold, the dollar, defence and bitcoin.
   - **Hypothetical shocks:** a month-long Hormuz closure and its reopening, a NATO-Russia escalation, a new pandemic, a rate shock and a 12-month bull market. Each is stated as factor moves with its reasoning, and all of them can be edited on the site.
   - **Monte Carlo:** 20-day block bootstrap of the last three years, with and without that period's average return.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # needs pdftotext (poppler)
.venv/bin/python build/ptr.py            # Pelosi filings -> data/pelosi_trades.json
.venv/bin/python build/build.py          # prices, simulations, stress tests -> data/site.json
.venv/bin/python build/ai_portfolio.py   # this month's Claude portfolio (needs the Claude Code CLI)
```

A GitHub Actions job rebuilds the data every weekday after the US close.

Not investment advice.
