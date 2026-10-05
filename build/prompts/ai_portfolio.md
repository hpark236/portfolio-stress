Build a long-only US stock portfolio to hold for one month starting {date}. The goal is the best risk-adjusted return relative to the S&P 500 over that month.

Rules:
- 10 to 20 holdings. US-listed common stocks or ETFs only, using their current Yahoo Finance tickers.
- Weights are percentages that add up to 100. No position above 15%.
- No leverage, shorts, options or cash.
- Use only what you know. Do not invent prices, news or data. If your knowledge has a cutoff before {date}, say so in "note".

Return only JSON, with no other text:
{"holdings": [{"ticker": "XXX", "weight": 10, "reason": "one sentence"}], "note": "two or three sentences on the thesis and your knowledge cutoff"}
