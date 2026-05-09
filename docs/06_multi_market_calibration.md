# 06 — Multi-Market Calibration Design

## What was done

`src/calibrate.py` was refactored to support three option markets via a unified
**MarketDataAdapter** pattern. The calibration math (BSM/Heston L-BFGS-B) is
unchanged; only the data-fetching layer differs per market.

## Usage

```bash
python src/calibrate.py --market us   # SPY via yfinance
python src/calibrate.py --market cn   # 50ETF via akshare
python src/calibrate.py --market hk   # HSI via yfinance
```

Outputs: `results/calibration_<market>_bsm.json`, `_heston.json`, `_plot.pdf`

## Market differences

| Market | Source    | Ticker      | Risk-free rate | Option style | Notes                          |
|--------|-----------|-------------|----------------|--------------|--------------------------------|
| US     | yfinance  | SPY         | 5% (Fed)       | American     | Volume filter > 10             |
| CN     | akshare   | 510050.SH   | 2% (SHIBOR)    | European     | 购 = call; OI filter > 100     |
| HK     | yfinance  | ^HSI        | 4% (HIBOR)     | European     | BSM fallback if < 5 live quotes|

## Why the adapter pattern

The three markets share identical calibration math once you have
`{K, T, mid, S, r}` records. Isolating the data layer means:
- Adding a new market (e.g. CME futures options) requires only a new subclass.
- The calibration functions `calibrate_bsm()` and `calibrate_heston()` are
  market-agnostic and testable in isolation.

## HK fallback logic

HKEX HSI options have limited yfinance coverage. If fewer than 5 live quotes
are returned, the adapter synthesizes ATM quotes using BSM with σ=0.20 (typical
HSI ATM vol). This is clearly labeled in the output and is sufficient for
demonstrating the calibration pipeline in a thesis context.

## CN data notes

akshare `option_finance_board()` returns SSE 50ETF options. Key parsing:
- Filter `名称` containing `购` (call) vs `沽` (put).
- `到期日` is in `YYYYMMDD` or `YYYY-MM-DD` format.
- Prices are in CNY per unit; no multiplier adjustment needed for calibration
  since both model and market prices use the same unit.

## Risk-free rates

Rates are hardcoded as of 2024 approximations:
- US: 5% (Fed Funds effective rate)
- CN: 2% (1-year SHIBOR)
- HK: 4% (1-month HIBOR)

For production use, these should be fetched dynamically from FRED/Wind/HKMA.
