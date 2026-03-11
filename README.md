# Yield Curve and S&P 500 Animation

This project generates a looping educational animation that pairs:

- A US Treasury yield curve snapshot
- An indexed S&P 500 path over time
- An S&P 500 P/E ratio panel
- NBER recession shading
- Yield-curve inversion markers and tracker

Outputs:

- [output/yield_curve_sp500_recessions.gif](output/yield_curve_sp500_recessions.gif)
- [output/yield_curve_sp500_recessions.mp4](output/yield_curve_sp500_recessions.mp4)

## Data sources

- FRED Treasury constant maturity series: `DGS3MO`, `DGS1`, `DGS2`, `DGS5`, `DGS10`
- FRED recession indicator: `USREC`
- Stooq daily S&P 500 price history: `^SPX`
- Multpl monthly S&P 500 P/E history

The S&P 500 line is shown as a price index, not a total-return series.

## Local run

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=$PWD/.mplconfig ./.venv/bin/python make_yield_sp500_gif.py
```

## Daily automation

The GitHub Actions workflow in `.github/workflows/daily-refresh.yml` regenerates the GIF and MP4 every day after the US cash close and commits updated outputs back to the repo.
