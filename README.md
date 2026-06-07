# US Top 250 Stock Performance Monitor

A lightweight daily monitoring system for the top 250 US listed stocks by market capitalisation.

## What it does

- Builds a daily universe of the largest 250 US stocks by market cap.
- Downloads one year of adjusted daily closing prices.
- Calculates:
  - 1 day return
  - 5 day return
  - 1 month return
  - 3 month return
  - 6 month return
  - 1 year return
  - YTD return
  - current drawdown from one year high
  - rank by one year performance
- Produces:
  - CSV output
  - HTML dashboard
  - alert file for large daily moves and drawdowns
  - underlying daily price history for local analytics
- Includes a local Streamlit analytics dashboard for screening stocks by recent price action.

## Data provider

This version uses Financial Modeling Prep because it has:
- stock screener capability for market cap based universe selection
- historical prices
- company profile data

You need an API key.

Set your API key:

```bash
export FMP_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
setx FMP_API_KEY "your_api_key_here"
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run daily monitor

```bash
python3 src/run_daily_monitor.py
```

Outputs will be created in:

```text
data/price_history.csv
data/next_day_predictions.csv
data/next_day_prediction_results.csv
data/top_10_intraday_picks.csv
data/top_10_intraday_pick_results.csv
data/sector_predictions.csv
data/sector_prediction_results.csv
data/historical_learning_summary.csv
data/historical_learning_backtest.csv
data/historical_learning_signals.csv
data/historical_learning_weights.csv
data/historical_learning_top25_recent_accuracy.csv
data/trading212_cfd_symbols.csv
reports/daily_stock_monitor.csv
reports/daily_stock_monitor.html
reports/alerts.csv
```

## Run daily monitor plus prediction tracking

This command refreshes the daily monitor, appends next day predictions, and
updates prediction-vs-actual results when the next trading day is available:

```bash
python3 src/run_daily_prediction_cycle.py
```

Prediction tracking creates:

```text
data/next_day_predictions.csv
data/next_day_prediction_results.csv
data/top_10_intraday_picks.csv
data/top_10_intraday_pick_results.csv
data/sector_predictions.csv
data/sector_prediction_results.csv
data/historical_learning_summary.csv
data/historical_learning_backtest.csv
data/historical_learning_signals.csv
data/historical_learning_weights.csv
data/historical_learning_top25_recent_accuracy.csv
data/trading212_cfd_symbols.csv
```

The comparison file grows over time. A comparison is added once a saved
prediction has a later trading day available in `data/price_history.csv`.
The Top 10 intraday pick review file follows the same pattern: picks are saved
daily, then reviewed when the next trading day is available.
Sector predictions are also saved daily and compared against actual 1, 5, and
20 trading day sector returns as future data becomes available.
The historical learning backtest learns 50 close-to-close signals from a 50
trading day window, tests on a 50 trading day walk-forward window, and repeats
weight updates up to 15 iterations or until it reaches 70% directional accuracy.
It also produces a top 25 stock table showing how accurate the model was for
each stock over the latest 5 backtest days.
The top 25 table is filtered through `data/trading212_cfd_symbols.csv`. Replace
that file with a verified Trading 212 CFD symbol export/list whenever you want a
stricter platform-specific universe.

## Run analytics dashboard

Run the daily monitor first so the dashboard has fresh CSV files:

```bash
python3 src/run_daily_monitor.py
```

Then start the local dashboard:

```bash
streamlit run app.py
```

The dashboard reads the latest generated files from `data/price_history.csv` and
`reports/daily_stock_monitor.csv`. Use the sidebar to choose an analysis type,
lookback period, dollar movement range, minimum stock price, and sector filter.
The `Ask a stock question` dashboard answers plain-English stock questions from
the saved analytics and can fetch recent online headlines for news-style queries
when the app host has internet access.

## Deploy online

The app is ready to deploy on Streamlit Community Cloud.

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app.
3. Select the GitHub repository and branch.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

The included GitHub Actions workflow runs the full refresh command on weekdays:

```bash
python3 src/run_daily_prediction_cycle.py
```

The workflow commits refreshed `data/` and `reports/` files back to GitHub.
That lets the hosted Streamlit app read fresh CSV files without your local
computer being awake.

## Schedule options

### Option 1: GitHub Actions
Use the included workflow in `.github/workflows/daily_monitor.yml`.
It refreshes the dashboard data and commits updated CSV files back to the repo.

### Option 2: Local Windows Task Scheduler
Run once daily after US market close, for example 10:30 PM UK time.

### Option 3: Other cloud
Run as an AWS Lambda, Azure Function, or small container job.
