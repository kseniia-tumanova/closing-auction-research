# Tokyo Stock Exchange Closing Auction Research

Python implementation of systematic execution strategies for the Tokyo Stock Exchange closing auction, including order sizing, execution timing, liquidity constraints, and historical validation.

## Project Overview

This project investigates execution strategies for the Tokyo Stock Exchange closing auction using historical order and trade data. The analysis includes data preparation, exploratory analysis, algorithm implementation, and evaluation of execution rules under different market conditions.

## Dataset

The analysis used approximately 78 million auction and trade records across 19 trading sessions, organized into 38 CSV files.

The original market data is not included in this repository.

## Methodology

- Process and validate historical auction and trade data
- Analyze trading activity and closing auction liquidity
- Implement order-sizing and execution-timing rules
- Evaluate alternative execution strategies using historical simulations
- Perform walk-forward validation and sensitivity analysis

## Code Structure

- `prepare_data.py` — Raw data preparation and processing
- `explore_data.py` — Exploratory analysis of individual stocks and trading sessions
- `auction_rules.py` — Order-sizing and execution-timing algorithms
- `validation.py` — Historical validation and sensitivity analysis
- `run_analysis.py` — Main analysis workflow
- `tests.py` — Implementation checks

## Requirements

Python 3 with the following libraries:

- pandas
- NumPy
- Matplotlib

## Running the Analysis

Place the required `auctionYYYYMMDD.csv` and `tradeYYYYMMDD.csv` files in a local data directory, then run:

```bash
python run_analysis.py --data-dir "/path/to/data"
```

## Outputs

The analysis generates results in the `output/` directory, including:

- Stock-level execution analysis
- Validation and sensitivity tables
- Figures summarizing execution strategy performance
- Prepared intermediate datasets

The full analysis requires access to the original market data, which is not distributed with this repository.
