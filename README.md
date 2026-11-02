# TSE closing auction

Code for the NumerixS Quant take-home assignment.

## Run

Place the raw `auctionYYYYMMDD.csv` and `tradeYYYYMMDD.csv` files in the same folder, then run:

```bash
python run_analysis.py --data-dir "/path/to/data"

Requires Python with:
- pandas
- numpy
- matplotlib

Files
- prepare_data.py — reads and prepares the raw data
- explore_data.py — optional walkthrough of one stock and trading day
- auction_rules.py — sizing and timing functions
- validation.py — historical validation and sensitivity checks
- run_analysis.py — runs the full analysis
- tests.py — core implementation checks

Main outputs
The analysis writes results to output/, including:
- Toyota_January_2025.csv
- validation and sensitivity tables
- figures used in the report
- prepared intermediate tables

The accompanying PDF contains the methodology, results, assumptions, and limitations.
