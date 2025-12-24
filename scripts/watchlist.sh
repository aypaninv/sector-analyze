
SYMBOL_FILE="files/watchlist.csv"
OUTPUT_DIR="output/watchlist"

# Pull 4hours data
python src/main/yfinance/yfinance_fetcher.py \
  --input $SYMBOL_FILE \
  --output $OUTPUT_DIR/yfinance_4hours.csv \
  --interval 4h \
  --period 30d

# Pull daily data
python src/main/yfinance/yfinance_fetcher.py \
  --input $SYMBOL_FILE \
  --output $OUTPUT_DIR/yfinance_daily.csv \
  --interval 1d \
  --period 3mo

# Pull weekly data
python src/main/yfinance/yfinance_fetcher.py \
  --input $SYMBOL_FILE \
  --output $OUTPUT_DIR/yfinance_weekly.csv \
  --interval 1wk \
  --period 2y

# Pull monthly data
python src/main/yfinance/yfinance_fetcher.py \
  --input $SYMBOL_FILE \
  --output $OUTPUT_DIR/yfinance_monthly.csv \
  --interval 1mo \
  --period 3y

# run Technical Calculator
python src/main/technical/technical_calculator.py \
  --fourhours-input $OUTPUT_DIR/yfinance_4hours.csv \
  --fourhours-output $OUTPUT_DIR/tech_4hours.csv \
  --daily-input $OUTPUT_DIR/yfinance_daily.csv \
  --daily-output $OUTPUT_DIR/tech_daily.csv \
  --weekly-input $OUTPUT_DIR/yfinance_weekly.csv \
  --weekly-output $OUTPUT_DIR/tech_weekly.csv \
  --monthly-input $OUTPUT_DIR/yfinance_monthly.csv \
  --monthly-output $OUTPUT_DIR/tech_monthly.csv

# generate technical data
python src/main/technical/technical_data.py \
  --fourhours-tech $OUTPUT_DIR/tech_4hours.csv \
  --daily-tech $OUTPUT_DIR/tech_daily.csv \
  --weekly-tech $OUTPUT_DIR/tech_weekly.csv \
  --monthly-tech $OUTPUT_DIR/tech_monthly.csv \
  --monthly-price $OUTPUT_DIR/yfinance_monthly.csv \
  --output $OUTPUT_DIR/technical_data.csv

# run portfolio report
python src/main/report/portfolio_report.py \
  --stock-file $SYMBOL_FILE \
  --technical-file $OUTPUT_DIR/technical_data.csv \
  --output $OUTPUT_DIR/portfolio_report.csv

