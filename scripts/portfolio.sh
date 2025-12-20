
SYMBOL_FILE="files/portfolio.csv"
OUTPUT_DIR="output/portfolio"

# Pull 4hours data
python src/main/yfinance/yfinance_fetcher.py \
  --input $SYMBOL_FILE \
  --output $OUTPUT_DIR/yf_4hours_output.csv \
  --interval 4h \
  --period 30d

# Pull daily data
python src/main/yfinance/yfinance_fetcher.py \
  --input $SYMBOL_FILE \
  --output $OUTPUT_DIR/yf_daily_output.csv \
  --interval 1d \
  --period 3mo

# Pull weekly data
python src/main/yfinance/yfinance_fetcher.py \
  --input $SYMBOL_FILE \
  --output $OUTPUT_DIR/yf_weekly_output.csv \
  --interval 1wk \
  --period 2y

# run MACD Calculator
python src/main/macd/macd_calculator.py \
  --daily-input $OUTPUT_DIR/yf_daily_output.csv \
  --daily-output $OUTPUT_DIR/macd_daily_output.csv \
  --weekly-input $OUTPUT_DIR/yf_weekly_output.csv \
  --weekly-output $OUTPUT_DIR/macd_weekly_output.csv \
  --fourhours-input $OUTPUT_DIR/yf_4hours_output.csv \
  --fourhours-output $OUTPUT_DIR/macd_4hours_output.csv

# run MACD report
python src/main/macd/macd_report.py \
  --daily-macd $OUTPUT_DIR/macd_daily_output.csv \
  --weekly-macd $OUTPUT_DIR/macd_weekly_output.csv \
  --fourhours-macd $OUTPUT_DIR/macd_4hours_output.csv \
  --output $OUTPUT_DIR/macd_report.csv

