import pandas as pd

class StockList:
    """Load stock list from CSV."""

    def __init__(self, filepath="data/sector-ridewinner.csv"):
        self.filepath = filepath

    def load(self):
        df = pd.read_csv(self.filepath)
        if df.empty:
            raise ValueError("Stock list CSV is empty.")
        return df
