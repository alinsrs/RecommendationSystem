import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.sentences = []

    def load_data(self, limit=None):
        self.df = pd.read_csv(self.file_path)
        if limit is not None:
            self.df = self.df.head(limit)

        print(f"✅ Total rows loaded: {len(self.df):,}")

    def prepare_sentences(self):
        if self.df is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        grouped = self.df.groupby("order_id")["product_id"].apply(list)
        self.sentences = grouped.tolist()
        print(f"🧾 Prepared {len(self.sentences):,} orders (sentences) for training.")

    def get_sentences(self):
        return self.sentences
