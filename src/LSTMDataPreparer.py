import pandas as pd
from collections import defaultdict

class LSTMDataPreparer:
    def __init__(self, filepath, min_length=2):
        self.filepath = filepath
        self.min_length = min_length  # حداقل تعداد محصولات در هر سفارش
        self.sequences = []
        self.targets = []
        self.num_products = 0

    def load_and_prepare(self):
        df = pd.read_csv(self.filepath)
        df = df.sort_values(by=["order_id", "add_to_cart_order"])

        orders = defaultdict(list)
        for _, row in df.iterrows():
            orders[row["order_id"]].append(int(row["product_id"]))

        all_products = set()

        for order in orders.values():
            if len(order) >= self.min_length:
                for i in range(1, len(order)):
                    seq = order[:i]
                    target = order[i]
                    self.sequences.append(seq)
                    self.targets.append(target)
                    all_products.update(seq)
                    all_products.add(target)

        self.num_products = max(all_products)
        print(f"✅ Total sequences prepared: {len(self.sequences)}")

    def get_sequences(self):
        return self.sequences

    def get_targets(self):
        return self.targets

    def get_num_products(self):
        return self.num_products
