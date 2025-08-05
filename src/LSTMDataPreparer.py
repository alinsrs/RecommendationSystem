import pandas as pd
from collections import defaultdict

class LSTMDataPreparer:
    def __init__(self, filepath, min_length=5):
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
            product_id = int(row["product_id"])
            orders[row["order_id"]].append(product_id)

        all_products = set()

        for order in orders.values():
            if len(order) >= self.min_length:
                for i in range(self.min_length, len(order)):
                    subseq = order[:i + 1]
                    self.sequences.append(subseq)
                    all_products.update(subseq)  # جمع‌آوری همه آیتم‌های منحصربه‌فرد

        if all_products:
            self.num_products = max(all_products)
        else:
            raise ValueError("No valid sequences found. Please check your data.")

        print(f"✅ Total sequences prepared: {len(self.sequences)}")

    def get_sequences(self):
        return [seq[:-1] for seq in self.sequences]

    def get_targets(self):
        return [seq[-1] for seq in self.sequences]

    def get_num_products(self):
        return self.num_products
