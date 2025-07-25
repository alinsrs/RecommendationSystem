import pickle
from src.CoOccurrenceModel import CoOccurrenceModel

class Recommender:
    def __init__(self):
        self.model = CoOccurrenceModel()

    def train(self, orders):
        self.model.train(orders)

    def save_model(self, path="models/co_occurrence.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"💾 Co-occurrence model saved to {path}")

    def load_model(self, path="models/co_occurrence.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print(f"📦 Co-occurrence model loaded from {path}")

    def recommend_similar(self, product_id, topn=5):
        return self.model.recommend_similar(product_id, topn=topn)

    def recommend_batch(self, context_products, topn=10):
        return self.model.recommend_similar_batch(context_products, topn=topn)
