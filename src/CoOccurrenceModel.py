from collections import defaultdict
import math

def default_dict_factory():
    return defaultdict(int)

class CoOccurrenceModel:
    def __init__(self):
        self.co_matrix = defaultdict(default_dict_factory)
        self.product_freq = defaultdict(int)


    def train(self, orders):
        for order in orders:
            for i, product in enumerate(order):
                self.product_freq[product] += 1
                for j, other_product in enumerate(order):
                    if i == j:
                        continue
                    weight = 2 if abs(i - j) == 1 else 1  # افزایش وزن اگر پشت‌سرهم باشند
                    self.co_matrix[product][other_product] += weight

    def recommend_similar(self, product_id, topn=5):
        if product_id not in self.co_matrix:
            return []

        related_products = self.co_matrix[product_id]
        scored = []

        for other_id, co_count in related_products.items():
            norm = math.sqrt(self.product_freq[product_id] * self.product_freq[other_id])
            normalized_score = co_count / norm if norm > 0 else 0
            scored.append((other_id, normalized_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topn]

    def recommend_similar_batch(self, context_products, topn=10):
        scores = defaultdict(float)
        context_set = set(context_products)

        for pid in context_set:
            for related_pid, co_count in self.co_matrix[pid].items():
                if related_pid in context_set:
                    continue
                norm = math.sqrt(self.product_freq[pid] * self.product_freq[related_pid])
                if norm > 0:
                    scores[related_pid] += co_count / norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked[:topn]]
