from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, orders, topn=10):
        self.model = model
        self.orders = orders
        self.topn = topn

    def evaluate_all_metrics(self, k=1):
        hit_count = 0
        total_targets = 0
        total_ranks = 0
        reciprocal_ranks = 0

        for order in tqdm(self.orders):
            if len(order) <= k:
                continue

            context = order[:-k]
            targets = order[-k:]

            candidates = self.model.recommend_similar_batch(context, topn=self.topn)

            for target in targets:
                total_targets += 1
                if target in candidates:
                    hit_count += 1
                    rank = candidates.index(target) + 1
                    total_ranks += rank
                    reciprocal_ranks += 1 / rank

        hit_rate = hit_count / total_targets if total_targets else 0
        recall = hit_rate  # چون یک هدف داریم
        mean_rank = total_ranks / hit_count if hit_count else 0
        mrr = reciprocal_ranks / total_targets if total_targets else 0

        return {
            "hit_rate": hit_rate,
            "recall": recall,
            "mean_rank": mean_rank,
            "mrr": mrr
        }
