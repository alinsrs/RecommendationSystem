from src.DataLoader import DataLoader
from src.Recommender import Recommender
from src.ModelEvaluator import ModelEvaluator

def main():
    print("\n🛠 Running Recommendation System - Version: v1.4-weighted-normalized")

    # 1. لود داده‌ها
    loader = DataLoader("data/order-Product_prior.csv")
    loader.load_data()
    loader.prepare_sentences()
    orders = loader.get_sentences()

    # 2. آموزش مدل
    recommender = Recommender()
    recommender.train(orders)
    recommender.save_model()
    recommender.load_model()

    # 3. تست محصولات
    for test_id in [24852, 47766, 47629, 26209]:
        similar = recommender.recommend_similar(test_id, topn=5)
        print(f"\n🔍 Top 5 products frequently bought with {test_id}:")
        for pid, score in similar:
            print(f"  ➤ Product {pid} (score: {score:.4f})")

    # 4. ارزیابی Leave-k-Out برای k=1 تا 3
    evaluator = ModelEvaluator(model=recommender.model, orders=orders, topn=10)

    for k in range(1, 4):
        results = evaluator.evaluate_all_metrics(k=k)
        print(f"\n📊 Evaluation with Leave-{k}-Out:")
        print(f"🎯 Hit Rate @10: {results['hit_rate']:.2%}")
        print(f"🔁 Recall @10: {results['recall']:.2%}")
        print(f"📊 Mean Rank: {results['mean_rank']:.2f}")
        print(f"🔄 MRR: {results['mrr']:.4f}")


if __name__ == "__main__":
    main()
