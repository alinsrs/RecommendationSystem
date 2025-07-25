from gensim.models import Word2Vec
import os


class ModelTrainer:
    def __init__(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        self.sentences = sentences
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def train_model(self):
        """آموزش مدل Word2Vec روی سفارش‌ها"""
        self.model = Word2Vec(
            sentences=self.sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1  # skip-gram mode
        )
        print("✅ Training finished.")

    def save_model(self, path="models/item2vec.model"):
        """ذخیره مدل آموزش دیده"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"💾 Model saved to {path}")

    def get_model(self):
        return self.model
