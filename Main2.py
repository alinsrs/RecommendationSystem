from src.LSTMDataPreparer import LSTMDataPreparer
from src.LSTMTrainer import LSTMTrainer
from src.LSTMTester import LSTMTester

def main():
    # مرحله 1: آماده‌سازی داده‌ها
    preparer = LSTMDataPreparer(filepath="data/order-Product_prior.csv")
    preparer.load_and_prepare()
    sequences = preparer.get_sequences()
    targets = preparer.get_targets()
    num_products = preparer.get_num_products()

    # مرحله 2: آموزش مدل
    trainer = LSTMTrainer(
        sequences, targets, num_products,
        embed_dim=64, hidden_dim=128,
        batch_size=128, lr=0.001, epochs=10
    )
    trainer.train()
    trainer.save_model("models/lstm_model.pth")

    # مرحله 3: ارزیابی مدل ذخیره‌شده
    trainer.load_model("models/lstm_model.pth")
    model = trainer.get_model()

    tester = LSTMTester(model, sequences, targets, num_products)
    tester.evaluate(topk=20)

if __name__ == "__main__":
    main()
