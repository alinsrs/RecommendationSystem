from src.LSTMDataPreparer import LSTMDataPreparer
from src.LSTMTrainer import LSTMTrainer
from src.LSTMTester import LSTMTester

def main():
    print("🚀 Starting LSTM v2.1 - LSTM + Attention Training & Evaluation")


    # مرحله 1: آماده‌سازی داده‌ها
    preparer = LSTMDataPreparer(filepath="data/order-Product_prior.csv")
    preparer.load_and_prepare()
    sequences = preparer.get_sequences()
    targets = preparer.get_targets()
    num_products = preparer.get_num_products()

    # مرحله 2: آموزش مدل
    trainer = LSTMTrainer(
        sequences, targets, num_products,
        embed_dim=128, hidden_dim=256,
        batch_size=128, lr=0.0005, epochs=100
    )
    trainer.train()
    trainer.save_model("models/lstm_attention_model.pth")

    # مرحله 3: ارزیابی مدل ذخیره‌شده
    trainer.load_model("models/lstm_attention_model.pth")
    model = trainer.get_model()

    tester = LSTMTester(model, sequences, targets, num_products)
    tester.evaluate(topk=20)

if __name__ == "__main__":
    main()
