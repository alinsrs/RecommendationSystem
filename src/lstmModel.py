import torch
import torch.nn as nn

class LSTMRecommender(nn.Module):
    def __init__(self, num_products, embed_dim=128, hidden_dim=256):
        super(LSTMRecommender, self).__init__()
        self.embedding = nn.Embedding(num_products, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,  # فقط وقتی num_layers > 1 باشد، dropout اعمال می‌شود
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_products)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out[:, -1, :])  # فقط خروجی آخر
        return out
