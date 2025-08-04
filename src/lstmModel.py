import torch
import torch.nn as nn

class LSTMRecommender(nn.Module):
    def __init__(self, num_products, embed_dim=64, hidden_dim=128):
        super(LSTMRecommender, self).__init__()
        self.embedding = nn.Embedding(num_products, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_products)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out[:, -1, :])  # فقط خروجی آخر
        return out
