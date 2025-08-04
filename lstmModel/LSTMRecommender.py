import torch.nn as nn

class LSTMRecommender(nn.Module):
    def __init__(self, num_products, embed_dim=64, hidden_dim=128):
        super(LSTMRecommender, self).__init__()
        self.embedding = nn.Embedding(num_products, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_products)

    def forward(self, x):
        # x: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        output = self.fc(last_output)  # (batch_size, num_products)
        return output
