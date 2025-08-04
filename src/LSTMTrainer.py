import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class OrderDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)


def collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded_sequences, targets


class LSTMRecommender(nn.Module):
    def __init__(self, num_products, embed_dim=64, hidden_dim=128):
        super(LSTMRecommender, self).__init__()
        self.embedding = nn.Embedding(num_products + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_products + 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        logits = self.fc(hn.squeeze(0))
        return logits


class LSTMTrainer:
    def __init__(self, sequences, targets, num_products, embed_dim=64, hidden_dim=128, batch_size=128, lr=0.001, epochs=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = OrderDataset(sequences, targets)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.model = LSTMRecommender(num_products, embed_dim, hidden_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for inputs, targets in loop:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / len(self.dataloader):.4f}")

    def save_model(self, path="models/lstm_model.pth"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"📦 Model saved to {path}")

    def load_model(self, path="models/lstm_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded from {path}")

    def get_model(self):
        return self.model
