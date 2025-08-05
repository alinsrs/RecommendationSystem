import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np



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
    def __init__(self, num_products, embed_dim=64, hidden_dim=128, dropout=0.2):
        super(LSTMRecommender, self).__init__()
        self.embedding = nn.Embedding(num_products + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_products + 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        hn = self.dropout(hn.squeeze(0))
        logits = self.fc(hn)
        return logits



class LSTMTrainer:
    def __init__(self, sequences, targets, num_products, embed_dim=128, hidden_dim=256, batch_size=128, lr=0.0005, epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = OrderDataset(sequences, targets)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.model = LSTMRecommender(num_products, embed_dim, hidden_dim, dropout=0.2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.epochs = epochs

    def train(self):
        self.model.train()
        best_loss = np.inf
        patience = 1
        counter = 0

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

                # بعد از محاسبه avg loss در پایان هر epoch:
                # avg_loss = total_loss / len(self.dataloader)
                # print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
                #
                # if avg_loss < best_loss:
                #     best_loss = avg_loss
                #     self.save_model("models/lstm_best_model.pth")
                #     counter = 0
                # else:
                #     counter += 1
                #     if counter >= patience:
                #         print("⛔ Early stopping triggered.")
                #         break

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
