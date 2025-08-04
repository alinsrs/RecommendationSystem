import torch
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

class LSTMTester:
    def __init__(self, model, sequences, targets, num_products, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = OrderDataset(sequences, targets)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        self.num_products = num_products

    def evaluate(self, topk=10):
        hit = 0
        total = 0
        mrr_total = 0

        for inputs, targets in tqdm(self.dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
                topk_preds = torch.topk(logits, topk, dim=1).indices

                for i in range(targets.size(0)):
                    target = targets[i].item()
                    predictions = topk_preds[i].tolist()
                    if target in predictions:
                        hit += 1
                        rank = predictions.index(target) + 1
                        mrr_total += 1.0 / rank
                    total += 1

        hit_rate = hit / total * 100
        mrr = mrr_total / total

        print(f"📊 Evaluation Results:")
        print(f"🎯 Hit Rate @{topk}: {hit_rate:.2f}%")
        print(f"🔄 MRR: {mrr:.4f}")
