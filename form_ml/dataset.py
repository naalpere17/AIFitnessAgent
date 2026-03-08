import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SquatSeqDataset(Dataset):
    def __init__(self, features_dir: str, labels_csv: str):
        self.features_dir = features_dir
        self.labels = pd.read_csv(labels_csv)
        self.items = []
        for _, row in self.labels.iterrows():
            vid = row["video"]
            label = int(row["label"])
            path = os.path.join(features_dir, f"features_{vid}.npz")
            if os.path.exists(path):
                self.items.append((path, label, vid))

        if len(self.items) == 0:
            raise RuntimeError("No feature files found. Did you export features_*.npz from analyze_squat?")

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        path, label, vid = self.items[idx]
        data = np.load(path)
        X = data["X"].astype(np.float32)  # (T, D)
        return torch.from_numpy(X), torch.tensor(label), vid

def collate_pad(batch):
    # batch: list of (X, y, vid) with variable T
    xs, ys, vids = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    D = xs[0].shape[1]
    T_max = int(lengths.max().item())

    x_pad = torch.zeros((len(xs), T_max, D), dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0], :] = x

    y = torch.stack(ys).long()
    return x_pad, lengths, y, vids