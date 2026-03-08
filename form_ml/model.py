import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, D)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, (h, c) = self.lstm(packed)
            # h: (num_layers*2, B, hidden)
            h_last = torch.cat([h[-2], h[-1]], dim=1)  # (B, hidden*2)
        else:
            out, (h, c) = self.lstm(x)
            h_last = torch.cat([h[-2], h[-1]], dim=1)

        h_last = self.dropout(h_last)
        return self.fc(h_last) 