import torch
import torch.nn as nn

from data import VOCAB_SIZE, SEQ_LEN


class TinyTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x, y=None):
        h = self.embed(x) + self.pos
        h = self.encoder(h)
        logits = self.head(h)
        if y is None:
            return logits
        return nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))