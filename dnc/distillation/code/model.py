from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class EmbClassifier(nn.Module):
    def __init__(self, emb_dim=300, hid_dim=512, num_classes=3):
        super().__init__()
        self.linear_hid1 = nn.Linear(emb_dim, hid_dim)
        self.linear_hid2 = nn.Linear(hid_dim, hid_dim)
        self.linear_hid3 = nn.Linear(hid_dim, hid_dim)
        self.linear_out = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.linear_hid1(x))
        x = F.relu(self.linear_hid2(x))
        x = F.relu(self.linear_hid3(x))
        return self.linear_out(x)

    def save(self, folder):
        folder = Path(folder).absolute()
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), folder / "model.pt")
