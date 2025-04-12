import torch
from torch import nn
from data import Batch

FT_OUT = 32

class TerasNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(768, FT_OUT)
        self.hidden1 = nn.Linear(FT_OUT * 2, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, batch):
        stm_out = self.ft(batch.stm_features)
        non_stm_out = self.ft(batch.non_stm_features)
        hidden = torch.clamp(torch.cat((stm_out, non_stm_out), dim=1), 0.0, 1.0)
        hidden = torch.clamp(self.hidden1(hidden), 0.0, 1.0)
        hidden = torch.clamp(self.hidden2(hidden), 0.0, 1.0)
        return torch.sigmoid(self.out(hidden))

