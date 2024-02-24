import torch
from mrpro.data import IData, KData

class Reconstruction(torch.nn.Module):
    def forward(self, kdata:KData) -> IData:
       ...
