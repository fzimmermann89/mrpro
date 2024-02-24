import torch

from mrpro.data import IData
from mrpro.data import KData


class Reconstruction(torch.nn.Module):
    def forward(self, kdata: KData) -> IData: ...
