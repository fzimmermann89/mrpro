from __future__ import annotations

from mrpro.algorithms.preprocess import prewhiten_kspace
from mrpro.algorithms.reconstructions._reconstruction import Reconstruction
from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data import KNoise
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp


class AdjointReconstruction(Reconstruction):
    def __init__(self, FourierOperator, csm: None | CsmData = None, noise: None | KNoise = None):
        super().__init__()
        self.FourierOperator = FourierOperator

        self.SensitivityOperator = SensitivityOp(csm) if csm is not None else None
        self.noise = noise

    @classmethod
    def from_kdata(cls, kdata: KData, noise: KNoise | None = None) -> AdjointReconstruction:
        """Create an instance of AdjointReconstruction from kdata with default settings"""
        if noise is not None:
            kdata = prewhiten_kspace(kdata, noise)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        # TODO: check if this is the correct way to apply the dcf
        FourierOperator = FourierOp.from_kdata(KData) * dcf.data
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        (FHkdata,) = FourierOperator.H @ kdata.data
        csm = CsmData.from_idata_walsh(IData(FHkdata, kdata.header))
        return cls(FourierOperator, csm, dcf, noise)

    def forward(self, kdata: KData) -> IData:
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)

        adjoint = self.FourierOperator.H @ kdata
        if self.SensitivityOperator is not None:
            adjoint = self.SensitivityOp.H @ adjoint
        return IData(adjoint, kdata.header)
