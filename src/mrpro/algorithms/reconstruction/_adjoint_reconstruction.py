from __future__ import annotations

from mrpro.algorithms.preprocess import prewhiten_kspace
from mrpro.algorithms.reconstruction._reconstruction import Reconstruction
from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data import KNoise
from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp


class AdjointReconstruction(Reconstruction):
    """Adjoint Reconstruction."""

    def __init__(
        self, FourierOperator, csm: None | CsmData = None, noise: None | KNoise = None, dcf: DcfData | None = None
    ):
        """Initialize AdjointReconstruction."""
        super().__init__()
        self.FourierOperator = FourierOperator
        self.dcf = dcf
        self.csm = csm
        self.noise = noise

    @classmethod
    def from_kdata(cls, kdata: KData, noise: KNoise | None = None, coil_combine=True) -> AdjointReconstruction:
        """Create an AdjointReconstruction from kdata with default settings."""
        if noise is not None:
            kdata = prewhiten_kspace(kdata, noise)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        FourierOperator = FourierOp.from_kdata(kdata)
        (FHkdata,) = (FourierOperator * dcf.data).H(kdata.data)
        csm = CsmData.from_idata_walsh(IData.from_tensor_and_kheader(FHkdata, kdata.header)) if coil_combine else None
        return cls(FourierOperator, csm, noise, dcf)

    def recalculate_fourierop(self, kdata: KData):
        """Update the Fourier Operator, e.g. for a new trajectory.

        Parameters
        ----------
        kdata
            KData to determine trajectory and recon/encoding matrix from.
        """
        self.FourierOperator = FourierOp.from_kdata(kdata)
        self.dcf = DcfData.from_traj_voronoi(kdata.traj)
        return self

    def recalculate_csm_walsh(self, kdata: KData, noise: KNoise | None = None):
        """Update the CSM from KData using Walsh."""
        adjoint = AdjointReconstruction(self.FourierOperator, dcf=self.dcf, noise=noise)
        image = adjoint(kdata)
        csm = CsmData.from_idata_walsh(image)
        self.SensitivityOperator = SensitivityOp(csm)

    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction.

        Parameters
        ----------
        kdata
            k-space data to reconstruct
        """

        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)
        if self.csm is not None:
            operator = self.FourierOperator @ SensitivityOp(self.csm)
        else:
            operator = self.FourierOperator
        (image_data,) = operator.H(kdata.data)
        image = IData.from_tensor_and_kheader(image_data, kdata.header)
        return image
