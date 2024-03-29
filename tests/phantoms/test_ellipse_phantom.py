"""Tests for ellipse phantom."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pytest
import torch
from mrpro.data import SpatialDimension
from mrpro.operators import FastFourierOp

from tests.helper import relative_image_difference


def test_image_space(ellipse_phantom):
    """Check if image space has correct shape."""
    img_dimension = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    img = ellipse_phantom.phantom.image_space(img_dimension)
    assert img.shape[-2:] == (ellipse_phantom.n_y, ellipse_phantom.n_x)


def test_kspace_correct_shape(ellipse_phantom):
    """Check if kspace has correct shape."""
    kdata = ellipse_phantom.phantom.kspace(ellipse_phantom.ky, ellipse_phantom.kx)
    assert kdata.shape == (ellipse_phantom.n_y, ellipse_phantom.n_x)


def test_kspace_raises_error(ellipse_phantom):
    """Check if kspace raises error if kx and ky have different shapes."""
    [kx_, _] = torch.meshgrid(
        torch.linspace(-ellipse_phantom.n_x // 2, ellipse_phantom.n_x // 2, ellipse_phantom.n_x + 1),
        torch.linspace(-ellipse_phantom.n_y // 2, ellipse_phantom.n_y // 2 + 1, ellipse_phantom.n_y),
        indexing='xy',
    )
    with pytest.raises(ValueError):
        ellipse_phantom.phantom.kspace(ellipse_phantom.ky, kx_)


def test_kspace_image_match(ellipse_phantom):
    """Check if fft of kspace matches image."""
    img_dimension = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    img = ellipse_phantom.phantom.image_space(img_dimension)
    kdata = ellipse_phantom.phantom.kspace(ellipse_phantom.ky, ellipse_phantom.kx)
    fourier_op = FastFourierOp(dim=(-1, -2))
    (reconstructed_img,) = fourier_op.adjoint(kdata)
    # Due to discretization artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img, img[0, 0, 0, :, :]) <= 0.05
