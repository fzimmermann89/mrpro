""""A pytorch implementation of scipy.spatial.transform.Rotation."""

# based on Scipy implementation, which has the following copyright:
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch

from mrpro.data import SpatialDimension

AXIS_ORDER = ('x', 'y', 'z')
QUAT_AXIS_ORDER = (*AXIS_ORDER, 'w')


def _compose_quaternions_single(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Calculate p * q."""
    cross = torch.cross(p[:3], q[:3])
    product = torch.stack(
        (
            p[3] * q[0] + q[3] * p[0] + cross[0],
            p[3] * q[1] + q[3] * p[1] + cross[1],
            p[3] * q[2] + q[3] * p[2] + cross[2],
            p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
        ),
        0,
    )
    return product


def _compose_quaternions(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p, q = torch.broadcast_tensors(p, q)
    product = torch.vmap(_compose_quaternions_single)(p.reshape(-1, 4), q.reshape(-1, 4)).reshape(p.shape)
    return product


def _canonical_quaternion(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    needs_inversion = (w < 0) | ((w == 0) & ((x < 0) | ((x == 0) & ((y < 0) | ((y == 0) & (z < 0))))))
    q = torch.where(needs_inversion.unsqueeze(-1), -q, q)
    return q


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'Invalid rotation matrix shape {matrix.shape}.')
    batch_shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.flatten(start_dim=-2), -1)

    xyzw = torch.nn.functional.relu(
        torch.stack(
            [
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
                1.0 + m00 + m11 + m22,
            ],
            dim=-1,
        )
    )
    x, y, z, w = xyzw.unbind(-1)

    candidates = torch.stack(
        (
            *(x, m10 + m01, m02 + m20, m21 - m12),
            *(m10 + m01, y, m12 + m21, m02 - m20),
            *(m20 + m02, m21 + m12, z, m10 - m01),
            *(m21 - m12, m02 - m20, m10 - m01, w),
        ),
        dim=-1,
    ).reshape(*batch_shape, 4, 4)

    # the choice will not influence the gradients.
    choice = xyzw.argmax(dim=-1)
    # quaternion = (candidates[...,choice]/(2*xyzw[...,choice].sqrt())).squeeze(-1)
    quaternion = candidates.take_along_dim(choice[..., None, None], -2).squeeze(-2) / (
        xyzw.take_along_dim(choice[..., None], -1).sqrt() * 2
    )
    return quaternion


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(-1)

    xx = x.square()
    yy = y.square()
    zz = z.square()
    ww = w.square()
    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.stack(
        (
            *(xx - yy - zz + ww, 2 * (xy - zw), 2 * (xz + yw)),
            *(2 * (xy + zw), -xx + yy - zz + ww, 2 * (yz - xw)),
            *(2 * (xz - yw), 2 * (yz + xw), -xx - yy + zz + ww),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return matrix


class Rotation(torch.nn.Module):
    """A pytorch implementation of scipy.spatial.transform.Rotation.

    Differences compared to scipy.spatial.transform.Rotation:
    - torch.nn.Module based
    - .apply if replaced by call.
    - not all features are implemented
    - arbitrary number of batching dimensions
    """

    def __init__(self, quaternions: torch.Tensor, normalize: bool = True, copy: bool = True):
        super().__init__()

        if torch.is_complex(quaternions):
            raise ValueError('quaternions should be real numbers')
        if not torch.is_floating_point(quaternions):
            # integer or boolean dtypes
            quaternions = quaternions.float()
        if quaternions.shape[-1] != 4:
            raise ValueError('Expected `quaternions` to have shape (..., 4), ' f'got {quaternions.shape}.')

        # If a single quaternion is given, convert it to a 2D 1 x 4 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods
        if quaternions.shape == (4,):
            quaternions = quaternions[None, :]
            self._single = True
        else:
            self._single = False

        if normalize:
            norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
            if torch.any(torch.isclose(norms.float(), torch.tensor(0.0))):
                raise ValueError('Found zero norm quaternion in `quaternions`.')
            quaternions = quaternions / norms
        elif copy:
            quaternions = quaternions.clone()
        self.register_buffer('_quaternions', quaternions)

    @property
    def single(self):
        """Returns true if this a single rotation."""
        return self._single

    @classmethod
    def from_quat(cls, quaternions: torch.Tensor) -> Rotation:
        """Initialize from quaternions.

        3D rotations can be represented using unit-norm quaternions [1]_.

        Parameters
        ----------
        quaternions
            shape (..., 4)
            Each row is a (possibly non-unit norm) quaternion representing an
            active rotation, in scalar-last (x, y, z, w) format. Each
            quaternion will be normalized to unit norm.

        Returns
        -------
        rotation
            Object containing the rotations represented by input quaternions.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        return cls(quaternions, normalize=True)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> Rotation:
        """Initialize from rotation matrix.

        Rotations in 3 dimensions can be represented with 3 x 3 proper
        orthogonal matrices [1]_. If the input is not proper orthogonal,
        an approximation is created using the method described in [2]_.

        Parameters
        ----------
        matrix
            A single matrix or a stack of matrices, shape (..., 3, 3)

        Returns
        -------
        rotation
            Object containing the rotations represented by the rotation
            matrices.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        .. [2] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
               Journal of guidance, control, and dynamics vol. 31.2, pp.
               440-442, 2008.
        """
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f'Expected `matrix` to have shape (..., 3, 3), got {matrix.shape}')
        quaternions = _matrix_to_quaternion(matrix)

        return cls(quaternions, normalize=True, copy=False)

    @classmethod
    def from_rotvec(cls, rotvec: torch.Tensor, degrees: bool = False) -> Rotation:
        if degrees:
            rotvec = torch.deg2rad(rotvec)

        if rotvec.shape[-1] != 3:
            raise ValueError('Expected `rot_vec` to have shape (..., 3), got {rotvec.shape}')
        angles = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
        scales = torch.special.sinc(angles / (2 * torch.pi)) / 2
        quaternions = torch.cat((scales * rotvec[..., :-1], torch.cos(angles / 2)), -1)
        return cls(quaternions, normalize=False, copy=False)

    @classmethod
    def from_euler(cls, seq: str, angles: torch.Tensor, degrees: bool = False) -> Rotation:
        """Initialize from Euler angles.

        Rotations in 3-D can be represented by a sequence of 3
        rotations around a sequence of axes. In theory, any three axes spanning
        the 3-D Euclidean space are enough. In practice, the axes of rotation are
        chosen to be the basis vectors.

        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centred frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [1]_.

        Parameters
        ----------
        seq
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
            {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.
        angles
            (..., [1 or 2 or 3])
            Euler angles specified in radians (`degrees` is False) or degrees
            (`degrees` is True).
        degrees
            If True, then the given angles are assumed to be in degrees.
            Default is False.

        Returns
        -------
        rotation
            Object containing the rotation represented by the sequence of
            rotations around given axes with given angles.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        """
        raise NotImplementedError

    @classmethod
    def from_davenport(cls, axes: torch.Tensor, order: str, angles: torch.Tensor, degrees: bool = False):
        raise NotImplementedError

    @classmethod
    def from_mrp(cls, mrp: torch.Tensor) -> Rotation:
        raise NotImplementedError

    def as_quat(self, canonical: bool = False) -> torch.Tensor:
        """Represent as quaternions.

        Active rotations in 3 dimensions can be represented using unit norm
        quaternions [1]_. The mapping from quaternions to rotations is
        two-to-one, i.e. quaternions ``q`` and ``-q``, where ``-q`` simply
        reverses the sign of each component, represent the same spatial
        rotation. The returned value is in scalar-last (x, y, z, w) format.

        Parameters
        ----------
        canonical
            Whether to map the redundant double cover of rotation space to a
            unique "canonical" single cover. If True, then the quaternion is
            chosen from {q, -q} such that the w term is positive. If the w term
            is 0, then the quaternion is chosen such that the first nonzero
            term of the x, y, and z terms is positive.

        Returns
        -------
        quaternions
            shape (..., 4,), depends on shape of inputs used for initialization.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        quaternions: torch.Tensor = self._quaternions
        if canonical:
            quaternions = _canonical_quaternion(quaternions)
        if self.single:
            quaternions = quaternions[0]
        return quaternions

    def as_matrix(self) -> torch.Tensor:
        """Represent as rotation matrix.

        3D rotations can be represented using rotation matrices, which
        are 3 x 3 real orthogonal matrices with determinant equal to +1 [1]_.

        Returns
        -------
        matrix
            shape (..., 3, 3), depends on shape of inputs used for initialization.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        quaternions = self._quaternions
        matrix = _quaternion_to_matrix(quaternions)
        if self._single:
            return matrix[0]
        else:
            return matrix

    def as_rotvec(self, degrees: bool = False):
        """Represent as rotation vectors.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [1]_.

        Parameters
        ----------
        degrees
            Returned magnitudes are in degrees if this flag is True, else they are
            in radians. Default is False.

        Returns
        -------
        rotvec
            Shape (..., 3), depends on shape of inputs used for initialization.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        quaternions: torch.Tensor = self._quaternions
        quaternions = _canonical_quaternion(quaternions)  # w > 0 ensures that 0 <= angle <= pi

        angles = 2 * torch.atan2(torch.linalg.vector_norm(quaternions, dim=-1), quaternions[..., 3])
        scales = 2 / (torch.special.sinc(angles / (2 * torch.pi)))

        rotvec = scales[..., None] * quaternions[..., :3]

        if degrees:
            rotvec = torch.rad2deg(rotvec)

        if self._single:
            rotvec = rotvec[0]

        return rotvec

    def as_euler(self, seq: str, degrees: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def as_davenport(self, axes: torch.Tensor, order: str, degrees: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def as_mrp(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def concatenate(cls, rotations: Sequence[Rotation]) -> Rotation:
        raise NotImplementedError

    def forward(
        self,
        vectors: Sequence[float] | torch.Tensor | SpatialDimension[torch.Tensor] | SpatialDimension[float],
        inverse: bool = False,
    ) -> torch.Tensor | SpatialDimension[torch.Tensor] | tuple[float, float, float]:
        """Apply this rotation to a set of vectors.

        If the original frame rotates to the final frame by this rotation, then
        its application to a vector can be seen in two ways:

            - As a projection of vector components expressed in the final frame
              to the original frame.
            - As the physical rotation of a vector being glued to the original
              frame as it rotates. In this case the vector components are
              expressed in the original frame before and after the rotation.

        In terms of rotation matrices, this application is the same as
        ``self.as_matrix() @ vectors``.

        Parameters
        ----------
        vectors
            Shape(..., 3). Each `vectors[i]` represents a vector in 3D space.
            A single vector can either be specified with shape `(3, )` or `(1, 3)`.
            The number of rotations and number of vectors given must follow standard
            pytorch broadcasting rules.
        inverse
            If True then the inverse of the rotation(s) is applied to the input
            vectors. Default is False.

        Returns
        -------
        rotated_vectors
            Result of applying rotation on input vectors.
            Shape depends on the following cases:
                - If object contains a single rotation (as opposed to a stack
                  with a single rotation) and a single vector is specified with
                  shape ``(3,)``, then `rotated_vectors` has shape ``(3,)``.
                - In all other cases, `rotated_vectors` has shape ``(..., 3)``,
                  where ``...`` is determined by broadcasting.
        """
        if input_is_spatialdimension := isinstance(vectors, SpatialDimension):
            vectors_tensor = torch.stack([torch.as_tensor(getattr(vectors, axis)) for axis in AXIS_ORDER], -1)
        elif input_is_sequence := isinstance(vectors, Sequence):
            vectors_tensor = torch.as_tensor(vectors)
        else:
            vectors_tensor = vectors
        if vectors_tensor.shape[-1] != 3:
            raise ValueError(f'Expected input of shape (..., 3), got {vectors_tensor.shape}.')
        matrix = self.as_matrix()
        if inverse:
            matrix = matrix.mT
        try:
            result = (matrix @ vectors_tensor.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            raise ValueError(
                f'The batch-shape of the rotation, {matrix.shape[:-2]},'
                f'is not compatible with the input batch shape {vectors_tensor.shape[:-1]}'
            ) from None

        if self._single and vectors_tensor.shape == (3,):
            result = result[0]

        if input_is_spatialdimension:
            return SpatialDimension(
                x=result[..., AXIS_ORDER.index('x')],
                y=result[..., AXIS_ORDER.index('y')],
                z=result[..., AXIS_ORDER.index('z')],
            )

        elif input_is_sequence:
            return tuple(result)

        else:
            return result

    def __mul__(self, other: Rotation) -> Rotation:
        """For compatibility with sp.spatial.transform.Rotation."""
        warnings.warn(
            'Using Rotation*Rotation is deprecated, consider Rotation@Rotation', DeprecationWarning, stacklevel=2
        )
        return self @ other

    def __matmul__(self, other: Rotation) -> Rotation:
        """Compose this rotation with the other.

        If `p` and `q` are two rotations, then the composition of 'q followed
        by p' is equivalent to `p * q`. In terms of rotation matrices,
        the composition can be expressed as
        ``p.as_matrix() @ q.as_matrix()``.

        Parameters
        ----------
        other
            Object containing the rotations to be composed with this one. Note
            that rotation compositions are not commutative, so ``p * q`` is
            generally different from ``q * p``.

        Returns
        -------
        composition
            This function supports composition of multiple rotations at a time.
            The following cases are possible:

            - Either ``p`` or ``q`` contains a single rotation. In this case
              `composition` contains the result of composing each rotation in
              the other object with the single rotation.
            - Both ``p`` and ``q`` contain ``N`` rotations. In this case each
              rotation ``p[i]`` is composed with the corresponding rotation
              ``q[i]`` and `output` contains ``N`` rotations.
        """
        p = self._quaternions
        q = other._quaternions
        # TODO: broadcasting
        result = _compose_quaternions(p, q)
        if self._single and other._single:
            result = result[0]
        return self.__class__(result, normalize=True, copy=False)

    def __pow__(self, n: float, modulus: None = None):
        """Compose this rotation with itself `n` times.

        Composition of a rotation ``p`` with itself can be extended to
        non-integer ``n`` by considering the power ``n`` to be a scale factor
        applied to the angle of rotation about the rotation's fixed axis. The
        expression ``q = p ** n`` can also be expressed as
        ``q = Rotation.from_rotvec(n * p.as_rotvec())``.

        If ``n`` is negative, then the rotation is inverted before the power
        is applied. In other words, ``p ** -abs(n) == p.inv() ** abs(n)``.

        Parameters
        ----------
        n
            The number of times to compose the rotation with itself.
        modulus
            This overridden argument is not applicable to Rotations and must be
            ``None``.

        Returns
        -------
        power : `Rotation` instance
            If the input Rotation ``p`` contains ``N`` multiple rotations, then
            the output will contain ``N`` rotations where the ``i`` th rotation
            is equal to ``p[i] ** n``

        Notes
        -----
        For example, a power of 2 will double the angle of rotation, and a
        power of 0.5 will halve the angle. There are three notable cases: if
        ``n == 1`` then the original rotation is returned, if ``n == 0``
        then the identity rotation is returned, and if ``n == -1`` then
        ``p.inv()`` is returned.

        Note that fractional powers ``n`` which effectively take a root of
        rotation, do so using the shortest path smallest representation of that
        angle (the principal root). This means that powers of ``n`` and ``1/n``
        are not necessarily inverses of each other. For example, a 0.5 power of
        a +240 degree rotation will be calculated as the 0.5 power of a -120
        degree rotation, with the result being a rotation of -60 rather than
        +120 degrees.
        """
        if modulus is not None:
            raise NotImplementedError('modulus not supported')

        # Exact short-cuts
        if n == 0:
            return Rotation.identity(None if self._single else self._quaternions.shape[:-1])
        elif n == -1:
            return self.inv()
        elif n == 1:
            if self._single:
                return self.__class__(self._quaternions[0], copy=True)
            else:
                return self.__class__(self._quaternions, copy=True)
        else:  # general scaling of rotation angle
            return Rotation.from_rotvec(n * self.as_rotvec())

    def inv(self) -> Rotation:
        """Invert this rotation.

        Composition of a rotation with its inverse results in an identity
        transformation.

        Returns
        -------
        inverse
            Object containing inverse of the rotations in the current instance.
        """
        quaternions = self._quaternions * torch.tensor([-1, -1, -1, 1])
        if self._single:
            quaternions = quaternions[0]
        return self.__class__(quaternions, copy=False)

    def magnitude(self) -> torch.Tensor:
        """Get the magnitude(s) of the rotation(s).

        Returns
        -------
        magnitude
            Angles in radians. The magnitude will always be in the range [0, pi].
        """
        angles = 2 * torch.atan2(
            torch.linalg.vector_norm(self._quaternions[..., :3], dim=-1), torch.abs(self._quaternions[..., 3])
        )
        if self._single:
            angles = angles[0]
        return angles

    def approx_equal(self, other: Rotation, atol: float = 1e-8, degrees: bool = False) -> torch.Tensor | bool:
        """Determine if another rotation is approximately equal to this one.

        Equality is measured by calculating the smallest angle between the
        rotations, and checking to see if it is smaller than `atol`.

        Parameters
        ----------
        other
            Object containing the rotations to measure against this one.
        atol
            The absolute angular tolerance, below which the rotations are
            considered equal.
        degrees
            If True and `atol` is given, then `atol` is measured in degrees. If
            False (default), then atol is measured in radians.

        Returns
        -------
        approx_equal :
            Whether the rotations are approximately equal, bool if object
            contains a single rotation and Tensor if object contains multiple
            rotations.
        """
        if degrees:
            atol = np.deg2rad(atol)
        angles = (other @ self.inv()).magnitude()
        return angles < atol

    def __getitem__(self, indexer: int | slice | torch.Tensor) -> Rotation:
        """Extract rotation(s) at given index(es) from object.

        Create a new `Rotation` instance containing a subset of rotations
        stored in this object.

        Parameters
        ----------
        indexer:
            Specifies which rotation(s) to extract.

        Returns
        -------
        rotation

        Raises
        ------
        TypeError if the instance was created as a single rotation.
        """
        if self._single:
            raise TypeError('Single rotation is not subscriptable.')

        return self.__class__(self._quaternions[indexer, :], normalize=False)

    @property
    def quaternion_x(self) -> torch.Tensor:
        """Get x component of the quaternion."""
        axis = AXIS_ORDER.index('x')
        return self._quaternions[..., axis]

    @quaternion_x.setter
    def quaternion_x(self, quat_x: torch.Tensor):
        """Set x component of the quaternion."""
        axis = AXIS_ORDER.index('x')
        self._quaternions[..., axis] = quat_x

    @property
    def quaternion_y(self) -> torch.Tensor:
        """Get y component of the quaternion."""
        axis = AXIS_ORDER.index('y')
        return self._quaternions[..., axis]

    @quaternion_y.setter
    def quaternion_y(self, quat_y: torch.Tensor):
        """Set y component of the quaternion."""
        axis = AXIS_ORDER.index('y')
        self._quaternions[..., axis] = quat_y

    @property
    def quaternion_z(self) -> torch.Tensor:
        """Get z component of the quaternion."""
        axis = AXIS_ORDER.index('z')
        return self._quaternions[..., axis]

    @quaternion_z.setter
    def quaternion_z(self, quat_z: torch.Tensor):
        """Set z component of the quaternion."""
        axis = AXIS_ORDER.index('z')
        self._quaternions[..., axis] = quat_z

    @property
    def quaternion_w(self) -> torch.Tensor:
        """Get w component of the quaternion."""
        axis = AXIS_ORDER.index('w')
        return self._quaternions[..., axis]

    @quaternion_w.setter
    def quaternion_w(self, quat_w: torch.Tensor):
        """Set w component of the quaternion."""
        axis = AXIS_ORDER.index('w')
        self._quaternions[..., axis] = quat_w

    def __setitem__(self, indexer: int | slice | torch.Tensor, value: Rotation):
        """Set rotation(s) at given index(es) from object.

        Parameters
        ----------
        indexer
            Specifies which rotation(s) to replace.
        value
            The rotations to set.

        Raises
        ------
        TypeError if the instance was created as a single rotation.
        """
        if self._single:
            raise TypeError('Single rotation is not subscriptable.')

        if not isinstance(value, Rotation):
            raise TypeError('value must be a Rotation object')
        self._quaternions[indexer, :] = value.as_quat()

    @classmethod
    def identity(cls, shape: int | None | tuple[int, ...] = None) -> Rotation:
        """Get identity rotation(s).

        Composition with the identity rotation has no effect.

        Parameters
        ----------
        shape
            Number of identity rotations to generate. If None (default), then a
            single rotation is generated.

        Returns
        -------
        identity : Rotation object
            The identity rotation.
        """
        match shape:
            case None:
                q = torch.zeros(4)
            case int():
                q = torch.zeros(shape, 4)
            case tuple():
                q = torch.zeros(*shape, 4)
        q[..., -1] = 1
        return cls(q, normalize=False)

    @classmethod
    def align_vectors(
        cls, a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor | None = None, return_sensitivity: bool = False
    ) -> tuple[Rotation, float] | tuple[Rotation, float, torch.Tensor]:
        raise NotImplementedError
