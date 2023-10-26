"""SO(3) group transformations."""

import kornia.geometry.conversions as C
from kornia.geometry.conversions import normalize_quaternion
from kornia.core import stack, tensor
import torch
from torch import Tensor
from math import pi as PI


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    r"""Convert a quaternion to a rotation matrix.

    The quaternion should be in (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.

    Return:
        the rotation matrix of shape :math:`(*, 3, 3)`.

    Example:
        >>> quaternion = tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}")

    # normalize the input quaternion
    quaternion_norm: Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w = quaternion_norm[..., 0]
    x = quaternion_norm[..., 1]
    y = quaternion_norm[..., 2]
    z = quaternion_norm[..., 3]

    # compute the actual conversion
    tx: Tensor = 2.0 * x
    ty: Tensor = 2.0 * y
    tz: Tensor = 2.0 * z
    twx: Tensor = tx * w
    twy: Tensor = ty * w
    twz: Tensor = tz * w
    txx: Tensor = tx * x
    txy: Tensor = ty * x
    txz: Tensor = tz * x
    tyy: Tensor = ty * y
    tyz: Tensor = tz * y
    tzz: Tensor = tz * z
    one: Tensor = tensor(1.0)

    matrix_flat: Tensor = stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    )

    # this slightly awkward construction of the output shape is to satisfy torchscript
    # output_shape = [*list(quaternion.shape[:-1]), 3, 3]
    output_shape = [quaternion.shape[0], quaternion.shape[1], quaternion.shape[2], quaternion.shape[3], 3, 3]
    matrix = matrix_flat.reshape(output_shape)

    return matrix


@torch.jit.script
def quat_to_mat(quat_wxyz: Tensor) -> Tensor:
    """Convert scalar first quaternion to rotation matrix.

    Args:
        quat_wxyz: (...,4) Scalar first quaternions.

    Returns:
        (...,3,3) 3D rotation matrices.
    """
    return quaternion_to_rotation_matrix(
        quat_wxyz
    )


# @torch.jit.script
def mat_to_quat(mat: Tensor) -> Tensor:
    """Convert rotation matrix to scalar first quaternion.

    Args:
        mat: (...,3,3) 3D rotation matrices.

    Returns:
        (...,4) Scalar first quaternions.
    """
    return C.rotation_matrix_to_quaternion(
        mat, order=C.QuaternionCoeffOrder.WXYZ
    )


@torch.jit.script
def quat_to_xyz(
    quat_wxyz: Tensor, singularity_value: float = PI / 2
) -> Tensor:
    """Convert scalar first quaternion to Tait-Bryan angles.

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code_2

    Args:
        quat_wxyz: (...,4) Scalar first quaternions.
        singularity_value: Value that's set at the singularities.

    Returns:
        (...,3) The Tait-Bryan angles --- roll, pitch, and yaw.
    """
    qw = quat_wxyz[..., 0]
    qx = quat_wxyz[..., 1]
    qy = quat_wxyz[..., 2]
    qz = quat_wxyz[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    pitch = 2 * (qw * qy - qz * qx)
    is_out_of_range = torch.abs(pitch) >= 1
    pitch[is_out_of_range] = torch.copysign(
        torch.as_tensor(singularity_value), pitch[is_out_of_range]
    )
    pitch[~is_out_of_range] = torch.asin(pitch[~is_out_of_range])

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    xyz = torch.stack([roll, pitch, yaw], dim=-1)
    return xyz


@torch.jit.script
def quat_to_yaw(quat_wxyz: Tensor) -> Tensor:
    """Convert scalar first quaternion to yaw (rotation about vertical axis).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code_2

    Args:
        quat_wxyz: (...,4) Scalar first quaternions.

    Returns:
        (...,) The rotation about the z-axis in radians.
    """
    xyz = quat_to_xyz(quat_wxyz)
    yaw_rad: Tensor = xyz[..., -1]
    return yaw_rad


@torch.jit.script
def xyz_to_quat(xyz_rad: Tensor) -> Tensor:
    """Convert euler angles (xyz - pitch, roll, yaw) to scalar first quaternions.

    Args:
        xyz_rad: (...,3) Tensor of roll, pitch, and yaw in radians.

    Returns:
        (...,4) Scalar first quaternions (wxyz).
    """
    x_rad = xyz_rad[..., 0]
    y_rad = xyz_rad[..., 1]
    z_rad = xyz_rad[..., 2]

    cy = torch.cos(z_rad * 0.5)
    sy = torch.sin(z_rad * 0.5)
    cp = torch.cos(y_rad * 0.5)
    sp = torch.sin(y_rad * 0.5)
    cr = torch.cos(x_rad * 0.5)
    sr = torch.sin(x_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat_wxyz = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat_wxyz


@torch.jit.script
def yaw_to_quat(yaw_rad: Tensor) -> Tensor:
    """Convert yaw (rotation about the vertical axis) to scalar first quaternions.

    Args:
        yaw_rad: (...,1) Rotations about the z-axis.

    Returns:
        (...,4) scalar first quaternions (wxyz).
    """
    xyz_rad = torch.zeros_like(yaw_rad)[..., None].repeat_interleave(3, dim=-1)
    xyz_rad[..., -1] = yaw_rad
    quat_wxyz: Tensor = xyz_to_quat(xyz_rad)
    return quat_wxyz
