from math import exp
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Variable

from nerfstudio.model_components.losses import (
    MSELoss,
    ScaleAndShiftInvariantLoss,
    distortion_loss,
    interlevel_loss,
)


def monosdf_depth_loss(
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    directions_norm: Float[Tensor, "*batch 1"],
    is_euclidean: bool,
):
    """MonoSDF depth loss"""
    if not is_euclidean:
        termination_depth = termination_depth * directions_norm
    sift_depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
    mask = torch.ones_like(termination_depth).reshape(1, 32, -1).bool()
    return sift_depth_loss(
        predicted_depth.reshape(1, 32, -1),
        (termination_depth * 50 + 0.5).reshape(1, 32, -1),
        mask,
    )


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes total variance across each spatial plane in the grids.

    Args:
        multi_res_grids: Grids to compute total variance over

    Returns:
        Total variance
    """

    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = {0, 1, 2}
        else:
            spatial_planes = {0, 1, 3}

        for grid_id, grid in enumerate(grids):
            if grid_id in spatial_planes:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid, only_w=True)
            num_planes += 1
    return total / num_planes


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.

    Args:
        multi_res_grids: Grids to compute L1 distance over

    Returns:
         L1 distance from the multiplicative identity (1)
    """
    time_planes = [2, 4, 5]  # These are the spatiotemporal planes
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        for grid_id in time_planes:
            total += torch.abs(1 - grids[grid_id]).mean()
            num_planes += 1

    return total / num_planes


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across the temporal axis of a plane

    Args:
        t: Plane tensor

    Returns:
        Time smoothness
    """
    _, h, _ = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = (
        first_difference[..., 1:, :] - first_difference[..., : h - 2, :]
    )  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes smoothness across each time plane in the grids.

    Args:
        multi_res_grids: Grids to compute time smoothness over

    Returns:
        Time smoothness
    """
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id])
            num_planes += 1

    return total / num_planes
