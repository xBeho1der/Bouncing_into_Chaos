import contextlib
import math
from typing import Generator, Literal, Optional, Tuple, Union

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.utils import colors
from nerfstudio.utils.math import components_from_spherical_harmonics, safe_normalize

BackgroundColor = Union[
    Literal["random", "last_sample", "black", "white"],
    Float[Tensor, "3"],
    Float[Tensor, "*bs 3"],
]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


class ReflectanceRenderer(nn.Module):
    """_summary_
    Args:
        nn (_type_): _description_
    Returns:
        _type_: _description_
    """

    @classmethod
    def forward(
        cls,
        reflectance: Float[Tensor, "*batch num_samples 1"],
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs 1"]:
        n = torch.sum(weights * reflectance, dim=-2)
        return n


class OutdoorDepthRenderer(nn.Module):
    """Calculate depth along ray.

        Expected depth along ray. Same procedure as rendering rgb, but with depth.

    Args:
        method: Depth calculation method.
    """

    def __init__(self, far_plane) -> None:
        super().__init__()
        self.far_plane = far_plane

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*batch 1"]:
        """Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of depth values.
        """

        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            depth = nerfacc.accumulate_along_rays(
                weights[..., 0], values=steps, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulation = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
            depth = depth + (1 - accumulation) * self.far_plane
        else:
            depth = (
                torch.sum(weights * steps, dim=-2)
                + (1 - torch.sum(weights, -2)) * self.far_plane
            )

        depth = torch.clip(depth, steps.min(), steps.max())

        return depth


class RGBRenderer(nn.Module):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self, background_color: BackgroundColor = "random") -> None:
        super().__init__()
        self.background_color: BackgroundColor = background_color

    @classmethod
    def combine_rgb(
        cls,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image.
        If background color is random, no BG color is added - as if the background was black!

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError(
                    "Background color 'last_sample' not implemented for packed samples."
                )
            comp_rgb = nerfacc.accumulate_along_rays(
                weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "random":
            # If background color is random, the predicted color is returned without blending,
            # as if the background color was black.
            return comp_rgb
        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = rgb[..., -1, :]
        background_color = cls.get_background_color(
            background_color, shape=comp_rgb.shape, device=comp_rgb.device
        )

        assert isinstance(background_color, torch.Tensor)
        comp_rgb = comp_rgb + background_color * (1.0 - accumulated_weight)
        return comp_rgb

    @classmethod
    def get_background_color(
        cls,
        background_color: BackgroundColor,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        """Returns the RGB background color for a specified background color.
        Note:
            This function CANNOT be called for background_color being either "last_sample" or "random".

        Args:
            background_color: The background color specification. If a string is provided, it must be a valid color name.
            shape: Shape of the output tensor.
            device: Device on which to create the tensor.

        Returns:
            Background color as RGB.
        """
        assert background_color not in {"last_sample", "random"}
        assert shape[-1] == 3, "Background color must be RGB."
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color]
        assert isinstance(background_color, Tensor)

        # Ensure correct shape
        return background_color.expand(shape).to(device)

    def blend_background(
        self,
        image: Tensor,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Blends the background color into the image if image is RGBA.
        Otherwise no blending is performed (we assume opacity of 1).

        Args:
            image: RGB/RGBA per pixel.
            opacity: Alpha opacity per pixel.
            background_color: Background color.

        Returns:
            Blended RGB.
        """
        if image.size(-1) < 4:
            return image

        rgb, opacity = image[..., :3], image[..., 3:]
        if background_color is None:
            background_color = self.background_color
            if background_color in {"last_sample", "random"}:
                background_color = "black"
        background_color = self.get_background_color(
            background_color, shape=rgb.shape, device=rgb.device
        )
        assert isinstance(background_color, torch.Tensor)
        return rgb * opacity + background_color.to(rgb.device) * (1 - opacity)

    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Blends a background color into the ground truth and predicted image for
        loss computation.

        Args:
            gt_image: The ground truth image.
            pred_image: The predicted RGB values (without background blending).
            pred_accumulation: The predicted opacity/ accumulation.
        Returns:
            A tuple of the predicted and ground truth RGB values.
        """
        background_color = self.background_color
        if background_color == "last_sample":
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            background_color = torch.rand_like(pred_image)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)
        gt_image = self.blend_background(gt_image, background_color=background_color)
        return pred_image, gt_image

    def forward(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.
            background_color: The background color to use for rendering.

        Returns:
            Outputs of rgb values.
        """

        if background_color is None:
            background_color = self.background_color

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = self.combine_rgb(
            rgb,
            weights,
            background_color=background_color,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb
