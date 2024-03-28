from typing import Dict, Tuple

from mars.models.kplanes import KPlanesModelConfig
from mars.models.nerfacto import NerfactoModelConfig
from mars.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig


def get_kplanes_model_config(
    time_resolution: int = 40,
    **kwargs: Dict,
) -> Tuple[ModelConfig, Dict]:
    """
    Get nerfacto model config
    """
    model = KPlanesModelConfig(
        eval_num_rays_per_chunk=1 << 15,
        grid_base_resolution=[
            128,
            128,
            128,
            time_resolution,
        ],  # time-resolution should be half the time-steps
        grid_feature_dim=32,
        multiscale_res=[1, 2, 4],
        near_plane=0.01,
        far_plane=1000.0,
        num_samples=97,
        proposal_net_args_list=[
            # time-resolution should be half the time-steps
            {"num_output_coords": 8, "resolution": [128, 128, 128, time_resolution]},
            {"num_output_coords": 8, "resolution": [256, 256, 256, time_resolution]},
        ],
        loss_coefficients={
            "interlevel": 1.0,
            "distortion": 0.01,
            "plane_tv": 0.1,
            "plane_tv_proposal_net": 0.0001,
            "l1_time_planes": 0.001,
            "l1_time_planes_proposal_net": 0.0001,
            "time_smoothness": 0.1,
            "time_smoothness_proposal_net": 0.001,
        },
    )
    optim = {
        "proposal_networks": {
            # "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            # "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=200000),
            "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=200000),
        },
        "fields": {
            # "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            # "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=200000),
            "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=200000),
        },
    }
    return model, optim
