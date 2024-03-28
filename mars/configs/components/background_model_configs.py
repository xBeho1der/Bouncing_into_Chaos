from typing import Dict, Tuple

from mars.models.nerfacto import NerfactoModelConfig
from mars.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig


def get_bg_nerfacto_config(
    use_semantics: bool = False,
    num_final_samples=97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict,
) -> Tuple[ModelConfig, Dict]:
    """
    Get nerfacto model config
    """
    if use_semantics:
        model = SemanticNerfWModelConfig(
            num_nerf_samples_per_ray=num_final_samples,
            use_single_jitter=False,
            semantic_loss_weight=0.1,
            predict_normals=predict_normals,
            predict_reflectance=predict_reflectance,
        )
    else:
        model = NerfactoModelConfig(
            num_nerf_samples_per_ray=num_final_samples,
            predict_normals=predict_normals,
            predict_reflectance=predict_reflectance,
        )
    optim = {
        "background_model": {
            "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-5, max_steps=200000
            ),
        },
    }
    return model, optim


def get_bg_nerf_config(
    use_semantics: bool = False,
    num_final_samples: int = 97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict,
) -> Tuple[ModelConfig, Dict]:
    """generate nerf model config"""

    # TODO: assert
    # assert not predict_normals, "unsupported option now, use nerfacto or semantic-nerfw instead"
    # assert not use_semantics, "unsupported option: please use semantic-nerfw"
    num_coarse_samples = 32
    model = VanillaModelConfig(
        _target=NeRFModel,
        num_coarse_samples=num_coarse_samples,
        num_importance_samples=num_final_samples - num_coarse_samples - 1,
        # predict_normals=predict_normals,
    )

    optim = {
        "background_model": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    }

    return model, optim


def get_bg_mipnerf_config(
    use_semantics: bool = False,
    num_final_samples: int = 97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict,
) -> Tuple[ModelConfig, Dict]:
    """generate nerf model config"""

    # TODO: assert
    # assert not predict_normals, "unsupported option now, use nerfacto or semantic-nerfw instead"
    # assert not use_semantics, "unsupported option: please use semantic-nerfw"
    model = VanillaModelConfig(
        _target=MipNerfModel,
        num_coarse_samples=48,
        num_importance_samples=num_final_samples - 1,
        # predict_normals=predict_normals,
    )

    optim = {
        "background_model": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    }

    return model, optim


def get_bg_semanticnerfw_config(
    use_semantics: bool = True,
    num_final_samples: int = 97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict,
) -> Tuple[ModelConfig, Dict]:
    """generate nerf model config"""

    model = SemanticNerfWModelConfig(
        num_proposal_iterations=1,
        num_proposal_samples_per_ray=[48],
        num_nerf_samples_per_ray=97,
        use_single_jitter=False,
        semantic_loss_weight=0.1,
        predict_normals=predict_normals,
        predict_reflectance=predict_reflectance,
    )
    optim = {
        "background_model": {
            "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-5, max_steps=200000
            ),
        },
    }

    return model, optim


def sky_config_generator():
    return {
        "sky_model": {
            "optimizer": RAdamOptimizerConfig(lr=3e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-5, max_steps=200000
            ),
        },
    }
