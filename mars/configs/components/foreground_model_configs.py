from typing import Dict, Literal, Tuple

from mars.models.car_nerf import CarNeRF, CarNeRFModelConfig
from mars.models.nerfacto import NerfactoModelConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig


def get_fg_nerfacto_config(
    use_semantics: bool = False,
    use_car_model: bool = False,
    object_representation: Literal["class-wise", "object-wise"] = "object-wise",
    num_final_samples=97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict
) -> Tuple[ModelConfig, Dict]:
    """
    Get nerfacto model config
    """
    # TODO: assert
    # assert not use_semantics, "unsupported option: please use semantic-nerfw"
    # assert not use_car_model and object_representation == "object-wise", "unsupported option: please use car-nerf"
    model = NerfactoModelConfig(
        num_nerf_samples_per_ray=num_final_samples,
        predict_normals=predict_normals,
        predict_reflectance=predict_reflectance,
    )
    optim = {
        "object_model": {
            "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-5, max_steps=200000
            ),
        },
    }
    return model, optim


def get_fg_nerf_config(
    use_semantics: bool = False,
    use_car_model: bool = False,
    object_representation: Literal["class-wise", "object-wise"] = "object-wise",
    num_final_samples=97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict
) -> Tuple[ModelConfig, Dict]:
    """
    Get nerf model config
    """
    # TODO: assert
    # assert not use_semantics, "unsupported option: please use semantic-nerfw"
    # assert not use_car_model and object_representation == "object-wise", "unsupported option: please use car-nerf"
    # assert not predict_normals, "unsupported option now, use nerfacto instead"
    num_coarse_samples = 32
    num_fine_samples = num_final_samples - num_coarse_samples - 1
    model = VanillaModelConfig(
        _target=NeRFModel,
        num_coarse_samples=num_coarse_samples,
        num_importance_samples=num_fine_samples,
        # predict_normals=predict_normals,
    )
    optim = {
        "object_model": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    }
    return model, optim


def get_fg_mipnerf_config(
    use_semantics: bool = False,
    use_car_model: bool = False,
    object_representation: Literal["class-wise", "object-wise"] = "object-wise",
    num_final_samples=97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict
) -> Tuple[ModelConfig, Dict]:
    """
    Get mipnerf model config
    """
    # TODO: assert
    # assert not use_semantics, "unsupported option: please use semantic-nerfw"
    # assert not predict_normals, "unsupported option now, use nerfacto instead"
    # assert not use_car_model and object_representation == "object-wise", "unsupported option: please use car-nerf"
    model = VanillaModelConfig(
        _target=MipNerfModel,
        num_coarse_samples=32,
        num_importance_samples=num_final_samples - 1,
        # predict_normals=predict_normals,
    )
    optim = {
        "object_model": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    }
    return model, optim


def get_fg_carnerf_config(
    use_semantics: bool = False,
    use_car_model: bool = True,
    object_representation: Literal["class-wise", "object-wise"] = "class-wise",
    num_final_samples=97,
    predict_normals: bool = False,
    predict_reflectance: bool = False,
    **kwargs: Dict
) -> Tuple[ModelConfig, Dict]:
    """
    Get carnerf model config
    """
    # TODO: assert
    # assert not use_semantics, "unsupported option: please use semantic-nerfw"
    # assert use_car_model and object_representation == "class-wise", "unsupported option: please use class-wised option"
    # assert not predict_normals, "unsupported option now, use nerfacto instead"
    model = CarNeRFModelConfig(
        _target=CarNeRF,
        num_fine_samples=num_final_samples,
        predict_normals=predict_normals,
        predict_reflectance=predict_reflectance,
    )
    optim = {
        "object_model": {
            "optimizer": RAdamOptimizerConfig(lr=5e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-5, max_steps=200000
            ),
        },
    }
    return model, optim
