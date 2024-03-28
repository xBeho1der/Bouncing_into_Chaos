from __future__ import annotations

from typing import Callable, Literal

from mars.configs.components.background_model_configs import sky_config_generator
from mars.data.mars_datamanager import MarsDataManagerConfig
from mars.mars_pipeline import MarsPipelineConfig
from mars.models.scene_graph import SceneGraphModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

MAX_NUM_ITERATIONS = 600000
STEPS_PER_SAVE = 2000
STEPS_PER_EVAL_IMAGE = 500
STEPS_PER_EVAL_ALL_IMAGES = 5000


def compose_config(
    method_name: str,
    description: str,
    sequence_id: str,
    use_depth: bool,
    use_semantics: bool,
    predict_normals: bool,
    use_sky_model: bool,
    experiment_setting: Literal["reconstruction", "nvs-25", "nvs-50", "nvs-75"],
    object_representation: Literal["class-wise", "object-wise"],
    use_car_model: bool,
    dataparser_config_generator: Callable,
    background_config_generator: Callable,
    foreground_config_generator: Callable,
    kplanes_model_config_generator: Callable,
    num_final_samples: int = 97,
    mono_normal_loss_mult: float = 0.0,
    first_frame: int = 65,
    last_frame: int = 72,
    second_pass: bool = False,
    start_later: bool = False,
    start_second_pass: int = 20000,
    use_grad_scaler: bool = True,
    scale_factor: float = 1.0,
    depth_gt_type: Literal["gt", "mono"] = "gt",
    vis: Literal[
        "viewer", "wandb", "tensorboard", "viewer+wandb", "viewer+tensorboard"
    ] = "wandb",
    project_name: str = "nerfstudio-project",
) -> MethodSpecification:
    """
    Generate model configs.
    """
    assert (
        not use_car_model or object_representation == "class-wise"
    ), "[CONFLICT] car model requires class-wised object representation"
    optimizers = {}

    load_semantics = use_semantics or use_sky_model
    predict_reflectance = second_pass if not start_later else not second_pass

    bg_model, bg_optimizer = background_config_generator(
        use_depth=use_depth,
        use_semantics=use_semantics,
        num_final_samples=num_final_samples,
        predict_normals=predict_normals,
        predict_reflectance=predict_reflectance,
    )
    fg_model, fg_optimizer = foreground_config_generator(
        use_depth=use_depth,
        use_semantics=use_semantics,
        use_car_model=use_car_model,
        object_representation=object_representation,
        num_final_samples=num_final_samples,
        predict_normals=predict_normals,
        predict_reflectance=predict_reflectance,
    )
    kplanes_model, kplanes_optimizer = kplanes_model_config_generator()
    use_car_model = (
        use_car_model
        and object_representation == "class-wise"
        and foreground_config_generator.__name__ == "get_fg_carnerf_config"
    )
    dataparser = dataparser_config_generator(
        sequence_id=sequence_id,
        use_depth=use_depth,
        use_semantics=load_semantics,
        use_car_model=use_car_model,
        experiment_setting=experiment_setting,
        first_frame=first_frame,
        last_frame=last_frame,
        pred_normals=predict_normals,
        scale_factor=scale_factor,
        start_later=start_later,
    )
    sky_optimizer = sky_config_generator()
    optimizers.update(bg_optimizer)
    optimizers.update(fg_optimizer)
    optimizers.update(kplanes_optimizer)
    if use_sky_model:
        optimizers.update(sky_optimizer)
    return MethodSpecification(
        config=TrainerConfig(
            method_name=method_name,
            project_name=project_name,
            steps_per_eval_image=STEPS_PER_EVAL_IMAGE,
            steps_per_eval_all_images=STEPS_PER_EVAL_ALL_IMAGES,
            steps_per_save=STEPS_PER_SAVE,
            max_num_iterations=MAX_NUM_ITERATIONS,
            save_only_latest_checkpoint=False,
            mixed_precision=False,
            use_grad_scaler=use_grad_scaler,
            log_gradients=True,
            start_second_pass=start_second_pass,
            pipeline=MarsPipelineConfig(
                datamanager=MarsDataManagerConfig(
                    dataparser=dataparser,
                    train_num_rays_per_batch=4096,
                    eval_num_rays_per_batch=4096,
                    camera_optimizer=CameraOptimizerConfig(mode="off"),
                ),
                model=SceneGraphModelConfig(
                    background_model=bg_model,
                    use_sky_model=use_sky_model,
                    object_model_template=fg_model,
                    kplanes_model=kplanes_model,
                    object_representation=object_representation,
                    object_ray_sample_strategy="remove-bg",
                    predict_normals=predict_normals,
                    mono_normal_loss_mult=mono_normal_loss_mult,
                    mono_depth_loss_mult=0.01 if depth_gt_type == "mono" else 0.0,
                    depth_loss_mult=0.01 if depth_gt_type == "gt" else 0.0,
                    use_semantics=use_semantics,
                    second_pass=second_pass if not start_later else not second_pass,
                ),
            ),
            optimizers=optimizers,
            vis=vis,
        ),
        description=description,
    )
