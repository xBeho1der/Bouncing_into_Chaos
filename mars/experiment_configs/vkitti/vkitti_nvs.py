from mars.configs.base_config_template import compose_config
from mars.configs.components.background_model_configs import get_bg_nerfacto_config
from mars.configs.components.data_configs import get_vkitti_dataparser
from mars.configs.components.foreground_model_configs import get_fg_carnerf_config
from mars.configs.components.kplanes_model_configs import get_kplanes_model_config

VKITTI_nvs25_Car_Depth = compose_config(
    method_name="vkitti-nvs25",
    description="vkitti nvs-25 with car nerf, depth, nerfacto",
    sequence_id="Scene06",
    experiment_setting="nvs-25",
    object_representation="class-wise",
    use_car_model=True,
    use_depth=True,
    use_semantics=False,
    use_sky_model=False,
    dataparser_config_generator=get_vkitti_dataparser,
    foreground_config_generator=get_fg_carnerf_config,
    background_config_generator=get_bg_nerfacto_config,
    kplanes_model_config_generator=get_kplanes_model_config,
    first_frame=65,
    last_frame=120,
    predict_normals=False,
)

VKITTI_nvs50_Car_Depth = compose_config(
    method_name="vkitti-nvs50",
    description="vkitti nvs-50 with car nerf, depth, nerfacto",
    sequence_id="Scene06",
    experiment_setting="nvs-50",
    object_representation="class-wise",
    use_car_model=True,
    use_depth=True,
    use_semantics=False,
    use_sky_model=False,
    dataparser_config_generator=get_vkitti_dataparser,
    foreground_config_generator=get_fg_carnerf_config,
    background_config_generator=get_bg_nerfacto_config,
    kplanes_model_config_generator=get_kplanes_model_config,
    first_frame=65,
    last_frame=120,
    predict_normals=False,
)

VKITTI_nvs75_Car_Depth = compose_config(
    method_name="vkitti-nvs75",
    description="vkitti nvs-75 with car nerf, depth, nerfacto",
    sequence_id="Scene06",
    experiment_setting="nvs-75",
    object_representation="class-wise",
    use_car_model=True,
    use_depth=True,
    use_semantics=False,
    use_sky_model=False,
    dataparser_config_generator=get_vkitti_dataparser,
    foreground_config_generator=get_fg_carnerf_config,
    background_config_generator=get_bg_nerfacto_config,
    kplanes_model_config_generator=get_kplanes_model_config,
    first_frame=65,
    last_frame=120,
    predict_normals=False,
)
