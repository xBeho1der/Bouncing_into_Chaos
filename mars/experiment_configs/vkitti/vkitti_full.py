from mars.configs.base_config_template import compose_config
from mars.configs.components.background_model_configs import (
    get_bg_nerfacto_config,
    get_bg_semanticnerfw_config,
)
from mars.configs.components.data_configs import get_vkitti_dataparser
from mars.configs.components.foreground_model_configs import (
    get_fg_carnerf_config,
    get_fg_nerfacto_config,
)
from mars.configs.components.kplanes_model_configs import get_kplanes_model_config

VKITTI_Recon_Car_Depth_Semantic = compose_config(
    method_name="vkitti-full-carnerf",
    description="vkitti reconstruction with car nerf, depth, semantic nerfacto, sky model",
    sequence_id="Scene06",
    experiment_setting="reconstruction",
    object_representation="class-wise",
    use_car_model=True,
    use_depth=True,
    use_semantics=False,
    use_sky_model=False,
    dataparser_config_generator=get_vkitti_dataparser,
    foreground_config_generator=get_fg_carnerf_config,
    background_config_generator=get_bg_nerfacto_config,
    kplanes_model_config_generator=get_kplanes_model_config,
    first_frame=80,
    last_frame=100,
    scale_factor=0.1,
    predict_normals=True,
    second_pass=True,
    start_later=True,
    start_second_pass=20000,
    mono_normal_loss_mult=0.05,
    project_name="reflection",
)
