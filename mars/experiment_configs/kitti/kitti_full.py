from mars.configs.base_config_template import compose_config
from mars.configs.components.background_model_configs import (
    get_bg_nerfacto_config,
    get_bg_semanticnerfw_config,
)
from mars.configs.components.data_configs import get_kitti_dataparser
from mars.configs.components.foreground_model_configs import (
    get_fg_carnerf_config,
    get_fg_nerfacto_config,
)
from mars.configs.components.kplanes_model_configs import get_kplanes_model_config

KITTI_Recon_Car_Depth_Semantic = compose_config(
    method_name="kitti-full-carnerf",
    description="kitti reconstruction with car nerf, depth, semantic nerfw, sky model",
    sequence_id="0020",
    experiment_setting="reconstruction",
    object_representation="class-wise",
    use_car_model=True,
    use_depth=True,
    use_semantics=False,
    use_sky_model=False,
    dataparser_config_generator=get_kitti_dataparser,
    foreground_config_generator=get_fg_carnerf_config,
    background_config_generator=get_bg_nerfacto_config,
    kplanes_model_config_generator=get_kplanes_model_config,
    first_frame=300,
    last_frame=309,
    predict_normals=True,
    depth_gt_type="mono",
    scale_factor=0.01,
    project_name="reflection",
    second_pass=True,
    mono_normal_loss_mult=0.05,
)
