from typing import Literal

from mars.data.mars_kitti_dataparser import MarsKittiDataParserConfig
from mars.data.mars_vkitti_dataparser import MarsVKittiDataParserConfig


def get_kitti_dataparser(
    sequence_id: str = "0006",
    use_depth: bool = True,
    use_semantics: bool = True,
    use_car_model: bool = True,
    experiment_setting: Literal[
        "reconstruction", "nvs-25", "nvs-50", "nvs-75"
    ] = "reconstruction",
    first_frame: int = 65,
    last_frame: int = 120,
    pred_normals: bool = True,
    scale_factor: float = 1.0,
    start_later: bool = False,
):
    """get kitti dataparser"""
    # TODO: assert
    # assert not use_semantics, "currently unsupported"
    dataparser = MarsKittiDataParserConfig(
        sequence_id=sequence_id,
        use_car_latents=use_car_model,
        use_depth=use_depth,
        use_semantic=use_semantics,
        split_setting=experiment_setting,
        first_frame=first_frame,
        last_frame=last_frame,
        pred_normals=pred_normals,
        scale_factor=scale_factor,
        start_later=start_later,
    )
    return dataparser


def get_vkitti_dataparser(
    sequence_id: str = "Scene06",
    use_depth: bool = True,
    use_semantics: bool = True,
    use_car_model: bool = True,
    experiment_setting: Literal[
        "reconstruction", "nvs-25", "nvs-50", "nvs-75"
    ] = "reconstruction",
    first_frame: int = 65,
    last_frame: int = 120,
    pred_normals: bool = False,
    scale_factor: float = 0.1,
    start_later: bool = False,
):
    """get vkitti dataparser"""
    dataparser = MarsVKittiDataParserConfig(
        sequence_id=sequence_id,
        use_car_latents=use_car_model,
        use_depth=use_depth,
        use_semantic=use_semantics,
        split_setting=experiment_setting,
        first_frame=first_frame,
        last_frame=last_frame,
        pred_normals=pred_normals,
        scale_factor=scale_factor,
        start_later=start_later,
    )
    return dataparser
