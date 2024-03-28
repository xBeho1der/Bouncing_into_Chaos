from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from mars.configs.storage_configs.base_storage_config import StorageConfig


@dataclass
class AirMachine(StorageConfig):
    """
    Datasets in AIR 14/15/21/22 machine
    """

    datapath_dict: Dict[str, Path] = field(
        default_factory=lambda: {
            "KITTI-MOT-home": Path("/DATA_EDS2/chenjt2305/datasets/kitti-MOT/"),
            "vKITTI-home": Path("/DATA_EDS2/chenjt2305/datasets/vkitti-4/"),
            # "vKITTI-home": Path(
            #     "/data22/DISCOVER_summer2023/huangyx/NSG-Studio/vkitti-porsche/"
            # ),
            "CarNeRF-latents": Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/"),
            "CarNeRF-pretrained-model": Path(
                "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/"
            ),
            "Pandaset-home": Path("/data22/DISCOVER_summer2023/huangyx/pandaset/"),
        }
    )
