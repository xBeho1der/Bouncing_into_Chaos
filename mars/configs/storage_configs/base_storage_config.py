from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class StorageConfig:
    """
    store machine-specific configs
    """

    datapath_dict: Dict[str, Path] = field(
        default_factory=lambda: {
            "KITTI-MOT-home": Path("/DATA_EDS/wuzr/data/kitti-MOT/"),
            "vKITTI-home": Path("/DATA_EDS2/chenjt2305/datasets/vkitti-4/"),
            "CarNeRF-latents": Path("/DATA_EDS/liuty/ckpts/pretrain/car_nerf/"),
            "CarNeRF-pretrained-model": Path(
                "/DATA_EDS/liuty/ckpts/pretrain/car_nerf/"
            ),
            "Pandaset-home": Path("/data22/DISCOVER_summer2023/huangyx/pandaset/"),
        }
    )
