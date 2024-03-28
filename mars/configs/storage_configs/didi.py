from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from mars.configs.storage_configs.base_storage_config import StorageConfig


@dataclass
class DidiMachine(StorageConfig):
    """
    Datasets in didi machine
    """

    datapath_dict: Dict[str, Path] = field(
        default_factory=lambda: {
            "KITTI-MOT-home": Path("/data41/luoly/datasets/NSGstudio/kitti"),
            "vKITTI-home": Path("/data41/luoly/datasets/NSGstudio/vkitti"),
            "CarNeRF-latents": Path("/data1/chenjt/datasets/ckpts/pretrain/car_nerf"),
            "CarNeRF-pretrained-model": Path(
                "/data1/chenjt/datasets/ckpts/pretrain/car_nerf"
            ),
        }
    )
