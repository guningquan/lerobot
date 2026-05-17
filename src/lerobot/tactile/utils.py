#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .config import TactileSensorConfig
from .tactile_sensor import TactileSensor


def make_tactile_sensors_from_configs(
    sensor_configs: dict[str, TactileSensorConfig],
) -> dict[str, TactileSensor]:
    """Instantiate tactile sensors from a configuration dict.

    Args:
        sensor_configs: Dictionary mapping sensor names to their configs.

    Returns:
        Dictionary mapping sensor names to instantiated ``TactileSensor`` objects.
    """
    sensors: dict[str, TactileSensor] = {}

    for key, cfg in sensor_configs.items():
        if cfg.type == "xense_g1ws":
            from .xense_g1ws.xense_g1ws import XENSEG1WSSensor

            sensors[key] = XENSEG1WSSensor(cfg)
        elif cfg.type == "xense_g1ws_zmq":
            from .xense_g1ws.zmq_client import XENSEG1WSZMQClient

            sensors[key] = XENSEG1WSZMQClient(cfg)
        else:
            try:
                sensors[key] = cast(TactileSensor, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating tactile sensor {key} with config {cfg}: {e}") from e

    return sensors
