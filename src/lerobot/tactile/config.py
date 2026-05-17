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

import abc
from dataclasses import dataclass
from enum import Enum

import draccus  # type: ignore  # TODO: add type stubs for draccus


class TactileSensorMode(str, Enum):
    """Operating mode for tactile sensors that support multiple resolution/richness levels.

    - ``SIMPLE``: Downsampled / single-channel output, equivalent to a basic GelSight.
    - ``FULL``:   Full-resolution / multi-channel output with richer physical state information.
    """

    SIMPLE = "simple"
    FULL = "full"


@dataclass(kw_only=True)
class TactileSensorConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    """Base configuration for tactile sensors.

    Follows the same pattern as ``CameraConfig`` so tactile sensors integrate
    naturally with the LeRobot hardware stack.

    Attributes:
        fps: Requested frames per second.  ``None`` means sensor-native default.
        width: Frame width in taxels / pixels.
        height: Frame height in taxels / pixels.
        mode: Operating mode (simple / full).
    """

    fps: int | None = None
    width: int | None = None
    height: int | None = None
    mode: TactileSensorMode = TactileSensorMode.FULL

    @property
    def type(self) -> str:
        return str(self.get_choice_name(self.__class__))
