# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode

__all__ = ["RosImageCameraConfig", "ColorMode"]


@CameraConfig.register_subclass("ros_image")
@dataclass
class RosImageCameraConfig(CameraConfig):
    """Configuration for a camera read from a ROS 1 ``sensor_msgs/Image`` topic.

    Use it when another process already owns the camera device and publishes its frames, e.g. the
    ``usb_cam`` nodes started by ALOHA2's ``roslaunch aloha 4arms_tactile_teleop.launch``. The topic
    may publish faster than ``fps`` (usb_cam there runs at ~60Hz); reads always return the newest
    frame. Supported encodings are ``rgb8`` and ``bgr8``; frames are converted to ``color_mode``.

    Example:
        ```python
        RosImageCameraConfig(topic="/usb_cam_high/image_raw", width=640, height=480, fps=30)
        ```

    Attributes:
        topic: The ``sensor_msgs/Image`` topic to subscribe to.
        color_mode: Channel order of the returned frames. Defaults to RGB.
        max_age_ms: A frame older than this (time since it arrived) is treated as a stalled stream
            and raises instead of being returned.
        timeout_ms: How long ``connect()`` waits for the first frame.
    """

    topic: str
    color_mode: ColorMode = ColorMode.RGB
    max_age_ms: int = 500
    timeout_ms: int = 3000

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        if not self.topic:
            raise ValueError("`topic` cannot be empty.")
        if self.max_age_ms <= 0:
            raise ValueError(f"`max_age_ms` must be positive, but {self.max_age_ms} is provided.")
        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, but {self.timeout_ms} is provided.")
