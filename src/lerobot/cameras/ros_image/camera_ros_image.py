#!/usr/bin/env python

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

"""
RosImageCamera - reads frames from a ROS 1 ``sensor_msgs/Image`` topic.
"""

import logging
import time
from threading import Event, Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..camera import Camera
from ..configs import ColorMode
from .configuration_ros_image import RosImageCameraConfig

# rospy and the message packages come from the ROS install (`source /opt/ros/noetic/setup.bash`),
# not from pip, so they cannot be gated by the usual `_foo_available` flags.
try:
    import rospy
    from sensor_msgs.msg import Image

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Channel order of the encodings this camera can decode.
_ENCODING_CHANNEL_ORDER = {"rgb8": ColorMode.RGB, "bgr8": ColorMode.BGR}
# Large enough for a whole frame (640x480x3 is ~0.9 MB): with rospy's 64 KB default a subscriber
# reads big messages in many small chunks and falls behind the topic.
_SUBSCRIBER_BUFF_SIZE = 2**24


class RosImageCamera(Camera):
    """
    Camera whose frames come from a ROS 1 ``sensor_msgs/Image`` topic.

    The ROS subscriber thread only keeps the newest message; it is decoded when read, so a topic
    that publishes faster than the control loop costs one decode per consumed frame. If no ROS node
    exists yet in this process, ``connect()`` creates an anonymous one; create the robots that
    initialise their own node (``init_ros_node=True``) before connecting the cameras.

    Example usage:
        ```python
        from lerobot.cameras.ros_image import RosImageCamera, RosImageCameraConfig

        config = RosImageCameraConfig(topic="/usb_cam_high/image_raw", width=640, height=480, fps=30)
        camera = RosImageCamera(config)
        camera.connect()
        frame = camera.async_read()  # (480, 640, 3) uint8, RGB
        camera.disconnect()
        ```
    """

    def __init__(self, config: RosImageCameraConfig):
        if not ROS_AVAILABLE:
            raise ImportError(
                "RosImageCamera needs rospy and sensor_msgs. Source your ROS environment first, e.g. "
                "`source ~/interbotix_ws_noetic/devel/setup.bash`."
            )
        super().__init__(config)

        self.config = config
        self.topic = config.topic
        self.color_mode = config.color_mode

        self._subscriber: Any = None
        self.frame_lock: Lock = Lock()
        self.latest_msg: Any = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"RosImageCamera({self.topic})"

    @property
    def is_connected(self) -> bool:
        """Whether the topic subscription is active."""
        return self._subscriber is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Lists the ``sensor_msgs/Image`` topics currently published (needs a running ROS master)."""
        if not ROS_AVAILABLE:
            return []
        return [
            {"name": topic, "type": "ros_image", "id": topic}
            for topic, msg_type in rospy.get_published_topics()
            if msg_type == "sensor_msgs/Image"
        ]

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Subscribe to the topic and wait for the first frame.

        The first frame is always awaited (it is how the resolution is checked), so ``warmup`` has
        no further effect.
        """
        if not rospy.core.is_initialized():
            rospy.init_node("lerobot_ros_image_camera", anonymous=True, disable_signals=True)

        with self.frame_lock:
            self.latest_msg = None
            self.latest_timestamp = None
            self.new_frame_event.clear()
        self._subscriber = rospy.Subscriber(
            self.topic, Image, self._callback, queue_size=1, buff_size=_SUBSCRIBER_BUFF_SIZE
        )

        try:
            frame = self.async_read(timeout_ms=self.config.timeout_ms)
        except TimeoutError as e:
            self._unsubscribe()
            raise ConnectionError(
                f"{self} received no image within {self.config.timeout_ms} ms. Is the launch running? "
                f"(check with `rostopic hz {self.topic}`)"
            ) from e

        height, width = frame.shape[:2]
        if self.width is None or self.height is None:
            self.width = width
            self.height = height
        elif (width, height) != (self.width, self.height):
            self._unsubscribe()
            raise RuntimeError(
                f"{self} publishes {width}x{height}, but the config asks for {self.width}x{self.height}."
            )
        logger.info(f"{self} connected ({width}x{height}).")

    def _callback(self, msg: Any) -> None:
        with self.frame_lock:
            self.latest_msg = msg
            self.latest_timestamp = time.perf_counter()
        self.new_frame_event.set()

    def _decode(self, msg: Any) -> NDArray[Any]:
        source_order = _ENCODING_CHANNEL_ORDER.get(msg.encoding)
        if source_order is None:
            raise RuntimeError(f"{self} got unsupported encoding '{msg.encoding}' (expected rgb8 or bgr8).")
        rows: NDArray[np.uint8] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        frame = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if source_order != self.color_mode:
            frame = frame[..., ::-1]
        # Copy: the buffer belongs to the ROS message and is read-only.
        return frame.copy()

    def _check_age(self, timestamp: float, max_age_ms: float) -> None:
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms). "
                "The stream has stalled."
            )

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Wait for the next frame published after this call and return it."""
        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=self.config.timeout_ms)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the newest frame not returned yet, waiting up to ``timeout_ms`` for one.

        Raises:
            TimeoutError: If no new frame arrives within ``timeout_ms``, or the newest one is older
                than ``max_age_ms``.
        """
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} async_read timeout after {timeout_ms}ms")

        with self.frame_lock:
            msg = self.latest_msg
            timestamp = self.latest_timestamp
            self.new_frame_event.clear()

        if msg is None or timestamp is None:
            raise RuntimeError(f"{self} no frame available")
        self._check_age(timestamp, self.config.max_age_ms)
        return self._decode(msg)

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the newest frame immediately, whether or not it was returned before.

        Raises:
            TimeoutError: If the newest frame is older than ``max_age_ms``.
            RuntimeError: If no frame has arrived yet.
        """
        with self.frame_lock:
            msg = self.latest_msg
            timestamp = self.latest_timestamp

        if msg is None or timestamp is None:
            raise RuntimeError(f"{self} has not received any frames yet.")
        self._check_age(timestamp, max_age_ms)
        return self._decode(msg)

    def _unsubscribe(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None
        with self.frame_lock:
            self.latest_msg = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def disconnect(self) -> None:
        """Unsubscribe from the topic."""
        self._unsubscribe()
        logger.info(f"{self} disconnected.")
