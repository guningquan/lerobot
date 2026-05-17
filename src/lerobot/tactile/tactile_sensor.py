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
import warnings
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from .config import TactileSensorConfig


class TactileSensor(abc.ABC):
    """Base class for tactile sensor implementations.

    Follows the same lifecycle as :class:`lerobot.cameras.Camera` so that
    tactile sensors can be used interchangeably in robot classes.

    Lifecycle: ``__init__`` → ``connect`` → ``async_read`` / ``read`` → ``disconnect``.

    Attributes:
        fps: Configured frames per second (may be ``None``).
        width: Frame width in taxels / pixels.
        height: Frame height in taxels / pixels.
        mode: Operating mode (simple or full).
        channels: Number of channels returned by ``read()`` / ``async_read()``.
    """

    def __init__(self, config: TactileSensorConfig):
        self.fps: int | None = config.fps
        self.width: int | None = config.width
        self.height: int | None = config.height
        self.mode = config.mode
        self.channels: int = 1  # subclasses should override based on mode

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check whether the sensor is currently connected."""
        pass

    @staticmethod
    @abc.abstractmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Detect available tactile sensors connected to the system.

        Returns:
            A list of dicts, each describing a detected sensor
            (e.g. ``{"serial": "..."}``).
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the sensor.

        Args:
            warmup: If True, capture a warmup frame before returning.
        """
        pass

    @abc.abstractmethod
    def read(self) -> NDArray[Any]:
        """Synchronously capture and return a single frame (blocking)."""
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the most recent new frame.

        Blocks up to ``timeout_ms`` only if no unconsumed frame is available.
        """
        pass

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent frame immediately (non-blocking peek)."""
        warnings.warn(
            f"{self.__class__.__name__}.read_latest() is not implemented. "
            "Please override read_latest(); it will be required in future releases.",
            FutureWarning,
            stacklevel=2,
        )
        return self.async_read()

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the sensor and release resources."""
        pass
