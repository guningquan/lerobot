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

"""
XENSE G1-WS ZMQ tactile client — Python 3.12 compatible.

Receives multi-channel tactile frames from a ZMQ server running in a
separate Python 3.9/3.10 process (where the XENSE SDK is available).

Protocol (JSON over ZMQ PUB/SUB):
    {
        "timestamps": {"sensor_name": float},
        "tactile": {"sensor_name": "<base64-npy>"}
    }

The tactile frames are base64-encoded NumPy arrays in ``(C, H, W)`` float32 format.
"""

import base64
import io
import json
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.import_utils import _zmq_available, require_package

if _zmq_available:
    import zmq
else:
    zmq = None  # type: ignore[no-redef]

from ..config import TactileSensorMode
from ..tactile_sensor import TactileSensor
from .config_zmq import XENSEG1WSZMQConfig

logger = logging.getLogger(__name__)


class XENSEG1WSZMQClient(TactileSensor):
    """ZMQ subscriber client for XENSE G1-WS tactile data.

    Connects to a ZMQ PUB socket, receives JSON messages containing
    base64-encoded NumPy tactile frames, and exposes them through the
    standard LeRobot ``TactileSensor`` interface.

    Lifecycle: ``connect()`` → ``async_read()`` (per frame) → ``disconnect()``.
    """

    config_class = XENSEG1WSZMQConfig

    def __init__(self, config: XENSEG1WSZMQConfig):
        require_package("pyzmq", extra="pyzmq-dep", import_name="zmq")
        super().__init__(config)

        self._cfg = config
        self.server_address = config.server_address
        self.port = config.port
        self.sensor_name = config.sensor_name
        self.timeout_ms = config.timeout_ms
        self.warmup_s = config.warmup_s

        # ZMQ resources
        self._context: Any = None  # zmq.Context
        self._socket: Any = None   # zmq.Socket (SUB)
        self._connected = False

        # Threading
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._new_frame_event = Event()
        # Track how many frames we have received to differentiate between 'no frames yet' and 'no new frames'.
        self._frame_counter = 0

    @property
    def is_connected(self) -> bool:
        return self._connected and self._context is not None and self._socket is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            logger.warning(f"{self} already connected.")
            return

        logger.info(f"Connecting to {self}...")

        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.CONFLATE, True)
            self._socket.connect(f"tcp://{self.server_address}:{self.port}")
            self._connected = True

            # Auto-detect resolution & channels from first frame
            if self.width is None or self.height is None:
                temp_frame = self._read_from_hardware()
                self.channels, self.height, self.width = temp_frame.shape
                logger.info(f"{self} detected shape: (C={self.channels}, H={self.height}, W={self.width})")

            self._start_read_thread()
            logger.info(f"{self} connected.")

            if warmup:
                start = time.time()
                while time.time() - start < self.warmup_s:
                    try:
                        self.async_read(timeout_ms=int(self.warmup_s * 1000))
                        break
                    except TimeoutError:
                        time.sleep(0.1)
                with self._frame_lock:
                    if self._latest_frame is None:
                        raise ConnectionError(f"{self} failed to receive frames during warmup.")
        except Exception:
            self._cleanup()
            raise

    def disconnect(self) -> None:
        if self._thread is not None:
            if self._stop_event:
                self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

        self._cleanup()
        with self._frame_lock:
            self._latest_frame = None
        logger.info(f"{self} disconnected.")

    def _cleanup(self) -> None:
        self._connected = False
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Auto-discovery is not available for ZMQ tactile sensors. "
            "Use the ZMQ server's --list-sensors flag to discover sensors."
        )

    def _read_from_hardware(self) -> NDArray[Any]:
        """Read a single frame directly from the ZMQ socket (blocking)."""
        if not self.is_connected or self._socket is None:
            raise RuntimeError(f"{self} not connected.")

        try:
            message = self._socket.recv_string()
        except zmq.Again as e:
            raise TimeoutError(f"{self} recv timeout after {self.timeout_ms}ms") from e

        data = json.loads(message)

        if "tactile" not in data:
            raise RuntimeError(f"{self} invalid message: missing 'tactile' key")

        tactile_frames = data["tactile"]

        # Get frame by sensor name or first available
        if self.sensor_name and self.sensor_name in tactile_frames:
            frame_b64 = tactile_frames[self.sensor_name]
        elif tactile_frames:
            frame_b64 = next(iter(tactile_frames.values()))
            # Auto-set sensor_name on first frame
            if not self.sensor_name:
                self.sensor_name = next(iter(tactile_frames.keys()))
                logger.info(f"{self} auto-detected sensor_name={self.sensor_name}")
        else:
            raise RuntimeError(f"{self} no tactile frames in message")

        # Decode base64 → NumPy array
        frame_bytes = base64.b64decode(frame_b64)
        frame = np.load(io.BytesIO(frame_bytes))

        if frame.ndim != 3:
            raise RuntimeError(f"{self} expected 3D tactile frame, got shape {frame.shape}")

        return frame.astype(np.float32)

    def read(self) -> NDArray[Any]:
        """Synchronous blocking read."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} background thread is not running.")
        self._new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest unconsumed frame with a timeout."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} background thread is not running.")

        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} async_read timed out after {timeout_ms}ms")

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self} no frame available")
        return frame

    # ------------------------------------------------------------------
    # Background read thread
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        if self._stop_event is None:
            raise RuntimeError(f"{self} stop_event is not initialized.")

        failures = 0
        while not self._stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                with self._frame_lock:
                    self._latest_frame = frame
                self._new_frame_event.set()
                failures = 0
            except (TimeoutError, Exception) as e:
                failures += 1
                if failures <= 10:
                    logger.warning(f"{self} read error: {e}")
                else:
                    logger.error(f"{self} exceeded max consecutive failures.")
                    break

    def _start_read_thread(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._frame_lock:
            self._latest_frame = None
            self._new_frame_event.clear()

        self._stop_event = Event()
        self._thread = Thread(
            target=self._read_loop, daemon=True, name=f"xense_zmq_{self.sensor_name or 'client'}"
        )
        self._thread.start()
        time.sleep(0.1)

    def __repr__(self) -> str:
        mode_str = self._cfg.mode.value if hasattr(self._cfg.mode, 'value') else str(self._cfg.mode)
        return (
            f"XENSEG1WSZMQClient({self.sensor_name}@{self.server_address}:{self.port}, "
            f"mode={mode_str}, ch={self.channels})"
        )
