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

"""XENSE G1-WS photonic tactile sensor driver.

Wraps the ``xensesdk`` Python package (https://github.com/XenseRobotics/xensesdk).

Supports two operating modes:

- **SIMPLE**:  Grayscale ``Rectify`` only (1 channel) — GelSight-equivalent baseline.
- **FULL**:    Multi-channel stack of enabled ``OutputType`` values, resized to a
  common spatial resolution ``(height, width)``.

Features:
- Background-thread capture with async frame access (matching LeRobot Camera pattern).
- Mock mode for development/CI without hardware.
- Runtime config export for offline replay via ``SensorSolver``.
- Multi-sensor discovery via ``scanSerialNumber()``.
- On-demand calibration via ``calibrateSensor()``.

.. warning::
   The XENSE SDK requires **Python 3.9 or 3.10**.  LeRobot uses Python 3.12.
   For now the sensor must be run in a **separate Python 3.10 process** with data
   exchanged via shared memory, ZMQ, or file I/O.  The mock mode works on any
   Python version.

   See ``lerobot/transport/`` for inter-process transport examples.
"""

import logging
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.import_utils import is_package_available

from ..config import TactileSensorMode
from ..tactile_sensor import TactileSensor
from .config_xense_g1ws import XENSEG1WSConfig

logger = logging.getLogger(__name__)

# Guarded import — only required when mock=False
_XENSE_SDK_AVAILABLE = is_package_available("xensesdk", import_name="xensesdk")


def _require_xense_sdk() -> None:
    """Raise ImportError if xensesdk is not installed."""
    if not _XENSE_SDK_AVAILABLE:
        raise ImportError(
            "The 'xensesdk' package is required for real XENSE G1-WS hardware. "
            "Install it with: pip install xensesdk -i https://repo.huaweicloud.com/repository/pypi/simple/"
        )


# Map from config output_type keys → Sensor.OutputType enum attribute names
_OUTPUT_TYPE_MAP: dict[str, str] = {
    "rectify": "Rectify",
    "difference": "Difference",
    "depth": "Depth",
    "force": "Force",
    "force_norm": "ForceNorm",
    "force_resultant": "ForceResultant",
    "marker2d": "Marker2D",
    "marker3d": "Marker3D",
    "marker3d_init": "Marker3DInit",
    "marker3d_flow": "Marker3DFlow",
    "mesh3d": "Mesh3D",
    "mesh3d_init": "Mesh3DInit",
    "mesh3d_flow": "Mesh3DFlow",
    "timestamp": "TimeStamp",
}


class XENSEG1WSSensor(TactileSensor):
    """XENSE G1-WS photonic tactile sensor.

    Lifecycle: ``connect()`` → ``async_read()`` (per frame) → ``disconnect()``.

    The output is always a ``(C, H, W)`` float32 tensor in [0, 1]:

    - **SIMPLE mode**: ``(1, H, W)`` — grayscale contact image (GelSight-equivalent).
    - **FULL mode**:  ``(C, H, W)`` — stacked physical-state channels, each resized
      to ``(height, width)`` from their native resolutions.
    """

    config_class = XENSEG1WSConfig

    def __init__(self, config: XENSEG1WSConfig):
        super().__init__(config)
        self._cfg = config
        self._mock = config.mock

        # Background-thread state
        self._thread: Thread | None = None
        self._stop_event = False
        self._lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._frame_ready = False
        self._connected = False

        # Real SDK handle (populated by connect)
        self._sensor: Any = None

        # Determine channel count
        # SIMPLE mode: grayscale replicated to 3-ch for RGB PNG compatibility
        if config.mode == TactileSensorMode.SIMPLE:
            self.channels = 3
        else:
            self.channels = config.num_channels

        # Output resolution for policy (after resizing)
        self.height = config.height
        self.width = config.width

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def serial_number(self) -> str:
        return self._cfg.serial_number

    # ------------------------------------------------------------------
    # Static discovery
    # ------------------------------------------------------------------

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Discover connected XENSE G1-WS sensors via the SDK.

        Wraps ``Sensor.scanSerialNumber()``.

        Returns:
            List of dicts with keys ``serial_number`` and ``camera_id``.
        """
        _require_xense_sdk()
        from xensesdk import Sensor  # type: ignore[import-untyped]

        serials = Sensor.scanSerialNumber()
        return [{"serial_number": sn, "camera_id": cid} for sn, cid in serials.items()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, warmup: bool = True) -> None:
        if self._connected:
            logger.warning(f"{self} already connected.")
            return

        if self._mock:
            self._connect_mock(warmup)
        else:
            self._connect_real(warmup)

        self._stop_event = False
        self._thread = Thread(
            target=self._capture_loop,
            name=f"xense_g1ws_{self._cfg.serial_number or 'mock'}",
            daemon=True,
        )
        self._thread.start()
        self._connected = True
        logger.info(
            f"{self} connected (mode={self._cfg.mode.value}, "
            f"channels={self.channels}, output={self.width}x{self.height})."
        )

    def disconnect(self) -> None:
        self._stop_event = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._connected and not self._mock and self._sensor is not None:
            try:
                self._sensor.release()
            except Exception:
                logger.exception(f"{self} error releasing sensor.")
            self._sensor = None

        with self._lock:
            self._latest_frame = None
            self._frame_ready = False
        self._connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> None:
        """Recalibrate the XENSE G1-WS sensor.

        Wraps ``Sensor.calibrateSensor()``, which must be called when
        **no physical contact** is present on the sensor surface.
        """
        if self._mock:
            logger.info(f"{self} mock calibrate — no-op.")
            return
        if self._sensor is None:
            raise RuntimeError(f"{self} not connected — cannot calibrate.")
        _require_xense_sdk()
        self._sensor.calibrateSensor()
        logger.info(f"{self} calibrated successfully.")

    # ------------------------------------------------------------------
    # Runtime config export (for offline replay)
    # ------------------------------------------------------------------

    def export_runtime_config(self, save_dir: str | Path, binary: bool = False) -> str | None:
        """Export the sensor's runtime configuration for offline replay.

        Wraps ``Sensor.exportRuntimeConfig(save_dir, binary)``.  The exported file
        ``runtime_<serial_number>`` can later be used with ``create_solver()`` to
        perform offline depth/force/difference computation from saved rectified images.

        Args:
            save_dir: Directory to save the runtime config file.
            binary: If True, return encrypted binary data instead of writing a file.

        Returns:
            The file path if ``binary=False``, or encrypted bytes if ``binary=True``,
            or None in mock mode.
        """
        if self._mock:
            logger.info(f"{self} mock export_runtime_config — no-op.")
            return None
        if self._sensor is None:
            raise RuntimeError(f"{self} not connected — cannot export runtime config.")
        _require_xense_sdk()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result = self._sensor.exportRuntimeConfig(str(save_dir), binary=binary)
        if not binary:
            fpath = save_dir / f"runtime_{self._cfg.serial_number}"
            logger.info(f"{self} runtime config exported to {fpath}")
            return str(fpath)
        else:
            logger.info(f"{self} runtime config exported as binary ({len(result)} bytes).")
            return result

    @staticmethod
    def create_solver(runtime_path: str | Path) -> Any:
        """Create an offline ``SensorSolver`` from an exported runtime config.

        Wraps ``Sensor.createSolver(runtime_path)``.  The solver can be used to
        compute depth, force, difference, and other derived outputs from previously
        saved rectified images.

        Args:
            runtime_path: Path to a ``runtime_<serial>`` file exported by
                :meth:`export_runtime_config`.

        Returns:
            A ``SensorSolver`` instance, or ``None`` on failure.

        Example:
            >>> solver = XENSEG1WSSensor.create_solver("runtime_OP000064")
            >>> depth, force = solver.selectSensorInfo(
            ...     Sensor.OutputType.Depth,
            ...     Sensor.OutputType.Force,
            ...     rectify_image=saved_rectify_img,
            ... )
            >>> solver.release()
        """
        _require_xense_sdk()
        from xensesdk import Sensor  # type: ignore[import-untyped]

        solver = Sensor.createSolver(str(runtime_path))
        if solver is False:
            logger.error(f"Failed to create solver from {runtime_path}")
            return None
        return solver

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> NDArray[Any]:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected.")
        return self._capture_frame()

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected.")

        deadline = time.perf_counter() + timeout_ms / 1000.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frame_ready and self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._frame_ready = False
                    return frame
            time.sleep(0.001)
        raise TimeoutError(f"{self} async_read timed out after {timeout_ms}ms.")

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        interval = 1.0 / self.fps if self.fps else 1.0 / 30.0
        while not self._stop_event:
            t_start = time.perf_counter()
            try:
                frame = self._capture_frame()
                with self._lock:
                    self._latest_frame = frame
                    self._frame_ready = True
            except Exception:
                logger.exception(f"{self} capture error in background thread.")

            elapsed = time.perf_counter() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _capture_frame(self) -> NDArray[Any]:
        if self._mock:
            return self._mock_frame()
        return self._real_frame()

    # ------------------------------------------------------------------
    # Real-hardware implementation (xensesdk)
    # ------------------------------------------------------------------

    def _connect_real(self, warmup: bool = True) -> None:
        _require_xense_sdk()
        from xensesdk import Sensor  # type: ignore[import-untyped]

        serial = self._cfg.serial_number
        if not serial:
            sensors = Sensor.scanSerialNumber()
            if not sensors:
                raise RuntimeError("No XENSE G1-WS sensors found. Check USB connection and udev rules.")
            serial = list(sensors.keys())[0]
            logger.info(f"{self} auto-selected sensor serial={serial}")

        # Build kwargs for Sensor.create(), omitting empty/None values
        create_kwargs: dict[str, Any] = {"cam_id": serial, "use_gpu": self._cfg.use_gpu}
        if self._cfg.config_path:
            create_kwargs["config_path"] = self._cfg.config_path
        if not self._cfg.check_serial:
            create_kwargs["check_serial"] = self._cfg.check_serial

        self._sensor = Sensor.create(**create_kwargs)
        logger.info(f"{self} created sensor handle (serial={serial}, gpu={self._cfg.use_gpu}).")

        if warmup:
            _ = self._real_frame()

    def _real_frame(self) -> NDArray[np.float32]:
        if self._sensor is None:
            raise RuntimeError(f"{self} sensor handle is None — not connected.")

        from xensesdk import Sensor  # type: ignore[import-untyped]

        if self._cfg.mode == TactileSensorMode.SIMPLE:
            return self._real_simple_frame(Sensor)
        else:
            return self._real_full_frame(Sensor)

    def _real_simple_frame(self, Sensor: Any) -> NDArray[np.float32]:
        """SIMPLE mode: grayscale replicated to 3-ch RGB — GelSight-equivalent."""
        rectify = self._sensor.selectSensorInfo(Sensor.OutputType.Rectify)
        import cv2  # type: ignore[import-untyped]

        gray = cv2.cvtColor(rectify, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self.width and self.height:
            gray = cv2.resize(gray, (self.width, self.height))
        rgb = np.stack([gray, gray, gray], axis=0)  # (3, H, W)
        return rgb

    def _real_full_frame(self, Sensor: Any) -> NDArray[np.float32]:
        """FULL mode: multi-channel stack of enabled output types.

        All output types are requested in a **single** ``selectSensorInfo`` call
        to ensure frame alignment and optimal throughput (per SDK recommendation).
        """
        import cv2  # type: ignore[import-untyped]

        # Build the list of OutputType enums in one pass
        output_enums = []
        for ot_key in self._cfg.output_types:
            ot_attr = _OUTPUT_TYPE_MAP.get(ot_key)
            if ot_attr is None:
                logger.warning(f"{self} unknown output_type '{ot_key}', skipping.")
                continue
            ot_enum = getattr(Sensor.OutputType, ot_attr, None)
            if ot_enum is None:
                logger.warning(f"{self} Sensor.OutputType.{ot_attr} not found, skipping.")
                continue
            output_enums.append((ot_key, ot_enum))

        if not output_enums:
            return np.zeros((1, self.height, self.width), dtype=np.float32)

        # Single SDK call for all types — guarantees frame alignment
        try:
            raw_data = self._sensor.selectSensorInfo(*[enum for _, enum in output_enums])
        except Exception:
            logger.exception(f"{self} selectSensorInfo failed.")
            return np.zeros((1, self.height, self.width), dtype=np.float32)

        # selectSensorInfo returns a single value or tuple depending on count
        if len(output_enums) == 1:
            raw_data = (raw_data,)

        # Normalize and stack channels
        channels: list[NDArray[np.float32]] = []
        target_h, target_w = self.height, self.width

        for (ot_key, _), data in zip(output_enums, raw_data):
            processed = self._normalize_output(ot_key, data, target_h, target_w, cv2)
            if processed.ndim == 2:
                channels.append(processed)
            elif processed.ndim == 3:
                for c in range(processed.shape[0]):
                    channels.append(processed[c])

        if not channels:
            return np.zeros((1, target_h, target_w), dtype=np.float32)
        return np.stack(channels, axis=0).astype(np.float32)  # (C, H, W)

    @staticmethod
    def _normalize_output(
        ot_key: str, data: NDArray[Any], target_h: int, target_w: int, cv2: Any
    ) -> NDArray[np.float32]:
        """Normalize and resize a single OutputType to [0, 1] range.

        Args:
            ot_key: Output type key (e.g. "rectify", "force").
            data: Raw numpy array from the SDK.
            target_h: Target height after resizing.
            target_w: Target width after resizing.
            cv2: OpenCV module reference.

        Returns:
            ``(H, W)`` or ``(C, H, W)`` float32 array in [0, 1].
        """
        if ot_key in ("rectify", "difference"):
            # BGR (H, W, 3) → grayscale → resize → normalize
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if gray.shape[:2] != (target_h, target_w):
                gray = cv2.resize(gray, (target_w, target_h))
            return gray  # (H, W)

        elif ot_key == "depth":
            # Depth map in mm (H, W) — clip to [0, 5] mm, normalize
            clipped = np.clip(data.astype(np.float32), 0.0, 5.0) / 5.0
            if clipped.shape[:2] != (target_h, target_w):
                clipped = cv2.resize(clipped, (target_w, target_h))
            return clipped  # (H, W)

        elif ot_key in ("force", "force_norm"):
            # (35, 20, 3) → per-channel normalize → resize to (3, H, W)
            data_f = data.astype(np.float32)
            result = []
            for c in range(data_f.shape[2]):
                ch = data_f[:, :, c]
                max_val = float(np.abs(ch).max() or 1.0)
                ch_norm = ch / max_val * 0.5 + 0.5
                ch_resized = cv2.resize(ch_norm, (target_w, target_h))
                result.append(ch_resized)
            return np.stack(result, axis=0)  # (3, H, W)

        elif ot_key in ("mesh3d", "mesh3d_init", "mesh3d_flow",
                        "marker3d", "marker3d_init", "marker3d_flow"):
            # (35, 20, 3) — same pattern as force
            data_f = data.astype(np.float32)
            result = []
            for c in range(data_f.shape[2]):
                ch = data_f[:, :, c]
                max_val = float(np.abs(ch).max() or 1.0)
                ch_norm = ch / max_val * 0.5 + 0.5
                ch_resized = cv2.resize(ch_norm, (target_w, target_h))
                result.append(ch_resized)
            return np.stack(result, axis=0)

        elif ot_key == "marker2d":
            # (26, 14, 2) → per-channel normalize → resize → (2, H, W)
            data_f = data.astype(np.float32)
            result = []
            for c in range(data_f.shape[2]):
                ch = data_f[:, :, c]
                max_val = float(np.abs(ch).max() or 1.0)
                ch_norm = ch / max_val * 0.5 + 0.5
                ch_resized = cv2.resize(ch_norm, (target_w, target_h))
                result.append(ch_resized)
            return np.stack(result, axis=0)

        elif ot_key == "force_resultant":
            # (6,) 1D vector — tile spatially to (1, H, W)
            data_f = data.astype(np.float32)
            max_val = float(np.abs(data_f).max() or 1.0)
            vec = data_f / max_val * 0.5 + 0.5
            return np.tile(vec[:, np.newaxis, np.newaxis], (1, target_h, target_w))

        elif ot_key == "timestamp":
            # Scalar timestamp — fill a constant channel
            return np.full((target_h, target_w), float(data), dtype=np.float32)

        else:
            # Generic fallback
            data_f = data.astype(np.float32)
            if data_f.ndim == 2:
                max_val = float(data_f.max() or 1.0)
                normalized = data_f / max_val
                if normalized.shape[:2] != (target_h, target_w):
                    normalized = cv2.resize(normalized, (target_w, target_h))
                return normalized
            return np.zeros((target_h, target_w), dtype=np.float32)

    def _disconnect_real(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.release()
            except Exception:
                pass
            self._sensor = None

    # ------------------------------------------------------------------
    # Mock (no-hardware) implementation
    # ------------------------------------------------------------------

    def _connect_mock(self, warmup: bool = True) -> None:
        logger.info(f"{self} using MOCK mode — synthetic tactile frames at {self.width}x{self.height}.")
        if warmup:
            _ = self._mock_frame()

    def _mock_frame(self) -> NDArray[np.float32]:
        """Generate synthetic tactile data matching the real output structure.

        Native sensor resolution is 700×400; mock data is generated at the target
        output resolution directly for efficiency.

        SIMPLE mode: (1, H, W) GelSight-like deformation image.
        FULL mode:  (C, H, W) stacked channels simulating enabled output types.
        """
        h, w = self.height, self.width

        if self._cfg.mode == TactileSensorMode.SIMPLE:
            return self._mock_gelsight_like(h, w)

        # Generate shared contact blob for spatial coherence across channels
        cy = h // 2 + np.random.randint(-20, 20)
        cx = w // 2 + np.random.randint(-20, 20)
        ys, xs = np.ogrid[:h, :w]
        contact_mask = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 25**2)).astype(np.float32)

        channels: list[NDArray[np.float32]] = []
        for ot_key in self._cfg.output_types:
            if ot_key == "rectify":
                ch = contact_mask + np.random.randn(h, w).astype(np.float32) * 0.05
                channels.append(np.clip(ch, 0.0, 1.0))
            elif ot_key == "difference":
                ch = contact_mask * 0.7 + np.random.randn(h, w).astype(np.float32) * 0.03
                channels.append(np.clip(ch, 0.0, 1.0))
            elif ot_key == "depth":
                ch = contact_mask * np.random.uniform(0.3, 1.0)
                channels.append(np.clip(ch, 0.0, 1.0))
            elif ot_key in ("force", "force_norm"):
                ch_n = contact_mask * np.random.uniform(0.2, 1.0)
                ch_sx = contact_mask * np.sin(xs * 0.1).astype(np.float32) * 0.3
                ch_sy = contact_mask * np.cos(ys * 0.1).astype(np.float32) * 0.3
                for ch in [ch_n, ch_sx, ch_sy]:
                    channels.append(np.clip(ch, -1.0, 1.0) * 0.5 + 0.5)
            elif ot_key in ("mesh3d", "mesh3d_init", "mesh3d_flow",
                            "marker3d", "marker3d_init", "marker3d_flow"):
                for _ in range(3):
                    ch = contact_mask * np.random.uniform(0.1, 0.5) + \
                         np.random.randn(h, w).astype(np.float32) * 0.02
                    channels.append(np.clip(ch, 0.0, 1.0))
            elif ot_key == "marker2d":
                ch_dx = contact_mask * np.sin(xs * 0.2).astype(np.float32) * 0.2
                ch_dy = contact_mask * np.cos(ys * 0.2).astype(np.float32) * 0.2
                for ch in [ch_dx, ch_dy]:
                    channels.append(np.clip(ch, -1.0, 1.0) * 0.5 + 0.5)
            elif ot_key == "force_resultant":
                channels.append(contact_mask * 0.5)
            elif ot_key == "timestamp":
                channels.append(np.full((h, w), time.monotonic() % 60.0, dtype=np.float32))
            else:
                channels.append(np.zeros((h, w), dtype=np.float32))

        if not channels:
            return np.zeros((1, h, w), dtype=np.float32)
        return np.stack(channels, axis=0).astype(np.float32)

    @staticmethod
    def _mock_gelsight_like(h: int, w: int) -> NDArray[np.float32]:
        """Generate a GelSight-like single-channel tactile deformation image."""
        frame = np.random.randn(h, w).astype(np.float32) * 0.08
        cy = h // 2 + np.random.randint(-20, 20)
        cx = w // 2 + np.random.randint(-20, 20)
        ys, xs = np.ogrid[:h, :w]
        blob = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 18**2)).astype(np.float32)
        frame += blob * 0.6
        frame = np.clip(frame, 0.0, 1.0)
        return np.repeat(frame[np.newaxis, ...], 3, axis=0)  # (3, H, W)

    def __repr__(self) -> str:
        return (
            f"XENSEG1WSSensor(serial={self._cfg.serial_number or 'mock'}, "
            f"mode={self._cfg.mode.value}, ch={self.channels}, "
            f"out={self.width}x{self.height})"
        )
