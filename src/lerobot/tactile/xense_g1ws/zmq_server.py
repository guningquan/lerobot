#!/usr/bin/env python

"""
XENSE G1-WS ZMQ Tactile Server — Python 3.9/3.10.

Publishes multi-channel tactile frames from XENSE G1-WS sensors via ZMQ PUB socket
so that LeRobot (Python 3.12+) can receive them through :class:`XENSEG1WSZMQClient`.

Protocol (JSON message over ZMQ PUB/SUB):
    {
        "timestamps": {"left_fingertip": 1.234, ...},
        "tactile": {"left_fingertip": "<base64-npy>", ...}
    }

Each tactile frame is a ``(C, H, W)`` float32 NumPy array serialized via
``np.save`` + base64 encoding.

Usage:
    conda activate xense_env              # Python 3.10 env with xensesdk + pyzmq
    python zmq_server.py \\
        --serial_numbers OP000001 OP000002 OP000003 OP000004 \\
        --sensor_names left_fingertip left_knuckle right_fingertip right_knuckle \\
        --mode full \\
        --port 5556 \\
        --fps 30 \\
        --width 160 --height 120

    # Test with mock mode (no hardware):
    python zmq_server.py --mock --num_sensors 4 --port 5556
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("xense_zmq_server")


# ======================================================================
# Server
# ======================================================================


class TactileZMQServer:
    """Publishes tactile frames from multiple XENSE G1-WS sensors via ZMQ.

    Parameters
    ----------
    serial_numbers : list[str]
        XENSE sensor serial numbers (one per sensor).
    sensor_names : list[str]
        Logical names for each sensor (must match client subscription).
    mode : str
        ``"simple"`` or ``"full"``.
    port : int
        ZMQ PUB socket port.
    fps : int
        Target publication rate.
    width : int
        Output frame width after resizing.
    height : int
        Output frame height after resizing.
    output_types : list[str] | None
        OutputTypes for FULL mode (default: rectify, depth, force, force_norm, marker2d).
    mock : bool
        If True, generate synthetic frames (no hardware required).
    use_gpu : bool
        Whether to enable GPU inference in the XENSE SDK.
    """

    def __init__(
        self,
        serial_numbers: list[str],
        sensor_names: list[str],
        mode: str = "full",
        port: int = 5556,
        fps: int = 30,
        width: int = 160,
        height: int = 120,
        output_types: list[str] | None = None,
        mock: bool = False,
        use_gpu: bool = True,
    ) -> None:
        if len(serial_numbers) != len(sensor_names):
            raise ValueError(
                f"Number of serial_numbers ({len(serial_numbers)}) must match "
                f"sensor_names ({len(sensor_names)})."
            )

        self.serial_numbers = serial_numbers
        self.sensor_names = sensor_names
        self.mode = mode
        self.port = port
        self.fps = fps
        self.width = width
        self.height = height
        self.output_types = output_types or ["rectify", "depth", "force", "force_norm", "marker2d"]
        self.mock = mock
        self.use_gpu = use_gpu

        # Per-sensor SDK handles (None when mock=True)
        self._sensors: list[Any] = []
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect sensors and begin publishing frames.  Blocks until SIGINT/SIGTERM."""
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{self.port}")
        logger.info(f"ZMQ PUB socket bound to tcp://*:{self.port}")

        # Connect to all sensors
        for i, (sn, name) in enumerate(zip(self.serial_numbers, self.sensor_names)):
            sensor_handle = self._create_sensor(sn, name, i)
            self._sensors.append(sensor_handle)

        self._running = True
        logger.info(
            f"Server started: {len(self._sensors)} sensor(s), "
            f"mode={self.mode}, {self.fps} FPS, "
            f"output={self.width}x{self.height}"
        )

        # Publish loop
        interval = 1.0 / self.fps
        frame_count = 0
        try:
            while self._running:
                t_start = time.perf_counter()

                message = self._capture_frame()
                self._socket.send_string(json.dumps(message))

                frame_count += 1
                elapsed = time.perf_counter() - t_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if frame_count % (self.fps * 5) == 0:  # log every 5 s
                    logger.debug(f"Published {frame_count} frames")
        finally:
            self.stop()

    def stop(self) -> None:
        """Release all sensors and close ZMQ socket."""
        self._running = False
        for i, sensor in enumerate(self._sensors):
            if sensor is not None and not self.mock:
                try:
                    sensor.release()
                    logger.info(f"Released sensor {self.sensor_names[i]}")
                except Exception:
                    logger.exception(f"Error releasing sensor {self.sensor_names[i]}")
        self._sensors.clear()

        if hasattr(self, "_socket") and self._socket:
            self._socket.close()
        if hasattr(self, "_context") and self._context:
            self._context.term()
        logger.info("Server stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_sensor(self, serial: str, name: str, index: int) -> Any:
        """Create a single sensor handle (real or mock)."""
        if self.mock:
            logger.info(f"[{name}] mock sensor created (index={index}).")
            return None  # mock placeholder

        try:
            from xensesdk import Sensor  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "xensesdk not found. Install it with: "
                "pip install xensesdk -i https://repo.huaweicloud.com/repository/pypi/simple/"
            )
            raise

        sensor = Sensor.create(serial, use_gpu=self.use_gpu)
        logger.info(f"[{name}] sensor created (serial={serial}, gpu={self.use_gpu}).")
        return sensor

    def _capture_frame(self) -> dict[str, Any]:
        """Capture one frame from all sensors and build the JSON message."""
        timestamps: dict[str, float] = {}
        tactile: dict[str, str] = {}

        for i, (name, sensor) in enumerate(zip(self.sensor_names, self._sensors)):
            t0 = time.perf_counter()
            try:
                if self.mock:
                    frame = self._mock_frame(index=i)
                else:
                    frame = self._real_frame(sensor)
            except Exception:
                logger.exception(f"[{name}] capture error — sending zeros.")
                frame = np.zeros(self._channel_count(), dtype=np.float32)[
                    :, :self.height, :self.width
                ]

            # Serialize NumPy → base64
            buf = io.BytesIO()
            np.save(buf, frame.astype(np.float32))
            buf.seek(0)
            frame_b64 = base64.b64encode(buf.read()).decode("ascii")

            timestamps[name] = t0
            tactile[name] = frame_b64

        return {"timestamps": timestamps, "tactile": tactile}

    def _channel_count(self) -> int:
        if self.mode == "simple":
            return 1
        ch_map = {
            "rectify": 1, "difference": 1, "depth": 1,
            "force": 3, "force_norm": 3,
            "marker2d": 2,
            "mesh3d": 3, "mesh3d_init": 3, "mesh3d_flow": 3,
            "marker3d": 3, "marker3d_init": 3, "marker3d_flow": 3,
            "force_resultant": 1, "timestamp": 1,
        }
        return sum(ch_map.get(ot, 1) for ot in self.output_types)

    def _real_frame(self, sensor: Any) -> NDArray[np.float32]:
        """Capture from real XENSE hardware.

        Uses a single ``selectSensorInfo`` call for all output types
        (frame-aligned, per SDK recommendation).
        """
        from xensesdk import Sensor  # type: ignore[import-untyped]

        _OUTPUT_TYPE_MAP = {
            "rectify": "Rectify", "difference": "Difference", "depth": "Depth",
            "force": "Force", "force_norm": "ForceNorm",
            "force_resultant": "ForceResultant",
            "marker2d": "Marker2D",
            "marker3d": "Marker3D", "marker3d_init": "Marker3DInit",
            "marker3d_flow": "Marker3DFlow",
            "mesh3d": "Mesh3D", "mesh3d_init": "Mesh3DInit",
            "mesh3d_flow": "Mesh3DFlow", "timestamp": "TimeStamp",
        }

        import cv2

        if self.mode == "simple":
            rectify = sensor.selectSensorInfo(Sensor.OutputType.Rectify)
            gray = cv2.cvtColor(rectify, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if gray.shape != (self.height, self.width):
                gray = cv2.resize(gray, (self.width, self.height))
            return gray[np.newaxis, ...]  # (1, H, W)

        # FULL mode: build OutputType list
        output_enums = []
        for ot_key in self.output_types:
            ot_attr = _OUTPUT_TYPE_MAP.get(ot_key)
            if ot_attr is None:
                continue
            ot_enum = getattr(Sensor.OutputType, ot_attr, None)
            if ot_enum is not None:
                output_enums.append((ot_key, ot_enum))

        if not output_enums:
            return np.zeros((1, self.height, self.width), dtype=np.float32)

        raw_data = sensor.selectSensorInfo(*[e for _, e in output_enums])
        if len(output_enums) == 1:
            raw_data = (raw_data,)

        # Normalize + resize each output type
        channels: list[NDArray[np.float32]] = []
        for (ot_key, _), data in zip(output_enums, raw_data):
            processed = _normalize_output(ot_key, data, self.height, self.width, cv2)
            if processed.ndim == 2:
                channels.append(processed)
            elif processed.ndim == 3:
                for c in range(processed.shape[0]):
                    channels.append(processed[c])

        if not channels:
            return np.zeros((1, self.height, self.width), dtype=np.float32)
        return np.stack(channels, axis=0).astype(np.float32)  # (C, H, W)

    # ------------------------------------------------------------------
    # Mock frame generation
    # ------------------------------------------------------------------

    def _mock_frame(self, index: int = 0) -> NDArray[np.float32]:
        """Generate synthetic tactile frame for testing."""
        h, w = self.height, self.width

        if self.mode == "simple":
            frame = np.random.randn(h, w).astype(np.float32) * 0.08
            cy = h // 2 + np.random.randint(-20, 20)
            cx = w // 2 + np.random.randint(-20, 20)
            ys, xs = np.ogrid[:h, :w]
            blob = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 18**2)).astype(np.float32)
            frame += blob * 0.6
            return np.clip(frame, 0.0, 1.0)[np.newaxis, ...]

        # Shared contact blob
        cy = h // 2 + np.random.randint(-20, 20)
        cx = w // 2 + np.random.randint(-20, 20)
        ys, xs = np.ogrid[:h, :w]
        mask = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 25**2)).astype(np.float32)

        channels: list[NDArray[np.float32]] = []
        for ot in self.output_types:
            if ot == "rectify":
                channels.append(np.clip(mask + np.random.randn(h, w) * 0.05, 0.0, 1.0))
            elif ot == "difference":
                channels.append(np.clip(mask * 0.7 + np.random.randn(h, w) * 0.03, 0.0, 1.0))
            elif ot == "depth":
                channels.append(np.clip(mask * np.random.uniform(0.3, 1.0), 0.0, 1.0))
            elif ot in ("force", "force_norm"):
                for ch in [
                    mask * np.random.uniform(0.2, 1.0),
                    mask * np.sin(xs * 0.1).astype(np.float32) * 0.3,
                    mask * np.cos(ys * 0.1).astype(np.float32) * 0.3,
                ]:
                    channels.append(np.clip(ch, -1.0, 1.0) * 0.5 + 0.5)
            elif ot in ("mesh3d", "mesh3d_init", "mesh3d_flow",
                        "marker3d", "marker3d_init", "marker3d_flow"):
                for _ in range(3):
                    ch = mask * np.random.uniform(0.1, 0.5) + np.random.randn(h, w) * 0.02
                    channels.append(np.clip(ch, 0.0, 1.0))
            elif ot == "marker2d":
                for ch in [
                    mask * np.sin(xs * 0.2).astype(np.float32) * 0.2,
                    mask * np.cos(ys * 0.2).astype(np.float32) * 0.2,
                ]:
                    channels.append(np.clip(ch, -1.0, 1.0) * 0.5 + 0.5)
            elif ot == "force_resultant":
                channels.append(mask * 0.5)
            elif ot == "timestamp":
                channels.append(np.full((h, w), time.monotonic() % 60.0, dtype=np.float32))
            else:
                channels.append(np.zeros((h, w), dtype=np.float32))

        if not channels:
            return np.zeros((1, h, w), dtype=np.float32)
        return np.stack(channels, axis=0).astype(np.float32)


# ======================================================================
# Output normalization (shared with xense_g1ws.py — duplicated for
# standalone server use without depending on lerobot).
# ======================================================================

def _normalize_output(
    ot_key: str, data: NDArray[Any], target_h: int, target_w: int, cv2: Any
) -> NDArray[np.float32]:
    """Normalize + resize a single OutputType to [0, 1] range."""
    if ot_key in ("rectify", "difference"):
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray.shape[:2] != (target_h, target_w):
            gray = cv2.resize(gray, (target_w, target_h))
        return gray

    elif ot_key == "depth":
        clipped = np.clip(data.astype(np.float32), 0.0, 5.0) / 5.0
        if clipped.shape[:2] != (target_h, target_w):
            clipped = cv2.resize(clipped, (target_w, target_h))
        return clipped

    elif ot_key in ("force", "force_norm", "mesh3d", "mesh3d_init", "mesh3d_flow",
                    "marker3d", "marker3d_init", "marker3d_flow"):
        data_f = data.astype(np.float32)
        result = []
        for c in range(data_f.shape[2]):
            ch = data_f[:, :, c]
            max_val = float(np.abs(ch).max() or 1.0)
            ch_norm = ch / max_val * 0.5 + 0.5
            result.append(cv2.resize(ch_norm, (target_w, target_h)))
        return np.stack(result, axis=0)

    elif ot_key == "marker2d":
        data_f = data.astype(np.float32)
        result = []
        for c in range(data_f.shape[2]):
            ch = data_f[:, :, c]
            max_val = float(np.abs(ch).max() or 1.0)
            ch_norm = ch / max_val * 0.5 + 0.5
            result.append(cv2.resize(ch_norm, (target_w, target_h)))
        return np.stack(result, axis=0)

    elif ot_key == "force_resultant":
        data_f = data.astype(np.float32)
        max_val = float(np.abs(data_f).max() or 1.0)
        vec = data_f / max_val * 0.5 + 0.5
        return np.tile(vec[:, np.newaxis, np.newaxis], (1, target_h, target_w))

    elif ot_key == "timestamp":
        return np.full((target_h, target_w), float(data), dtype=np.float32)

    data_f = data.astype(np.float32)
    if data_f.ndim == 2:
        max_val = float(data_f.max() or 1.0)
        normalized = data_f / max_val
        if normalized.shape[:2] != (target_h, target_w):
            normalized = cv2.resize(normalized, (target_w, target_h))
        return normalized
    return np.zeros((target_h, target_w), dtype=np.float32)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="XENSE G1-WS ZMQ Tactile Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--serial_numbers", type=str, nargs="+",
        default=["OP000001", "OP000002", "OP000003", "OP000004"],
        help="XENSE sensor serial numbers (one per sensor)."
    )
    parser.add_argument(
        "--sensor_names", type=str, nargs="+",
        default=["left_fingertip", "left_knuckle", "right_fingertip", "right_knuckle"],
        help="Logical names matching client subscription."
    )
    parser.add_argument(
        "--mode", type=str, default="full", choices=["simple", "full"],
        help="Tactile mode."
    )
    parser.add_argument(
        "--port", type=int, default=5556,
        help="ZMQ PUB socket port."
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Target publication rate."
    )
    parser.add_argument(
        "--width", type=int, default=160,
        help="Output frame width after resizing."
    )
    parser.add_argument(
        "--height", type=int, default=120,
        help="Output frame height after resizing."
    )
    parser.add_argument(
        "--output_types", type=str, nargs="*",
        default=["rectify", "depth", "force", "force_norm", "marker2d"],
        help="OutputTypes for FULL mode."
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run without hardware (synthetic frames for testing)."
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=True,
        help="Enable GPU inference in XENSE SDK."
    )
    parser.add_argument(
        "--list_sensors", action="store_true",
        help="List available sensors and exit."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_sensors:
        if args.mock:
            print("Mock mode — no real sensors to list.")
            return
        try:
            from xensesdk import Sensor
            serials = Sensor.scanSerialNumber()
            if not serials:
                print("No XENSE sensors found.")
            else:
                print(f"Found {len(serials)} sensor(s):")
                for sn, cid in serials.items():
                    print(f"  Serial: {sn}  Camera ID: {cid}")
        except ImportError:
            print("xensesdk not installed. Cannot list sensors.")
        except Exception as e:
            print(f"Error scanning sensors: {e}")
        return

    server = TactileZMQServer(
        serial_numbers=args.serial_numbers,
        sensor_names=args.sensor_names,
        mode=args.mode,
        port=args.port,
        fps=args.fps,
        width=args.width,
        height=args.height,
        output_types=args.output_types,
        mock=args.mock,
        use_gpu=args.use_gpu,
    )

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, stopping...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    server.start()


if __name__ == "__main__":
    main()
