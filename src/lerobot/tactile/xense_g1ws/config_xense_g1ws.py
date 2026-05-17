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

from dataclasses import dataclass, field

from ..config import TactileSensorConfig, TactileSensorMode

# XENSE SDK OutputType → native spatial shape (H, W) for resizing.
# Rectify / Difference / Depth share the same 700×400 sensor image plane.
# Force / ForceNorm / Mesh3D / Mesh3DInit / Mesh3DFlow are lower-res spatial grids.
# Marker2D is a sparse marker array.
# Marker3D / Marker3DInit / Marker3DFlow are 3D marker arrays.
# ForceResultant is a 1D vector (6,).
_OUTPUT_TYPE_SHAPES: dict[str, tuple[int, int]] = {
    "rectify": (700, 400),
    "difference": (700, 400),
    "depth": (700, 400),
    "force": (35, 20),
    "force_norm": (35, 20),
    "marker2d": (26, 14),
    "mesh3d": (35, 20),
    "mesh3d_init": (35, 20),
    "mesh3d_flow": (35, 20),
    "marker3d": (35, 20),       # 3D markers share force resolution grid
    "marker3d_init": (35, 20),
    "marker3d_flow": (35, 20),
}

# Number of channels per OutputType (used to compute total output channels).
_OUTPUT_TYPE_CHANNELS: dict[str, int] = {
    "rectify": 1,        # BGR → grayscale
    "difference": 1,     # BGR → grayscale
    "depth": 1,          # scalar depth
    "force": 3,          # (35, 20, 3)
    "force_norm": 3,     # (35, 20, 3)
    "marker2d": 2,       # (26, 14, 2)
    "mesh3d": 3,         # (35, 20, 3)
    "mesh3d_init": 3,    # (35, 20, 3)
    "mesh3d_flow": 3,    # (35, 20, 3)
    "marker3d": 3,       # (35, 20, 3)
    "marker3d_init": 3,  # (35, 20, 3)
    "marker3d_flow": 3,  # (35, 20, 3)
    "force_resultant": 6,  # (6,) — tiled to (6, H, W), unstacked to 6 channels
    "timestamp": 1,
}


@TactileSensorConfig.register_subclass("xense_g1ws")
@dataclass
class XENSEG1WSConfig(TactileSensorConfig):
    """Configuration for XENSE G1-WS photonic tactile sensors.

    Wraps the ``xensesdk`` Python package.  SDK documentation:
    https://xensedoc.readthedocs.io/en/latest/

    The XENSE G1-WS exposes 11 :class:`Sensor.OutputType` values:

    ==================== =================== ====================================
    OutputType           Shape               Description
    ==================== =================== ====================================
    ``Rectify``          ``(700, 400, 3)``   Rectified BGR image
    ``Difference``       ``(700, 400, 3)``   Difference image (BGR)
    ``Depth``            ``(700, 400)``      Depth map (mm)
    ``Marker2D``         ``(26, 14, 2)``     Tangential displacement
    ``Marker3D``         ``(35, 20, 3)``     3D marker positions
    ``Marker3DInit``     ``(35, 20, 3)``     Initial 3D markers
    ``Marker3DFlow``     ``(35, 20, 3)``     Marker deformation vectors
    ``Force``            ``(35, 20, 3)``     3D force distribution
    ``ForceNorm``        ``(35, 20, 3)``     Normal force component
    ``ForceResultant``   ``(6,)``            6-DOF resultant force
    ``Mesh3D``           ``(35, 20, 3)``     Current 3D mesh
    ``Mesh3DInit``       ``(35, 20, 3)``     Initial (reference) 3D mesh
    ``Mesh3DFlow``       ``(35, 20, 3)``     Mesh deformation vectors
    ``TimeStamp``        ``float``           Sensor timestamp (seconds)
    ==================== =================== ====================================

    **Operating modes:**

    - ``SIMPLE``: Grayscale ``Rectify`` only (1 channel) — GelSight-equivalent baseline.
    - ``FULL``:   Multi-channel stack of enabled ``output_types``, each resized to
      a common spatial resolution ``(width, height)``.

    Attributes:
        serial_number: Sensor serial number (e.g. ``"OP000064"``).
        mode: ``TactileSensorMode.SIMPLE`` or ``FULL``.
        fps: Requested frames per second.
        width: Output frame width after resizing (default 160 for policy efficiency).
        height: Output frame height after resizing (default 120).
        output_types: Which ``Sensor.OutputType`` values to capture in FULL mode.
        mock: If True, run without hardware (synthetic frames for dev/testing).
        use_gpu: Whether to enable GPU inference in the XENSE SDK.
        config_path: Optional path to calibration/config directory for the SDK.
        check_serial: Whether the SDK should verify the sensor serial number.
    """

    serial_number: str = ""
    mock: bool = False

    # Output types for FULL mode (lowercase keys matching Sensor.OutputType names).
    output_types: list[str] = field(
        default_factory=lambda: ["rectify", "difference", "depth", "force", "force_norm", "force_resultant", "marker2d"]
    )

    # SDK-level parameters
    use_gpu: bool = True
    config_path: str = ""
    check_serial: bool = True

    def __post_init__(self) -> None:
        if self.width is None:
            self.width = 160
        if self.height is None:
            self.height = 120
        if self.fps is None:
            self.fps = 30

    @property
    def num_channels(self) -> int:
        """Total number of output channels based on mode and output_types."""
        if self.mode == TactileSensorMode.SIMPLE:
            return 1
        total = 0
        for ot in self.output_types:
            total += _OUTPUT_TYPE_CHANNELS.get(ot, 1)
        return total

    @staticmethod
    def get_output_type_shape(output_type: str) -> tuple[int, int] | None:
        """Return the native (H, W) shape for an output type, or None if 1D."""
        return _OUTPUT_TYPE_SHAPES.get(output_type)
