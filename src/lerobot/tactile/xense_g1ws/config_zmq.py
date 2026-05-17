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

from dataclasses import dataclass

from ..config import TactileSensorConfig, TactileSensorMode


@TactileSensorConfig.register_subclass("xense_g1ws_zmq")
@dataclass
class XENSEG1WSZMQConfig(TactileSensorConfig):
    """Configuration for XENSE G1-WS tactile sensors accessed via ZMQ bridge.

    Use this when the XENSE SDK (Python 3.9/3.10) runs in a separate process
    from LeRobot (Python 3.12+).  The ZMQ server publishes tactile frames and
    this client subscribes to receive them.

    Server startup (Python 3.10 environment):
        python -m lerobot.tactile.xense_g1ws.zmq_server \
            --serial_numbers OP000001 OP000002 OP000003 OP000004 \
            --port 5556 \
            --mode full \
            --fps 30

    Client usage (Python 3.12, within LeRobot):
        config = XENSEG1WSZMQConfig(
            server_address="localhost",
            port=5556,
            sensor_name="left_fingertip",
            mode=TactileSensorMode.FULL,
        )

    Attributes:
        server_address: IP/hostname of the ZMQ server (default "localhost").
        port: ZMQ PUB socket port (must match server).
        sensor_name: Logical sensor name matching the server's published topics.
        mode: SIMPLE or FULL (must match server mode for correct channel count).
        fps: Expected frame rate (informational).
        width: Output frame width (set by first received frame if None).
        height: Output frame height (set by first received frame if None).
        timeout_ms: ZMQ receive timeout in milliseconds.
        warmup_s: Seconds to wait for first frame during connect().
    """

    server_address: str = "localhost"
    port: int = 5556
    sensor_name: str = ""
    timeout_ms: int = 5000
    warmup_s: int = 1

    def __post_init__(self) -> None:
        from ..config import TactileSensorMode
        self.mode = TactileSensorMode(self.mode)
        if not self.server_address:
            raise ValueError("server_address cannot be empty.")
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}.")
