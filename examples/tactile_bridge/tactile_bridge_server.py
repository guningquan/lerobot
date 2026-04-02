import argparse
import pickle
import signal
import socket
import struct
import time
from typing import Dict, List, Set, Tuple

import numpy as np
from xensesdk import Sensor


def build_output_map() -> Dict[str, object]:
    return {
        "Rectify": Sensor.OutputType.Rectify,
        "Difference": Sensor.OutputType.Difference,
        "Depth": Sensor.OutputType.Depth,
        "Marker2D": Sensor.OutputType.Marker2D,
        "Force": Sensor.OutputType.Force,
        "ForceNorm": Sensor.OutputType.ForceNorm,
        "ForceResultant": Sensor.OutputType.ForceResultant,
        "Mesh3D": Sensor.OutputType.Mesh3D,
        "Mesh3DInit": Sensor.OutputType.Mesh3DInit,
        "Mesh3DFlow": Sensor.OutputType.Mesh3DFlow,
        "TimeStamp": Sensor.OutputType.TimeStamp,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream tactile sensor outputs over TCP. Supports one or more sensors."
    )
    parser.add_argument(
        "--sensor-id",
        nargs="+",
        default=["OG000614"],
        metavar="SERIAL",
        help=(
            "One or more sensor serial numbers for Sensor.create(), "
            "e.g. OG000635 or OG000635 OG000614."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for the bridge server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19090,
        help="Bind port for the bridge server.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Max streaming rate. Set <= 0 for no throttling.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["TimeStamp", "ForceResultant", "Depth"],
        help="Requested output names, for example: Rectify Depth Force TimeStamp",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference (default: CPU).",
    )
    return parser.parse_args()


def encode_value(value: object) -> Dict[str, object]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": value.shape,
            "data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        value = value.item()
    return {"kind": "scalar", "value": value}


def build_packet(sensor_frames: Dict[str, Dict[str, object]]) -> bytes:
    """Wire format: server_time + sensors[SERIAL].items[name] -> encoded value."""
    payload = {
        "server_time": time.time(),
        "sensors": {
            serial: {
                "items": {
                    name: encode_value(value)
                    for name, value in items.items()
                }
            }
            for serial, items in sensor_frames.items()
        },
    }
    packed = pickle.dumps(payload, protocol=4)
    return struct.pack("!I", len(packed)) + packed


class SensorBridgeServer:
    def __init__(
        self,
        sensor_ids: List[str],
        host: str,
        port: int,
        output_names: List[str],
        fps: float,
        use_gpu: bool = False,
    ) -> None:
        output_map = build_output_map()
        invalid = [name for name in output_names if name not in output_map]
        if invalid:
            raise ValueError(
                f"Unsupported output names: {invalid}. Available: {sorted(output_map)}"
            )

        if not sensor_ids:
            raise ValueError("At least one --sensor-id is required.")

        seen: Set[str] = set()
        unique_ids: List[str] = []
        for sid in sensor_ids:
            if sid not in seen:
                seen.add(sid)
                unique_ids.append(sid)
        self.sensor_ids = unique_ids
        self.host = host
        self.port = port
        self.output_names = output_names
        self.output_enums = [output_map[name] for name in output_names]
        self.frame_interval = 0.0 if fps <= 0 else 1.0 / fps
        self.should_stop = False
        self._sensors: Dict[str, Sensor] = {}
        self.use_gpu = use_gpu

    def stop(self, *_args: object) -> None:
        self.should_stop = True

    def _get_sensor(self, serial: str) -> Sensor:
        if serial not in self._sensors:
            self._sensors[serial] = Sensor.create(serial, use_gpu=self.use_gpu)
        return self._sensors[serial]

    def close_sensors(self) -> None:
        for sensor in self._sensors.values():
            sensor.release()
        self._sensors.clear()

    def sample_once(self) -> Dict[str, Dict[str, object]]:
        frames: Dict[str, Dict[str, object]] = {}
        for serial in self.sensor_ids:
            sensor = self._get_sensor(serial)
            result = sensor.selectSensorInfo(*self.output_enums)
            if len(self.output_names) == 1:
                values: Tuple[object, ...] = (result,)
            else:
                values = tuple(result)
            frames[serial] = {
                name: val for name, val in zip(self.output_names, values)
            }
        return frames

    def handle_client(self, conn: socket.socket) -> None:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Client connected: {conn.getpeername()}")
            while not self.should_stop:
                started_at = time.time()
                frames = self.sample_once()
                packet = build_packet(frames)
                conn.sendall(packet)

                if self.frame_interval > 0:
                    remaining = self.frame_interval - (time.time() - started_at)
                    if remaining > 0:
                        time.sleep(remaining)

    def serve_forever(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(1.0)

            infer_backend = "GPU" if self.use_gpu else "CPU"
            print(
                f"Bridge server listening on {self.host}:{self.port}, "
                f"sensors={self.sensor_ids}, infer={infer_backend}, "
                f"outputs={self.output_names}"
            )

            while not self.should_stop:
                try:
                    conn, _addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                try:
                    self.handle_client(conn)
                except (BrokenPipeError, ConnectionResetError, EOFError) as exc:
                    print(f"Client disconnected: {exc}")
                except Exception as exc:
                    print(f"Streaming error: {exc}")
                    self.close_sensors()

        self.close_sensors()
        print("Bridge server stopped.")


def main() -> None:
    args = parse_args()
    server = SensorBridgeServer(
        sensor_ids=args.sensor_id,
        host=args.host,
        port=args.port,
        output_names=args.outputs,
        fps=args.fps,
        use_gpu=args.gpu,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
