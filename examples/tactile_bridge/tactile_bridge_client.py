import argparse
import pickle
import socket
import struct
from typing import Dict, Iterator

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive tactile sensor outputs from tactile_bridge_server.py."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bridge server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19090,
        help="Bridge server port.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 to run forever.",
    )
    return parser.parse_args()


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_value(item: Dict[str, object]) -> object:
    kind = item["kind"]
    if kind == "none":
        return None
    if kind == "scalar":
        return item["value"]
    if kind == "ndarray":
        dtype = np.dtype(item["dtype"])
        shape = tuple(item["shape"])
        array = np.frombuffer(item["data"], dtype=dtype)
        return array.reshape(shape).copy()
    raise ValueError(f"Unknown payload kind: {kind}")


def decode_frame_message(message: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Normalize to sensors[SERIAL][output_name] = value."""
    if "sensors" in message:
        return {
            serial: {
                name: decode_value(enc)
                for name, enc in block["items"].items()
            }
            for serial, block in message["sensors"].items()
        }
    if "items" in message:
        return {
            "default": {
                name: decode_value(enc)
                for name, enc in message["items"].items()
            }
        }
    raise ValueError("Message has neither 'sensors' nor 'items'.")


class TactileBridgeClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sock = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port))

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def iter_frames(self) -> Iterator[Dict[str, object]]:
        if self.sock is None:
            self.connect()

        assert self.sock is not None
        while True:
            header = recv_exact(self.sock, 4)
            payload_size = struct.unpack("!I", header)[0]
            payload = recv_exact(self.sock, payload_size)
            message = pickle.loads(payload)

            sensors = decode_frame_message(message)
            yield {
                "server_time": message["server_time"],
                "sensors": sensors,
            }


def summarize_value(name: str, value: object) -> str:
    if value is None:
        return f"{name}: None"
    if isinstance(value, np.ndarray):
        summary = f"{name}: shape={value.shape}, dtype={value.dtype}"
        if value.ndim == 0:
            summary += f", value={value.item()}"
        elif value.size > 0 and np.issubdtype(value.dtype, np.number):
            summary += (
                f", min={float(np.min(value)):.4f},"
                f" max={float(np.max(value)):.4f}"
            )
        return summary
    return f"{name}: {value}"


def main() -> None:
    args = parse_args()
    client = TactileBridgeClient(args.host, args.port)

    try:
        for frame_index, frame in enumerate(client.iter_frames(), start=1):
            print(f"Frame {frame_index}: server_time={frame['server_time']:.6f}")
            for serial, items in frame["sensors"].items():
                label = serial if serial != "default" else "sensor (legacy)"
                print(f"  [{label}]")
                for name, value in items.items():
                    print("    " + summarize_value(name, value))
            print("-" * 80)

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        client.close()


if __name__ == "__main__":
    main()
