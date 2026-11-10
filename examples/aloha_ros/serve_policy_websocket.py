#!/usr/bin/env python
"""Serve a LeRobot checkpoint over openpi's websocket protocol, for act-triple-plus-aloha2.

act-triple-plus-aloha2 drives the robot (ROS, Python 3.8) and talks to its policies through
aloha_scripts/policy_client.py, the websocket + msgpack protocol of openpi's scripts/serve_policy.py:
on connect the server sends one packed metadata dict, then every send/recv pair is one inference.
This server speaks that protocol for a LeRobot checkpoint, so act-triple's run_pi_policy.py and
hil_rollout.py run LeRobot policies (pi0.5, MolmoAct2, ACT, GR00T, ...) with `--policy lerobot`
exactly as they run openpi and N0-VTLA. The robot computer needs neither LeRobot nor its GPU.

    # on the GPU server, in this repo's environment (needs `websockets` and `msgpack`)
    python examples/aloha_ros/serve_policy_websocket.py \
        --policy_path <.../pretrained_model> --task "packaging the items" --port 8000

    # on the robot computer, in act-triple-plus-aloha2 (conda env `aloha`)
    python3 aloha_scripts/run_pi_policy.py --policy lerobot --host <server ip> --port 8000

Only checkpoints trained on ALOHA2 hdf5 episodes (act-triple's convert_aloha_hdf5_to_lerobot_v30.py,
or openpi's converter) fit: the observation is ALOHA2's RealEnv layout, i.e. what
aloha_ros_eval.py --data_format aloha2 feeds, and the returned actions are the hdf5 /action layout
act-triple sends to the followers.

Request (LerobotAlohaAdapter in act-triple's policy_client.py), one per action chunk:
    state, velocity, effort   (14,) float qpos / qvel / effort, left arm first
    images                    {camera: JPEG buffer (1-D uint8) or (H, W, 3) uint8}, RGB
    prompt                    task instruction; omitted -> --task
Reply: {"actions": (n_action_steps, action_dim) float32 absolute actions, "policy_timing",
"server_timing"}. A failed request is answered with the traceback as a text frame (openpi does the
same), which policy_client.PolicyClient raises as RuntimeError; the connection stays usable.

One client at a time: the policy and its processors are reset on every new connection, and a
second client waits until the first disconnects.
"""

import argparse
import logging
import threading
import time
import traceback
from typing import Any

import cv2
import msgpack
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR

try:  # policies may return (actions, prediction); newer LeRobot unpacks that with these helpers
    from lerobot.processor.converters import policy_output_to_transition, transition_to_policy_action
except ImportError:  # older LeRobot: predict_action_chunk returns the action tensor itself
    policy_output_to_transition = transition_to_policy_action = None

# Request keys -> LeRobot features, as aloha_ros_eval.py's aloha2_observation_frame() builds them
NUMERIC_KEYS = {"state": OBS_STATE, "velocity": f"{OBS_STR}.velocity", "effort": f"{OBS_STR}.effort"}
IMAGE_PREFIX = f"{OBS_STR}.images."


# msgpack-numpy, wire-compatible with openpi_client/msgpack_numpy.py (bytes keys, see policy_client.py)
def pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype for the wire: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj or "__ndarray__" in obj:
        data, dtype, shape = (obj.get(key, obj.get(key.decode())) for key in (b"data", b"dtype", b"shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=tuple(shape))
    if b"__npgeneric__" in obj or "__npgeneric__" in obj:
        data, dtype = (obj.get(key, obj.get(key.decode())) for key in (b"data", b"dtype"))
        return np.dtype(dtype).type(data)
    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=pack_array, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=unpack_array, raw=False)


class LeRobotAlohaPolicy:
    """A LeRobot checkpoint with its saved processors, predicting one action chunk per observation."""

    def __init__(self, policy_path: str, device: str, task: str | None, n_action_steps: int | None):
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = device
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            raise ValueError("Serve ACT without temporal ensembling: chunks are predicted, not single steps.")
        self.cfg = cfg
        self.device = torch.device(device)
        self.task = task
        self.policy = get_policy_class(cfg.type).from_pretrained(policy_path, config=cfg).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=policy_path, preprocessor_overrides={"device_processor": {"device": device}}
        )
        self.image_keys = [key for key in cfg.input_features if key.startswith(IMAGE_PREFIX)]
        # Policies that feed only some of their dataset's cameras (MolmoAct2's image_keys) need only those
        used = [key for key in getattr(cfg, "image_keys", None) or [] if key in self.image_keys]
        self.image_keys = used or self.image_keys
        self.image_shapes = {key: tuple(cfg.input_features[key].shape) for key in self.image_keys}  # (C, H, W)
        self.numeric_keys = [key for key in cfg.input_features if not key.startswith(IMAGE_PREFIX)]
        provided = set(NUMERIC_KEYS.values())
        missing = [key for key in self.numeric_keys if key not in provided]
        if missing:
            raise ValueError(f"The policy expects {missing}, which an ALOHA2 observation does not provide.")
        chunk = getattr(cfg, "chunk_size", None) or getattr(cfg, "n_action_steps", None)
        self.n_action_steps = n_action_steps or getattr(cfg, "n_action_steps", None) or chunk
        self.action_names = getattr(cfg, "action_feature_names", None)
        self._resize_warned: set[str] = set()

    def metadata(self) -> dict:
        return {
            "server": "lerobot",
            "data_format": "aloha2",
            "policy_type": self.cfg.type,
            "policy_path": str(self.cfg.pretrained_path),
            "cameras": [key[len(IMAGE_PREFIX):] for key in self.image_keys],
            "image_shapes": {key[len(IMAGE_PREFIX):]: list(shape[1:]) for key, shape in self.image_shapes.items()},
            "numeric_inputs": [key for key, feature in NUMERIC_KEYS.items() if feature in self.numeric_keys],
            "n_action_steps": int(self.n_action_steps),
            "action_dim": int(self.cfg.output_features[ACTION].shape[0]),
            "action_names": list(self.action_names) if self.action_names else None,
            "default_task": self.task,
        }

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _image(self, key: str, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 1:  # JPEG; imdecode keeps the channel order it was encoded with (RGB)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"{key}: could not decode the JPEG buffer")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key}: expected an (H, W, 3) uint8 image, got {image.shape} {image.dtype}")
        height, width = self.image_shapes[key][1:]
        if image.shape[:2] != (height, width):
            if key not in self._resize_warned:
                logging.warning(f"{key}: resizing {image.shape[:2]} to the trained {(height, width)}")
                self._resize_warned.add(key)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image)

    def frame(self, request: dict) -> tuple[dict, str]:
        """LeRobot observation frame and task from one request."""
        frame = {}
        for request_key, feature in NUMERIC_KEYS.items():
            if feature in self.numeric_keys:
                if request_key not in request:
                    raise KeyError(f"The request has no '{request_key}', which the policy needs ({feature})")
                frame[feature] = np.asarray(request[request_key], dtype=np.float32).reshape(-1)
        images = request.get("images") or {}
        for key in self.image_keys:
            camera = key[len(IMAGE_PREFIX):]
            if camera not in images:
                raise KeyError(f"The request has no image '{camera}' (got {sorted(images)})")
            frame[key] = self._image(key, images[camera])
        task = request.get("prompt") or self.task
        if not task and self.cfg.type not in ("act", "diffusion", "vqbet"):
            raise ValueError(f"A {self.cfg.type} policy needs a task: send 'prompt' or start the server with --task")
        return frame, task or ""

    @torch.inference_mode()
    def infer(self, request: dict) -> np.ndarray:
        frame, task = self.frame(request)
        batch = prepare_observation_for_inference(frame, self.device, task, "aloha")
        batch = self.preprocessor(batch)
        actions = self.policy.predict_action_chunk(batch)
        if transition_to_policy_action is not None:
            actions = transition_to_policy_action(policy_output_to_transition(actions))
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        # Unnormalize, and for relative-action checkpoints add back the state of this request
        actions = self.postprocessor(actions)
        actions = actions[0, : self.n_action_steps].float().cpu().numpy()
        if not np.isfinite(actions).all():
            raise ValueError("The policy returned non-finite actions")
        return actions

    def dummy_request(self) -> dict:
        request: dict = {key: np.zeros(14, np.float32) for key in NUMERIC_KEYS}
        request["images"] = {
            key[len(IMAGE_PREFIX):]: np.zeros((*shape[1:], 3), np.uint8) for key, shape in self.image_shapes.items()
        }
        request["prompt"] = self.task or "warm up"
        return request


def serve(policy: LeRobotAlohaPolicy, host: str, port: int) -> None:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve as ws_serve

    lock = threading.Lock()  # one client owns the policy at a time

    def handler(websocket: Any) -> None:
        peer = websocket.remote_address
        with lock:
            logging.info(f"Client {peer} connected")
            policy.reset()
            websocket.send(packb(policy.metadata()))
            served = 0
            while True:
                try:
                    message = websocket.recv()
                except ConnectionClosed:
                    break
                start = time.perf_counter()
                try:
                    request = unpackb(message)
                    infer_start = time.perf_counter()
                    actions = policy.infer(request)
                    infer_ms = 1e3 * (time.perf_counter() - infer_start)
                    reply = {
                        "actions": actions,
                        "policy_timing": {"infer_ms": infer_ms},
                        "server_timing": {"total_ms": 1e3 * (time.perf_counter() - start)},
                    }
                    websocket.send(packb(reply))
                    served += 1
                    if served == 1 or served % 50 == 0:
                        logging.info(f"{peer}: {served} chunks, last inference {infer_ms:.0f} ms")
                except ConnectionClosed:
                    break
                except Exception:
                    logging.exception("Inference failed")
                    try:
                        websocket.send(traceback.format_exc())
                    except ConnectionClosed:
                        break
            logging.info(f"Client {peer} disconnected after {served} chunks")

    with ws_serve(handler, host, port, compression=None, max_size=None) as server:
        logging.info(f"Serving {policy.cfg.type} on ws://{host}:{port}")
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy_path", required=True, help="The checkpoint's pretrained_model directory")
    parser.add_argument("--task", default=None, help="Instruction used when a request carries no prompt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n_action_steps", type=int, default=None, help="Actions returned per chunk (default: the checkpoint's)"
    )
    parser.add_argument("--no_warmup", action="store_true", help="Skip the warm-up inference at startup")
    args = parser.parse_args()
    # force: importing the policy stack can already have configured the root logger at WARNING
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    policy = LeRobotAlohaPolicy(args.policy_path, args.device, args.task, args.n_action_steps)
    logging.info(f"Loaded {policy.cfg.type}: {policy.metadata()}")
    if not args.no_warmup:
        start = time.perf_counter()
        policy.infer(policy.dummy_request())
        policy.reset()
        logging.info(f"Warm-up inference took {time.perf_counter() - start:.1f} s")
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
