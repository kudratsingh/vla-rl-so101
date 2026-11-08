"""WO-07 policy server: one VLA behind gRPC, returning absolute-joint action chunks (rad).

Runs on the GPU box; the Mac client (``src/robot/client.py``) reaches it through an SSH tunnel,
so it binds 127.0.0.1 by default and is never exposed on a public interface:

    # GPU box
    uv run python -m src.robot.serve --preset rl --adapter <dir | user/repo/path[@rev]>
    # Mac
    ssh -N -L 50051:127.0.0.1:50051 root@<pod-ip> -p <pod-ssh-port>

Service ``vla_rl_so101.Policy``, two unary RPCs, generic handlers with raw-bytes messages (no
.proto, no grpcio-tools codegen, nothing generated for mypy):

* ``Info`` -> chunk_len, action_dim, image_size, protocol_version / protocol_hash (of the server's
  checkout), policy_id (the loaded adapter id or a mock name, never empty), the policy's
  ``describe()``, ``server_run`` (the server's run id and device), and ``adapter_sha256`` /
  ``adapter_revision``: the served adapter's content hash (``adapter_identity``) and Hub commit. The client refuses to run
  on any mismatch, and on a policy_id other than ``server.policy_checkpoint`` when that is set.

``main`` keeps a run record (``runs/<id>/record.json``) for the server's whole uptime: on the pod
it picks up the GPU name and RUNPOD_POD_ID, so the paid side of a real-arm session is priced.
* ``Act(request_id, instruction, image uint8 (S, S, 3) RGB, state float32 (6,) rad)`` ->
  ``actions float32 (C, 6)`` rad plus ``infer_ms`` (policy call) and ``server_ms`` (request
  received -> reply serialised). Bad requests abort INVALID_ARGUMENT, policy errors INTERNAL.

Wire format of every message (``pack`` / ``unpack``): u32 little-endian header length, a JSON
header, then the raw little-endian array bytes the header's ``_arrays`` table points into.

Inference is serialised (one worker, one lock): a single arm makes one request at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import signal
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
from pathlib import Path
from typing import Any, Protocol

import grpc
import numpy as np
import numpy.typing as npt

from src.config import Config, add_config_args, load_config
from src.eval.protocol import PROTOCOL_VERSION, protocol_hash
from src.robot.units import NUM_JOINTS
from src.runs import RUNS_DIR, Run, new_run_id

log = logging.getLogger(__name__)

SERVICE = "vla_rl_so101.Policy"
INFO_METHOD = f"/{SERVICE}/Info"
ACT_METHOD = f"/{SERVICE}/Act"
WIRE_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051
_ALLOWED_DTYPES = {"|u1", "<f4"}  # uint8 images, float32 state / actions
_HEADER = struct.Struct("<I")


# ------------------------------------------------------------------------------------------------
# Wire format
# ------------------------------------------------------------------------------------------------
class WireError(ValueError):
    """A message that does not decode to the expected header + arrays."""


def pack(header: Mapping[str, Any], arrays: Mapping[str, npt.ArrayLike] | None = None) -> bytes:
    table: dict[str, dict[str, Any]] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, value in (arrays or {}).items():
        a = np.asarray(value)
        a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
        if a.dtype.str not in _ALLOWED_DTYPES:
            raise WireError(f"array {name!r}: dtype {a.dtype.str} not in {sorted(_ALLOWED_DTYPES)}")
        table[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "offset": offset}
        blobs.append(a.tobytes())
        offset += a.nbytes
    head = json.dumps({**header, "_arrays": table}, default=str).encode()
    return _HEADER.pack(len(head)) + head + b"".join(blobs)


def unpack(buf: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(buf) < _HEADER.size:
        raise WireError("message shorter than its length prefix")
    (n,) = _HEADER.unpack_from(buf)
    end = _HEADER.size + n
    if end > len(buf):
        raise WireError("header length runs past the message")
    try:
        header = json.loads(buf[_HEADER.size : end])
    except ValueError as e:
        raise WireError(f"header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise WireError("header must be a JSON object")
    table = header.pop("_arrays", {})
    body = memoryview(buf)[end:]
    arrays: dict[str, np.ndarray] = {}
    for name, spec in table.items():
        if spec.get("dtype") not in _ALLOWED_DTYPES:
            raise WireError(f"array {name!r}: dtype {spec.get('dtype')!r} not allowed")
        dtype = np.dtype(spec["dtype"])
        shape = tuple(int(s) for s in spec["shape"])
        start = int(spec["offset"])
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if start < 0 or start + nbytes > len(body):
            raise WireError(f"array {name!r} runs past the message")
        arrays[name] = (
            np.frombuffer(body[start : start + nbytes], dtype=dtype).reshape(shape).copy()
        )
    return header, arrays


# ------------------------------------------------------------------------------------------------
# Policies
# ------------------------------------------------------------------------------------------------
class ServedPolicy(Protocol):
    """What the server needs: ``src.policy.base.VLAPolicy.act`` plus a JSON-able description."""

    chunk_len: int
    action_dim: int

    def act(self, obs: Mapping[str, np.ndarray], instruction: str) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


class MockPolicy:
    """TEST / DRY-RUN ONLY. Deterministic chunks that walk the arm through ``waypoints``.

    Action k of a chunk is the state moved towards the current waypoint by at most
    ``(k + 1) * step_rad`` per joint (clamped at the waypoint). The waypoint index advances every
    ``decisions_per_waypoint`` calls and then stays on the last one. A waypoint outside the safety
    limits (e.g. gripper -0.17 rad, sim "closed", below the real -5 deg floor) exercises clipping.

    ``mode`` (mutable, read on every call): ``ok`` | ``nan`` (non-finite chunk) | ``bad_shape`` |
    ``error`` (raises) | ``hang`` (blocks until ``release`` is set, at most ``hang_s`` real seconds).
    ``infer_s`` is charged to the injected ``sleep`` (a fake clock in tests, so latency is exact).
    ``reset`` restarts the walk at the first waypoint (the pilot calls it before every trial).
    """

    def __init__(
        self,
        waypoints: Sequence[npt.ArrayLike],
        *,
        chunk_len: int = 8,
        step_rad: float = float(np.radians(2.0)),
        decisions_per_waypoint: int = 3,
        infer_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        mode: str = "ok",
        hang_s: float = 10.0,
    ):
        self.waypoints = [np.asarray(w, dtype=np.float64).reshape(NUM_JOINTS) for w in waypoints]
        if not self.waypoints:
            raise ValueError("MockPolicy needs at least one waypoint")
        self.chunk_len = int(chunk_len)
        self.action_dim = NUM_JOINTS
        self.step_rad = float(step_rad)
        self.decisions_per_waypoint = max(1, int(decisions_per_waypoint))
        self.infer_s = float(infer_s)
        self.sleep = sleep
        self.mode = mode
        self.hang_s = float(hang_s)
        self.release = threading.Event()
        self.calls = 0
        self.instructions: list[str] = []

    def reset(self) -> None:
        self.calls = 0

    def act(self, obs: Mapping[str, np.ndarray], instruction: str) -> np.ndarray:
        self.calls += 1
        self.instructions.append(instruction)
        if self.mode == "error":
            raise RuntimeError("mock policy failure")
        if self.mode == "hang":
            self.release.wait(self.hang_s)
        if self.infer_s > 0:
            self.sleep(self.infer_s)
        state = np.asarray(obs["state"], dtype=np.float64).reshape(NUM_JOINTS)
        wp = self.waypoints[
            min((self.calls - 1) // self.decisions_per_waypoint, len(self.waypoints) - 1)
        ]
        k = np.arange(1, self.chunk_len + 1, dtype=np.float64)[:, None] * self.step_rad
        chunk = state + np.clip(wp - state, -k, k)
        if self.mode == "nan":
            chunk[:] = np.nan
        if self.mode == "bad_shape":
            chunk = chunk[:, :3]
        return chunk.astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "mock",
            "chunk_len": self.chunk_len,
            "action_dim": self.action_dim,
            "n_waypoints": len(self.waypoints),
            "step_deg": float(np.degrees(self.step_rad)),
        }


class OFTServed:
    """A ``make_policy`` VLA in eval mode, called under ``torch.inference_mode``."""

    def __init__(self, policy: Any):
        self.policy = policy
        self.chunk_len = int(policy.chunk_len)
        self.action_dim = int(policy.action_dim)

    def act(self, obs: Mapping[str, np.ndarray], instruction: str) -> np.ndarray:
        import torch

        with torch.inference_mode():
            out = self.policy.act(obs, instruction)
        return np.asarray(out, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = json.loads(json.dumps(self.policy.describe(), default=str))
        return d


ADAPTER_FILES = (  # what make_policy loads from an adapter dir (norm stats: NORM_STATS_FILE)
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "norm_stats.json",
)


def adapter_identity(adapter_dir: Path) -> dict[str, Any]:
    """Content identity of a resolved adapter dir: ``adapter_sha256`` over the files make_policy
    loads (name + sha256 of each one present), and ``adapter_revision``, the Hub snapshot commit
    when the dir is a ``snapshots/<sha>/...`` download (None for a local dir). A re-pushed or
    overwritten adapter under the same name gets a different hash."""
    h = hashlib.sha256()
    found = []
    for name in ADAPTER_FILES:
        f = adapter_dir / name
        if f.is_file():
            found.append(name)
            digest = hashlib.sha256()
            with f.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
            h.update(f"{name}:{digest.hexdigest()}\n".encode())
    if "adapter_config.json" not in found:
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")
    parts = adapter_dir.resolve().parts
    rev = parts[parts.index("snapshots") + 1] if "snapshots" in parts[:-1] else None
    return {"adapter_sha256": h.hexdigest(), "adapter_revision": rev, "adapter_files": found}


def load_oft(cfg: Config, device: str, adapter: str | None) -> OFTServed:
    """GPU box only (not exercised by the CPU tests): base + LoRA adapter, as the sim eval runs it.
    ``make_policy`` does not switch to eval mode, so it is done here."""
    from src.policy.factory import make_policy

    policy = make_policy(cfg, device, adapter=adapter)
    policy.eval()
    return OFTServed(policy)


# ------------------------------------------------------------------------------------------------
# Server
# ------------------------------------------------------------------------------------------------
class PolicyServer:
    def __init__(
        self,
        policy: ServedPolicy,
        *,
        image_size: int,
        policy_id: str,
        clock: Callable[[], float] = time.perf_counter,
        server_run: Mapping[str, Any] | None = None,
        adapter: Mapping[str, Any] | None = None,
    ):
        if not policy_id or policy_id.strip().lower() in ("none", "null"):
            raise ValueError(f"policy_id must name the served checkpoint, got {policy_id!r}")
        self.policy = policy
        # content identity of the served weights (``adapter_identity``); None = not known
        self.adapter = dict(adapter) if adapter is not None else {}
        self.image_size = int(image_size)
        self.policy_id = policy_id
        self.clock = clock
        self.server_run = dict(server_run) if server_run is not None else None
        self.requests = 0
        self._lock = threading.Lock()

    def info(self) -> dict[str, Any]:
        return {
            "wire_version": WIRE_VERSION,
            "chunk_len": int(self.policy.chunk_len),
            "action_dim": int(self.policy.action_dim),
            "image_size": self.image_size,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "policy_id": self.policy_id,
            "describe": self.policy.describe(),
            "server_run": self.server_run,  # run_id + device of the server's own run record
            "adapter_sha256": self.adapter.get("adapter_sha256"),
            "adapter_revision": self.adapter.get("adapter_revision"),
        }

    def handle_info(self, request: bytes, context: Any) -> bytes:
        return pack(self.info())

    def handle_act(self, request: bytes, context: Any) -> bytes:
        t_recv = self.clock()
        try:
            header, arrays = unpack(request)
            obs, instruction = self._validate(header, arrays)
        except (WireError, KeyError, TypeError, ValueError) as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"bad Act request: {e}")
            raise  # abort raises; this keeps mypy and a non-grpc caller honest
        with self._lock:
            self.requests += 1
            t0 = self.clock()
            try:
                actions = np.asarray(self.policy.act(obs, instruction), dtype=np.float32)
            except Exception as e:  # every policy failure goes back as INTERNAL
                log.exception("policy act failed")
                context.abort(grpc.StatusCode.INTERNAL, f"policy failed: {type(e).__name__}: {e}")
                raise
            infer_ms = (self.clock() - t0) * 1e3
        reply = {
            "request_id": header.get("request_id"),
            "infer_ms": infer_ms,
            "server_ms": (self.clock() - t_recv) * 1e3,
        }
        return pack(reply, {"actions": actions})

    def _validate(
        self, header: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], str]:
        instruction = header["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        image, state = arrays["image"], arrays["state"]
        s = self.image_size
        if image.dtype != np.uint8 or image.shape != (s, s, 3):
            raise ValueError(f"image must be uint8 ({s}, {s}, 3), got {image.dtype} {image.shape}")
        if state.dtype != np.float32 or state.shape != (NUM_JOINTS,):
            raise ValueError(
                f"state must be float32 ({NUM_JOINTS},), got {state.dtype} {state.shape}"
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("state has non-finite values")
        return {"image": image, "state": state}, instruction

    def generic_handler(self) -> Any:
        return grpc.method_handlers_generic_handler(
            SERVICE,
            {
                "Info": grpc.unary_unary_rpc_method_handler(self.handle_info),
                "Act": grpc.unary_unary_rpc_method_handler(self.handle_act),
            },
        )


def build_server(
    servicer: PolicyServer, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> tuple[Any, int]:
    """Started gRPC server and its bound port (``port=0`` picks a free one, as the tests do)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers((servicer.generic_handler(),))
    bound = int(server.add_insecure_port(f"{host}:{port}"))
    if bound == 0:
        raise OSError(f"could not bind {host}:{port}")
    server.start()
    log.info("policy server %s on %s:%d", servicer.policy_id, host, bound)
    return server, bound


def _sigterm(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt  # a stopped pod sends SIGTERM: close the run record the same way


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="WO-07 policy server (GPU box)")
    add_config_args(parser)
    parser.add_argument("--adapter", default=None, help="LoRA dir or user/repo/path[@rev]")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (keep 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.preset, args.config)
    policy_cfg = cfg.get("policy") or {}
    adapter = args.adapter or policy_cfg.get("adapter")  # what make_policy actually loads
    if adapter is None and not policy_cfg.get("tiny_random", False):
        parser.error("no adapter: pass --adapter or set policy.adapter (never serve a bare base)")
    policy_id = "tiny-random" if adapter is None else str(adapter)
    signal.signal(signal.SIGTERM, _sigterm)
    with Run(
        work_order="WO-07",
        script=__file__,
        cfg=cfg,
        device=args.device,
        runs_dir=args.runs_dir,
        run_id=new_run_id("WO-07", "serve"),
    ) as run:
        dev = run.record["device"]
        server_run = {"run_id": run.id} | {
            k: dev.get(k) for k in ("device", "name", "runpod_pod_id")
        }
        policy = load_oft(cfg, args.device, adapter)
        if adapter is None:
            identity: dict[str, Any] = {"adapter_sha256": "tiny-random:no-adapter"}
        else:
            from src.policy.openvla_oft import resolve_adapter

            adapter_dir = resolve_adapter(adapter)  # cached by the load above
            assert adapter_dir is not None
            identity = adapter_identity(adapter_dir)
        run.results.update(policy_id=policy_id, adapter=adapter, host=args.host, **identity)
        image_size = int((cfg.get("env") or {}).get("image_size", 224))
        servicer = PolicyServer(
            policy,
            image_size=image_size,
            policy_id=policy_id,
            server_run=server_run,
            adapter=identity,
        )
        server, port = build_server(servicer, args.host, args.port)
        run.results["port"] = port
        print(f"serving {policy_id} on {args.host}:{port} (run {run.id}); Ctrl-C to stop")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            server.stop(grace=1.0)
        finally:
            run.results["requests"] = servicer.requests
    return run.record_path


if __name__ == "__main__":
    main()
