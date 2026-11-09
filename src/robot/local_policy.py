"""In-process SmolVLA / ACT on the Mac (MPS, CPU fallback) inside the WO-07 deploy loop.

The gRPC path (``src/robot/serve.py`` on a GPU box, ``PolicyClient`` on the Mac) stays. This is the
second policy client the existing ``DeployLoop`` takes: ``LocalPolicyClient.act`` has the same
contract as ``PolicyClient.act`` (an ``ActReply`` of absolute joint targets in canonical radians,
or ``ServerFault``), so the deadman, safety clip, servo-health torque-off response, prompts and
trial records are the loop's own and unchanged.

    lp = LocalPolicySpec.from_config(cfg)                  # local_policy block of configs/real.yaml
    policy = load_local_policy(lp)                         # hash check BEFORE the weights load
    dcfg = local_deploy_config(cfg, policy)
    client = LocalPolicyClient(policy, timeout_s=dcfg.request_timeout_s)
    check_local(client.info(), dcfg, lp)                   # refuses on any mismatch
    loop = DeployLoop(arm, CameraRig(...), local_pipe, client, dcfg)

Data path, all through the frozen convention (``src/robot/convention.py``), nothing redefined:
the rig's front + wrist uint8 HWC frames and the measured state (canonical radians) ->
``convention.policy_observation`` -> the checkpoint's own LeRobot preprocessor (SmolVLA: rename
front -> camera1, wrist -> camera2 with ``convention.SMOLVLA_RENAME_MAP``; tokenizer; MEAN_STD
normalizer) -> ``predict_action_chunk`` -> the checkpoint's postprocessor (unnormalize) ->
``convention.action_targets`` -> (C, 6) float64 targets -> ``So101Driver.step`` (deadman ->
``SafetyFilter`` -> the bus conversion ``canonical_to_lerobot``).

Checkpoint identity (``checkpoint_identity`` = ``src.policy.lerobot_ckpt.checkpoint_identity``,
the trainer's, also used by the real eval and the vishwaroop export), as
``serve.adapter_identity`` does for an OFT adapter: ``checkpoint_sha256`` = sha256 over
(name + sha256), sorted by name, of the checkpoint files of a pretrained dir: ``config.json``,
``model.safetensors``, ``train_config.json`` when present, both processor JSONs and every
processor state file (the normalization stats), and ``checkpoint_revision`` = the Hub snapshot
commit (None for a local dir). Loading is the trainer's ``lerobot_ckpt.load_checkpoint``. With
``local_policy.checkpoint_sha256`` set, any other hash is refused before the weights are read
into the model and again by ``check_local``. A re-saved or retrained checkpoint under the same
name gets a different hash.

Provenance (``lerobot_ckpt.PROVENANCE_FILE``, written by the trainer, inside the hash; reviews
conv-1, F1): ``load_local_policy`` refuses a checkpoint whose cited convention hash is not this
code's, or whose ``unit_map`` (``robot.units``: arm sign / offset) differs from the live
``UnitMap`` it is given; a SmolVLA checkpoint loads its VLM, processor and tokenizer from the
snapshot it trained with (``lerobot_ckpt.pinned_vlm_path``: the recorded commit, every non-weight
file hash-checked), never whatever the Hub's ``main`` is today. ``require_provenance`` (the real
arm) refuses a checkpoint without one; the mock rig and the latency script run Hub bases and
untrained ACTs without one, recorded as such. ``strict_device`` (the real arm) refuses the
MPS -> CPU fallback instead of logging it (review F2: SmolVLA on the CPU misses every deadline).

Deadline. The policy runs on one worker thread; ``act`` waits at most ``timeout_s`` real seconds
for it (the gRPC deadline's counterpart) and then re-checks the elapsed time on the loop's clock,
as ``PolicyClient`` does. A late or hung call is ``ServerFault("timeout")``: the loop trips the
deadman (HOLD). A hung call keeps the worker busy; every later ``act`` is refused
(``ServerFault("error")``) until it returns, so a second inference never starts behind it.
The first chunk of a fresh process is slower (MPS kernel compilation): ``load_local_policy``
runs ``warmup`` calls on a synthetic observation before the arm is involved.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import torch

from src.config import Config
from src.eval import protocol
from src.policy import lerobot_ckpt
from src.robot import convention as conv
from src.robot.client import ActReply, DeployConfig, ServerFault, ServerMismatch
from src.robot.units import NUM_JOINTS

log = logging.getLogger(__name__)

FAMILIES = ("smolvla", "act")
DEVICES = ("mps", "cpu")
TRANSPORT = "in_process"
RENAME_STEP = "rename_observations_processor"


class CheckpointMismatch(RuntimeError):
    """The checkpoint is not the one the config names (hash, family, chunk, features)."""


# ------------------------------------------------------------------------------------------------
# Checkpoint identity: the trainer's (src/policy/lerobot_ckpt.py), one definition for the trainer,
# this path, the real eval and the vishwaroop export
# ------------------------------------------------------------------------------------------------
checkpoint_files = lerobot_ckpt.checkpoint_files
checkpoint_identity = lerobot_ckpt.checkpoint_identity


_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+(@[\w./-]+)?$")


def resolve_checkpoint(spec: str) -> Path:
    """A local pretrained_model dir, or a Hub ``user/repo[@revision]`` downloaded (only the files
    LeRobot loads) into the default Hugging Face cache. Anything that reads as a path (``/``,
    ``.``, ``~`` first, more than one ``/``, or a first component that exists here) is local only
    and must be a directory: a mistyped path is never fetched from the Hub (review F4)."""
    local = Path(spec).expanduser()
    if local.is_dir():
        return local
    first = spec.split("/", 1)[0]
    if not _HUB_ID.match(spec) or spec.startswith((".", "~", "/")) or Path(first).exists():
        raise FileNotFoundError(f"checkpoint {spec!r} is not a directory (and not a Hub id)")
    from huggingface_hub import snapshot_download

    repo_id, _, revision = spec.partition("@")
    path = snapshot_download(
        repo_id,
        revision=revision or None,
        allow_patterns=list(lerobot_ckpt.CHECKPOINT_GLOBS),
    )
    return Path(path)


# ------------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LocalPolicySpec:
    """The ``local_policy`` block of configs/real.yaml."""

    family: str
    base: str  # checkpoint dir or Hub id[@rev]: the model name comes from config
    device: str
    checkpoint_sha256: str | None
    request_timeout_s: float
    exec_steps: int | None
    warmup: int
    # the real arm: refuse a checkpoint without the trainer's provenance, refuse MPS -> CPU
    require_provenance: bool = False
    strict_device: bool = False

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"local_policy.family {self.family!r} not in {FAMILIES}")
        if self.device not in DEVICES:
            raise ValueError(f"local_policy.device {self.device!r} not in {DEVICES}")
        if not self.base:
            raise ValueError("local_policy.base names no checkpoint")
        if self.request_timeout_s <= 0 or self.warmup < 0:
            raise ValueError("local_policy.request_timeout_ms must be > 0 and warmup >= 0")

    @classmethod
    def from_config(
        cls, cfg: Config, *, base: str | None = None, family: str | None = None
    ) -> LocalPolicySpec:
        lp = cfg.get("local_policy") or {}
        exec_steps = lp.get("exec_steps")
        sha = lp.get("checkpoint_sha256")
        return cls(
            family=str(family or lp.get("family", "")),
            base=str(base or lp.get("base") or ""),
            device=str(lp.get("device", "mps")),
            checkpoint_sha256=None if sha is None else str(sha),
            request_timeout_s=float(lp["request_timeout_ms"]) / 1e3,
            exec_steps=None if exec_steps is None else int(exec_steps),
            warmup=int(lp.get("warmup", 2)),
        )


class DeviceUnavailable(CheckpointMismatch):
    """``strict_device``: the requested device is not available (no silent CPU fallback)."""


def _canon(v: Any) -> Any:
    return json.loads(json.dumps(v, sort_keys=True, default=str))


def provenance_problems(
    prov: Mapping[str, Any] | None,
    pcfg: Any,
    *,
    units: Mapping[str, Any] | None,
    required: bool,
) -> list[str]:
    """Why a checkpoint's provenance does not fit this deploy (empty = it does)."""
    if prov is None:
        if required:
            return [f"no {lerobot_ckpt.PROVENANCE_FILE}: not saved by scripts/wo07_train.py"]
        return []
    out = []
    cited = prov.get("convention") or {}
    if cited.get("hash") != conv.convention_hash():
        out.append(
            f"trained under convention {cited.get('version')} {str(cited.get('hash'))[:12]}, "
            f"this code is {conv.CONVENTION_VERSION} {conv.convention_hash()[:12]}"
        )
    if units is not None:
        if prov.get("unit_map") is None:
            if required:
                out.append("trained on a dataset without a recorded unit_map (synthetic)")
        elif _canon(prov["unit_map"]) != _canon(units):
            out.append(f"robot.units {_canon(units)} != the training data's {prov['unit_map']}")
    if pcfg.type == "smolvla" and not prov.get("vlm"):
        out.append("a SmolVLA checkpoint without its pinned VLM revision")
    return out


def pick_device(requested: str) -> str:
    """``mps`` when this torch build has a usable MPS device, else ``cpu`` (logged)."""
    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS is not available: running the policy on the CPU (fallback)")
        return "cpu"
    return requested


def local_deploy_config(cfg: Config, policy: LocalPolicy) -> DeployConfig:
    """``DeployConfig.from_config`` with the in-process policy's chunk, deadline and executed
    steps (``deploy.chunk_len`` / ``request_timeout_ms`` describe the gRPC server).
    ``server.policy_checkpoint`` does not apply: identity is ``checkpoint_sha256``."""
    lp = LocalPolicySpec.from_config(cfg, base=policy.spec.base, family=policy.spec.family)
    return dataclasses.replace(
        DeployConfig.from_config(cfg),
        chunk_len=policy.chunk_len,
        action_dim=policy.action_dim,
        exec_steps=policy.chunk_len if lp.exec_steps is None else lp.exec_steps,
        request_timeout_s=lp.request_timeout_s,
        policy_checkpoint=None,
    )


# ------------------------------------------------------------------------------------------------
# The policy
# ------------------------------------------------------------------------------------------------
def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


class LocalPolicy:
    """A LeRobot SmolVLA / ACT checkpoint with its own pre/post processors, in eval mode."""

    def __init__(
        self,
        policy: Any,
        pre: Callable[[dict[str, Any]], dict[str, Any]],
        post: Callable[[Any], Any],
        *,
        spec: LocalPolicySpec,
        device: str,
        identity: Mapping[str, Any],
        rename_map: Mapping[str, str],
        clock: Callable[[], float] = time.perf_counter,
        provenance: Mapping[str, Any] | None = None,
    ):
        self.policy, self.pre, self.post = policy, pre, post
        self.provenance = None if provenance is None else dict(provenance)
        self.spec, self.device = spec, device
        self.identity = dict(identity)
        self.rename_map = dict(rename_map)
        self.clock = clock
        self.chunk_len = int(policy.config.chunk_size)
        self.action_dim = int(policy.config.output_features[conv.ACTION_KEY].shape[0])
        self.last_model_ms: float | None = None
        self.warmup_ms: list[float] = []

    def act(self, images: Mapping[str, Any], state: npt.ArrayLike, instruction: str) -> np.ndarray:
        """One chunk: (C, 6) float64 absolute targets, canonical radians. The policy state is
        reset first (every query is a fresh chunk, as the deploy loop executes it)."""
        batch = conv.policy_observation(images, state, instruction)
        with torch.inference_mode():
            x = self.pre(batch)
            self.policy.reset()
            _sync(self.device)
            t0 = self.clock()
            chunk = self.policy.predict_action_chunk(x)
            _sync(self.device)
            self.last_model_ms = (self.clock() - t0) * 1e3
            out = self.post(chunk)
        return conv.action_targets(out)

    def warm(self, n: int) -> None:
        """``n`` calls on a mid-grey observation at the zero pose (MPS compiles kernels on the
        first call). Never touches the arm."""
        grey = np.full(conv.IMAGE_SHAPE_HWC, 128, np.uint8)
        images = {name: grey for name in conv.CAMERA_NAMES}
        for _ in range(n):
            t0 = self.clock()
            self.act(images, np.zeros(NUM_JOINTS), conv.DEMO_TASK_STRING)
            self.warmup_ms.append(round((self.clock() - t0) * 1e3, 3))

    def describe(self) -> dict[str, Any]:
        cfg = self.policy.config
        vlm = (self.provenance or {}).get("vlm")
        return {
            "kind": "local",
            "family": self.spec.family,
            "type": str(cfg.type),
            "base": self.spec.base,
            "device": self.device,
            "dtype": str(next(self.policy.parameters()).dtype),
            "chunk_size": self.chunk_len,
            "n_action_steps": int(cfg.n_action_steps),
            "image_features": sorted(cfg.image_features),
            "rename_map": dict(self.rename_map),
            "warmup_ms": list(self.warmup_ms),
            "provenance": None if self.provenance is None else lerobot_ckpt.PROVENANCE_FILE,
            "training_run_id": (self.provenance or {}).get("training_run_id"),
            "vlm": None if not vlm else {k: vlm[k] for k in ("name", "revision")},
        }


def _rename_for(pcfg: Any, ckpt: Path) -> tuple[dict[str, str], bool]:
    """The rename map the preprocessor applies, and whether it must be passed as an override.

    A checkpoint that saved a non-empty map (a fine-tune trained with ``--rename_map``) keeps it.
    One with an empty map gets no rename when it was trained on the convention's keys (ACT), or
    ``convention.SMOLVLA_RENAME_MAP`` when its cameras are smolvla_base's ``camera1..3``. Every
    convention camera must land on a policy camera input (ACT: all of them; SmolVLA may leave
    ``camera3`` empty, ``empty_cameras``); anything else is refused."""
    saved: dict[str, str] = {}
    pre_json = ckpt / "policy_preprocessor.json"
    if pre_json.is_file():
        for step in json.loads(pre_json.read_text()).get("steps", []):
            if step.get("registry_name") == RENAME_STEP:
                saved = dict(step.get("config", {}).get("rename_map") or {})
    wanted = set(pcfg.image_features)
    exact = pcfg.type == "act"  # ACT stacks every configured camera; none may be missing

    def fits(m: Mapping[str, str]) -> bool:
        renamed = {m.get(k, k) for k in conv.IMAGE_KEYS.values()}
        return renamed == wanted if exact else renamed <= wanted

    candidates = [saved] if saved else [{}, dict(conv.SMOLVLA_RENAME_MAP)]
    for m in candidates:
        if fits(m):
            return dict(m), bool(m) and not saved
    raise CheckpointMismatch(
        f"{ckpt}: camera inputs {sorted(wanted)} do not match the convention's "
        f"{sorted(conv.IMAGE_KEYS.values())} (saved rename map {saved}, "
        f"convention map {conv.SMOLVLA_RENAME_MAP})"
    )


def _check_features(pcfg: Any, spec: LocalPolicySpec, ckpt: Path) -> None:
    bad: list[str] = []
    if pcfg.type != spec.family:
        bad.append(f"type {pcfg.type!r} != local_policy.family {spec.family!r}")
    if int(pcfg.chunk_size) != conv.CHUNK_SIZE[spec.family]:
        bad.append(f"chunk_size {pcfg.chunk_size} != convention {conv.CHUNK_SIZE[spec.family]}")
    state = pcfg.input_features.get(conv.STATE_KEY)
    if state is None or tuple(state.shape) != (NUM_JOINTS,):
        bad.append(f"{conv.STATE_KEY} must be ({NUM_JOINTS},), got {state}")
    action = pcfg.output_features.get(conv.ACTION_KEY)
    if action is None or tuple(action.shape) != (NUM_JOINTS,):
        bad.append(f"{conv.ACTION_KEY} must be ({NUM_JOINTS},), got {action}")
    if bad:
        raise CheckpointMismatch(f"{ckpt}: " + "; ".join(bad))


def load_local_policy(
    spec: LocalPolicySpec,
    *,
    clock: Callable[[], float] = time.perf_counter,
    units: Mapping[str, Any] | None = None,
) -> LocalPolicy:
    """Resolve, hash (refusing a mismatch before the weights load), check the checkpoint and its
    provenance against the convention and ``units`` (``dataclasses.asdict`` of the live
    ``UnitMap``), load it with its own processors on ``spec.device`` (MPS -> CPU fallback, refused
    with ``strict_device``) and, for SmolVLA with a provenance, the pinned VLM snapshot; warm up."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import factory  # noqa: F401 - registers every policy config type

    ckpt = resolve_checkpoint(spec.base)
    identity = checkpoint_identity(ckpt)
    if (
        spec.checkpoint_sha256 is not None
        and identity["checkpoint_sha256"] != spec.checkpoint_sha256
    ):
        raise CheckpointMismatch(
            f"{spec.base}: checkpoint_sha256 {identity['checkpoint_sha256']} != expected "
            f"local_policy.checkpoint_sha256 {spec.checkpoint_sha256}"
        )
    device = pick_device(spec.device)
    if spec.strict_device and device != spec.device:
        raise DeviceUnavailable(
            f"local_policy.device {spec.device} is not available in this Python / torch build "
            f"(it would run on {device}); use the Mac's own Terminal and .venv"
        )
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    _check_features(pcfg, spec, ckpt)
    prov = lerobot_ckpt.read_provenance(ckpt)
    bad = provenance_problems(prov, pcfg, units=units, required=spec.require_provenance)
    if bad:
        raise CheckpointMismatch(f"{spec.base}: " + "; ".join(bad))
    vlm_path = None
    if prov is not None and prov.get("vlm") and pcfg.type == "smolvla":
        try:
            vlm_path = lerobot_ckpt.pinned_vlm_path(
                prov["vlm"], weights=bool(getattr(pcfg, "load_vlm_weights", False))
            )
        except (ValueError, OSError) as e:
            raise CheckpointMismatch(f"{spec.base}: {e}") from e
    rename_map, override_rename = _rename_for(pcfg, ckpt)
    # the trainer's loader (CPU: fp32, the saved weights re-read past LeRobot's bf16 expert);
    # every chunk is executed whole by the deploy loop, so n_action_steps = chunk_size
    policy, pre, post = lerobot_ckpt.load_checkpoint(
        ckpt,
        device,
        config=pcfg,
        n_action_steps=int(pcfg.chunk_size),
        preprocessor_overrides=(
            {RENAME_STEP: {"rename_map": rename_map}} if override_rename else None
        ),
        vlm_path=vlm_path,
    )
    local = LocalPolicy(
        policy,
        pre,
        post,
        spec=spec,
        device=device,
        identity=identity,
        rename_map=rename_map,
        clock=clock,
        provenance=prov,
    )
    local.warm(spec.warmup)
    return local


# ------------------------------------------------------------------------------------------------
# The deploy loop's policy client
# ------------------------------------------------------------------------------------------------
def local_pipe(images: Mapping[str, Any]) -> dict[str, Any]:
    """``DeployLoop.pipe`` for the in-process path: the rig's full frames, unchanged (the
    convention's ``policy_observation`` is the only preprocessing; no crop)."""
    return dict(images)


class LocalPolicyClient:
    """``PolicyClient``'s contract, in process: ``act`` -> ``ActReply`` or ``ServerFault``.

    ``infer_ms`` = the model call (``predict_action_chunk``, device-synchronised); ``server_ms`` =
    the whole in-process query (convention preprocessing, LeRobot pre/post processors, the model,
    ``action_targets``), so the loop's ``network_ms`` = rtt - server_ms is the thread hand-off."""

    def __init__(
        self,
        policy: LocalPolicy,
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.perf_counter,
        run_id: str | None = None,
    ):
        self.policy = policy
        self.timeout_s = float(timeout_s)
        self.clock = clock
        self.chunk_len, self.action_dim = policy.chunk_len, policy.action_dim
        self.run_id = run_id
        self.requests = 0
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="local-policy")
        self._busy: concurrent.futures.Future[tuple[np.ndarray, float]] | None = None

    def info(self) -> dict[str, Any]:
        """``PolicyServer.info``'s fields that apply in process, plus the checkpoint identity."""
        return {
            "transport": TRANSPORT,
            "chunk_len": self.chunk_len,
            "action_dim": self.action_dim,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_hash": protocol.protocol_hash(),
            "policy_id": self.policy.spec.base,
            "describe": self.policy.describe(),
            "server_run": {"run_id": self.run_id, "device": self.policy.device},
            # what the CHECKPOINT was trained under (its provenance), not this module's citation
            "convention": (self.policy.provenance or {}).get("convention"),
            "checkpoint_sha256": self.policy.identity.get("checkpoint_sha256"),
            "checkpoint_revision": self.policy.identity.get("checkpoint_revision"),
        }

    def _call(
        self, images: Mapping[str, Any], state: npt.ArrayLike, instruction: str
    ) -> tuple[np.ndarray, float]:
        t0 = self.clock()
        actions = self.policy.act(images, state, instruction)
        return actions, (self.clock() - t0) * 1e3

    def act(self, image: Any, state: npt.ArrayLike, instruction: str) -> ActReply:
        if self._busy is not None and not self._busy.done():
            raise ServerFault("error", "the previous policy call has not returned (timed out)")
        if not isinstance(image, Mapping):
            raise ServerFault("bad_reply", f"local policy needs every camera, got {type(image)}")
        self.requests += 1
        t_send = self.clock()
        fut = self._pool.submit(self._call, image, state, instruction)
        self._busy = fut
        try:
            actions, server_ms = fut.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError as e:
            raise ServerFault(
                "timeout", f"no chunk within {self.timeout_s * 1e3:.0f} ms (still running)"
            ) from e
        except conv.ConventionError as e:
            raise ServerFault("bad_reply", f"{type(e).__name__}: {e}") from e
        except Exception as e:  # every policy failure is a server error: the loop holds
            log.exception("local policy act failed")
            raise ServerFault("error", f"policy failed: {type(e).__name__}: {e}") from e
        t_recv = self.clock()
        if t_recv - t_send > self.timeout_s:  # the deadline on the loop's own clock
            raise ServerFault(
                "timeout",
                f"chunk after {(t_recv - t_send) * 1e3:.0f} ms > {self.timeout_s * 1e3:.0f} ms",
            )
        if actions.shape != (self.chunk_len, self.action_dim):
            raise ServerFault(
                "bad_reply",
                f"actions {actions.shape}, expected ({self.chunk_len}, {self.action_dim})",
            )
        infer_ms = self.policy.last_model_ms if self.policy.last_model_ms is not None else server_ms
        return ActReply(actions, float(infer_ms), float(server_ms), t_send, t_recv)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def local_mismatches(
    info: Mapping[str, Any], dcfg: DeployConfig, spec: LocalPolicySpec
) -> list[str]:
    """``client.server_mismatches`` for the in-process policy."""
    expected = {
        "transport": TRANSPORT,
        "chunk_len": dcfg.chunk_len,
        "action_dim": dcfg.action_dim,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_hash": protocol.protocol_hash(),
    }
    out = [
        f"{k}: policy {info.get(k)!r} != client {v!r}"
        for k, v in expected.items()
        if info.get(k) != v
    ]
    cited = info.get("convention")
    if cited is None:
        if spec.require_provenance:
            out.append("convention: the checkpoint cites none (no trainer provenance)")
    elif (cited.get("hash"), cited.get("version")) != (
        conv.convention_hash(),
        conv.CONVENTION_VERSION,
    ):
        out.append(f"convention: checkpoint {cited!r} != client {conv.citation()!r}")
    sha = info.get("checkpoint_sha256")
    if not sha:
        out.append("checkpoint_sha256: none recorded")
    elif spec.checkpoint_sha256 is not None and sha != spec.checkpoint_sha256:
        out.append(
            f"checkpoint_sha256: loaded {sha} != local_policy.checkpoint_sha256 "
            f"{spec.checkpoint_sha256}"
        )
    return out


def check_local(info: Mapping[str, Any], dcfg: DeployConfig, spec: LocalPolicySpec) -> None:
    bad = local_mismatches(info, dcfg, spec)
    if bad:
        raise ServerMismatch("local policy does not match this client:\n  " + "\n  ".join(bad))


# ------------------------------------------------------------------------------------------------
# A freshly initialised ACT (no trained checkpoint yet)
# ------------------------------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def init_act_checkpoint(
    out_dir: Path,
    *,
    pose: npt.ArrayLike | None = None,
    pose_std_rad: float = 1.0,
    **overrides: Any,
) -> Path:
    """Save an UNTRAINED ACT with the convention's features (front + wrist 3x480x640, state and
    action float32[6], chunk = ``convention.CHUNK_SIZE['act']``) and LeRobot's ACT defaults, as a
    pretrained_model dir. Normalization stats are PLACEHOLDERS: state / action mean ``pose``
    (default zeros) and std ``pose_std_rad``, images ImageNet. Its actions mean nothing (they
    scatter around ``pose``); it exists to time the architecture and exercise the deploy path.
    The ResNet18 backbone is randomly initialised (``pretrained_backbone_weights=None``: same
    compute, no torchvision download). ``overrides`` change ACTConfig fields (a tiny model)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    chw = (conv.IMAGE_CHANNELS, conv.IMAGE_HEIGHT, conv.IMAGE_WIDTH)
    inputs = {key: PolicyFeature(FeatureType.VISUAL, chw) for key in conv.IMAGE_KEYS.values()}
    inputs[conv.STATE_KEY] = PolicyFeature(FeatureType.STATE, (NUM_JOINTS,))
    fields: dict[str, Any] = {
        "chunk_size": conv.CHUNK_SIZE["act"],
        "n_action_steps": conv.CHUNK_SIZE["act"],
        "pretrained_backbone_weights": None,
        "device": "cpu",
        "push_to_hub": False,
        **overrides,
    }
    cfg = ACTConfig(
        input_features=inputs,
        output_features={conv.ACTION_KEY: PolicyFeature(FeatureType.ACTION, (NUM_JOINTS,))},
        **fields,
    )
    mean = torch.zeros(NUM_JOINTS) if pose is None else torch.as_tensor(np.asarray(pose)).float()
    std = torch.full((NUM_JOINTS,), float(pose_std_rad))
    img = {
        "mean": torch.tensor(IMAGENET_MEAN).view(3, 1, 1),
        "std": torch.tensor(IMAGENET_STD).view(3, 1, 1),
    }
    stats: dict[str, dict[str, torch.Tensor]] = {key: dict(img) for key in conv.IMAGE_KEYS.values()}
    stats[conv.STATE_KEY] = {"mean": mean.clone(), "std": std.clone()}
    stats[conv.ACTION_KEY] = {"mean": mean.clone(), "std": std.clone()}
    torch.manual_seed(0)
    policy = ACTPolicy(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    pre, post = make_act_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(out_dir)
    post.save_pretrained(out_dir)
    return out_dir
