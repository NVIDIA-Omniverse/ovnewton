# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident state and control transport between Newton and ovstage.

Warp kernels encode and decode on the model device. Inbound DLTensors are aliased
while their groups are alive and copied only when devices differ.
"""

from __future__ import annotations

import contextlib
import ctypes
from collections import Counter
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import warp as wp
from ovstage import (
    AttributeSemantic,
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLTensor,
    WriteDesc,
    make_dltensor,
)

from . import _stage
from ._errors import OvstageContractError
from ._schema_names import (
    BODY_ANGULAR_VELOCITY,
    BODY_VELOCITY,
    RESET_XFORM_STACK,
    WORLD_MATRIX,
    XFORM,
    joint_drive,
    joint_state,
)
from .ovnewton import (
    ObjectScope,
    Query,
    ReadGroup,
    ReadResult,
    SimObjectType,
    _destroy_path_lists,
    _finalize_path_lists,
    _ReadStorage,
)

_DEG_PER_RAD = 57.29577951308232
_FLOAT32_MAX = 3.4028234663852886e38
_NEWTON_LIMIT_MAX = 1.0e10

_BODY_Q = "body_q"
_BODY_QD = "body_qd"
_POSITION = "position"
_ORIENTATION = "orientation"
_LINEAR_VELOCITY = "linearVelocity"
_ANGULAR_VELOCITY = "angularVelocity"
_LINEAR_ACCELERATION = "linearAcceleration"
_ANGULAR_ACCELERATION = "angularAcceleration"
_GRAVITY = "gravity"
_MASS = "mass"
_INERTIA = "inertia"
_CENTER_OF_MASS_POSITION = "centerOfMassPosition"
_SHAPE_COUNT = "shapeCount"
_FRICTION = "friction"
_RESTITUTION = "restitution"
_ROOT_POSITION = "rootPosition"
_ROOT_ORIENTATION = "rootOrientation"
_ROOT_LINEAR_VELOCITY = "rootLinearVelocity"
_ROOT_ANGULAR_VELOCITY = "rootAngularVelocity"
_JOINT_POSITION = "jointPosition"
_JOINT_VELOCITY = "jointVelocity"
_JOINT_POSITION_TARGET = "jointPositionTarget"
_JOINT_VELOCITY_TARGET = "jointVelocityTarget"
_JOINT_STIFFNESS = "jointStiffness"
_JOINT_DAMPING = "jointDamping"
_JOINT_LIMIT = "jointLimit"
_JOINT_MAX_VELOCITY = "jointMaxVelocity"
_JOINT_MAX_FORCE = "jointMaxForce"
_JOINT_ARMATURE = "jointArmature"
_JOINT_FRICTION = "jointFriction"
_JOINT_Q = "joint_q"
_JOINT_QD = "joint_qd"


@dataclass(frozen=True)
class _OutputDescriptor:
    """One Newton output source behind the public read interface."""

    domain: str
    owner: str
    source: str
    width: int
    conversion: Optional[str] = None


_OUTPUT_DESCRIPTORS = MappingProxyType(
    {
        _BODY_Q: _OutputDescriptor("body", "state", "body_q", 7),
        _BODY_QD: _OutputDescriptor("body", "state", "body_qd", 6),
        _POSITION: _OutputDescriptor("body", "state", "body_q", 3, "transform_position"),
        _ORIENTATION: _OutputDescriptor("body", "state", "body_q", 4, "transform_orientation"),
        _LINEAR_VELOCITY: _OutputDescriptor("body", "state", "body_qd", 3, "spatial_linear"),
        _ANGULAR_VELOCITY: _OutputDescriptor("body", "state", "body_qd", 3, "spatial_angular"),
        _LINEAR_ACCELERATION: _OutputDescriptor("body", "state", "body_qdd", 3, "spatial_linear"),
        _ANGULAR_ACCELERATION: _OutputDescriptor("body", "state", "body_qdd", 3, "spatial_angular"),
        _GRAVITY: _OutputDescriptor("scene", "model", "gravity", 3),
        _MASS: _OutputDescriptor("body", "model", "body_mass", 1),
        _INERTIA: _OutputDescriptor("body", "model", "body_inertia", 9),
        _CENTER_OF_MASS_POSITION: _OutputDescriptor("body", "model", "body_com", 3),
        _SHAPE_COUNT: _OutputDescriptor("shape", "query", "shape_counts", 1),
        _FRICTION: _OutputDescriptor("shape", "model", "shape_material_mu", 0),
        _RESTITUTION: _OutputDescriptor("shape", "model", "shape_material_restitution", 0),
        _ROOT_POSITION: _OutputDescriptor("articulation", "state", "body_q", 3, "transform_position"),
        _ROOT_ORIENTATION: _OutputDescriptor("articulation", "state", "body_q", 4, "transform_orientation"),
        _ROOT_LINEAR_VELOCITY: _OutputDescriptor("articulation", "state", "body_qd", 3, "spatial_linear"),
        _ROOT_ANGULAR_VELOCITY: _OutputDescriptor("articulation", "state", "body_qd", 3, "spatial_angular"),
        _JOINT_POSITION: _OutputDescriptor("joint_position", "state", "joint_q", 0, "angular_degrees"),
        _JOINT_VELOCITY: _OutputDescriptor("joint_velocity", "state", "joint_qd", 0, "angular_degrees"),
        _JOINT_POSITION_TARGET: _OutputDescriptor(
            "joint_target_position", "control", "joint_target_q", 0, "angular_degrees"
        ),
        _JOINT_VELOCITY_TARGET: _OutputDescriptor(
            "joint_velocity", "control", "joint_target_qd", 0, "angular_degrees"
        ),
        _JOINT_STIFFNESS: _OutputDescriptor(
            "joint_velocity", "model", "joint_target_ke", 0, "angular_per_degree"
        ),
        _JOINT_DAMPING: _OutputDescriptor(
            "joint_velocity", "model", "joint_target_kd", 0, "angular_per_degree"
        ),
        _JOINT_LIMIT: _OutputDescriptor("joint_limit", "model", "joint_limit_lower", 0, "angular_degrees"),
        _JOINT_MAX_VELOCITY: _OutputDescriptor(
            "joint_velocity", "model", "joint_velocity_limit", 0, "angular_degrees"
        ),
        _JOINT_MAX_FORCE: _OutputDescriptor("joint_native_dof", "model", "joint_effort_limit", 0),
        _JOINT_ARMATURE: _OutputDescriptor("joint_native_dof", "model", "joint_armature", 0),
        _JOINT_FRICTION: _OutputDescriptor("joint_native_dof", "model", "joint_friction", 0),
        _JOINT_Q: _OutputDescriptor("joint", "state", "joint_q", 0),
        _JOINT_QD: _OutputDescriptor("joint", "state", "joint_qd", 0),
    }
)
_OUTPUT_ATTRIBUTES = tuple(_OUTPUT_DESCRIPTORS)


@dataclass
class _JointChannel:
    """Cached paths, indices, and buffers for one single-DOF USD instance."""

    instance: str
    rotational: bool
    paths: Tuple[str, ...]
    q_indices_host: Tuple[int, ...]
    qd_indices_host: Tuple[int, ...]
    target_q_indices_host: Tuple[int, ...]
    position_target_rows: frozenset[int]
    velocity_target_rows: frozenset[int]
    device: Any
    position_out: Any = field(init=False)
    velocity_out: Any = field(init=False)
    q_indices: Any = field(init=False)
    qd_indices: Any = field(init=False)

    def __post_init__(self) -> None:
        self.position_out = wp.empty(len(self.paths), dtype=wp.float32, device=self.device)
        self.velocity_out = wp.empty(len(self.paths), dtype=wp.float32, device=self.device)
        self.q_indices = wp.array(self.q_indices_host, dtype=wp.int32, device=self.device)
        self.qd_indices = wp.array(self.qd_indices_host, dtype=wp.int32, device=self.device)


@dataclass
class _PendingJointChannel:
    """Host-side accumulator used only while resolving immutable joint indices."""

    rotational: bool
    paths: List[str] = field(default_factory=list)
    q_indices: List[int] = field(default_factory=list)
    qd_indices: List[int] = field(default_factory=list)
    target_q_indices: List[int] = field(default_factory=list)
    position_target_rows: set[int] = field(default_factory=set)
    velocity_target_rows: set[int] = field(default_factory=set)

    def add(
        self,
        path: str,
        q: int,
        qd: int,
        target_q: int,
        *,
        position_target: bool,
        velocity_target: bool,
    ) -> None:
        row = len(self.paths)
        self.paths.append(path)
        self.q_indices.append(q)
        self.qd_indices.append(qd)
        self.target_q_indices.append(target_q)
        if position_target:
            self.position_target_rows.add(row)
        if velocity_target:
            self.velocity_target_rows.add(row)

    def finish(self, instance: str, device: Any) -> _JointChannel:
        return _JointChannel(
            instance=instance,
            rotational=self.rotational,
            paths=tuple(self.paths),
            q_indices_host=tuple(self.q_indices),
            qd_indices_host=tuple(self.qd_indices),
            target_q_indices_host=tuple(self.target_q_indices),
            position_target_rows=frozenset(self.position_target_rows),
            velocity_target_rows=frozenset(self.velocity_target_rows),
            device=device,
        )


@dataclass
class _RuntimeTransport:
    """Fixed model/path mapping plus binding-owned reusable device storage."""

    model: Any
    channels: Tuple[_JointChannel, ...]
    derived_articulation_indices_host: Tuple[int, ...]
    derived_q_indices_host: Tuple[int, ...]
    derived_qd_indices_host: Tuple[int, ...]
    device: Any = field(init=False)
    pose_out: Any = field(init=False)
    body_scale: Any = field(init=False, default=None)
    reset_xform_stack_out: Any = field(init=False)
    linear_out: Any = field(init=False)
    angular_out: Any = field(init=False)
    linear_in: Any = field(init=False)
    angular_in: Any = field(init=False)
    body_rows: Any = field(init=False)
    body_q_in: Any = field(init=False)
    body_qd_in: Any = field(init=False)
    joint_q_in: Any = field(init=False)
    joint_qd_in: Any = field(init=False)
    ik_joint_q: Any = field(init=False, default=None)
    ik_joint_qd: Any = field(init=False, default=None)
    target_q_in: Any = field(init=False, default=None)
    target_qd_in: Any = field(init=False, default=None)
    derived_articulation_indices: Any = field(init=False, default=None)
    derived_q_indices: Any = field(init=False, default=None)
    derived_qd_indices: Any = field(init=False, default=None)
    paths: Tuple[str, ...] = field(init=False, default=())
    channel_rows: Dict[str, Tuple[int, ...]] = field(init=False, default_factory=dict)
    read_plan: Optional["_ReadPlan"] = field(init=False, default=None)
    position_target_required: bool = field(init=False, default=False)
    velocity_target_required: bool = field(init=False, default=False)
    producer_event: Any = field(init=False, default=None)
    _index_arrays: Dict[Tuple[int, ...], Any] = field(init=False, default_factory=dict)
    _tokens: Dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.device = wp.get_device(self.model.device)
        if self.device.is_cuda:
            self.producer_event = wp.Event(self.device)
        body_count = self.model.body_count
        self.pose_out = wp.empty(body_count * 16, dtype=wp.float64, device=self.device)
        self.reset_xform_stack_out = wp.ones(body_count, dtype=wp.bool, device=self.device)
        self.linear_out = wp.empty(body_count * 3, dtype=wp.float32, device=self.device)
        self.angular_out = wp.empty(body_count * 3, dtype=wp.float32, device=self.device)
        self.linear_in = wp.empty(body_count, dtype=wp.vec3, device=self.device)
        self.angular_in = wp.empty(body_count, dtype=wp.vec3, device=self.device)
        self.body_rows = self.indices(range(body_count))
        self.body_q_in = wp.empty(body_count, dtype=wp.transformf, device=self.device)
        self.body_qd_in = wp.empty(body_count, dtype=wp.spatial_vectorf, device=self.device)
        self.joint_q_in = wp.empty(self.model.joint_q.shape, dtype=wp.float32, device=self.device)
        self.joint_qd_in = wp.empty(self.model.joint_qd.shape, dtype=wp.float32, device=self.device)
        if self.derived_articulation_indices_host:
            self.derived_articulation_indices = self.indices(self.derived_articulation_indices_host)
            self.derived_q_indices = self.indices(self.derived_q_indices_host)
            self.derived_qd_indices = self.indices(self.derived_qd_indices_host)
            self.ik_joint_q = wp.empty(self.model.joint_q.shape, dtype=wp.float32, device=self.device)
            self.ik_joint_qd = wp.empty(self.model.joint_qd.shape, dtype=wp.float32, device=self.device)
        control = self.model.control()
        if control.joint_target_q is not None:
            self.target_q_in = wp.empty(control.joint_target_q.shape, dtype=wp.float32, device=self.device)
        if control.joint_target_qd is not None:
            self.target_qd_in = wp.empty(control.joint_target_qd.shape, dtype=wp.float32, device=self.device)
        self.position_target_required = any(channel.position_target_rows for channel in self.channels)
        self.velocity_target_required = any(channel.velocity_target_rows for channel in self.channels)

    def indices(self, values: Sequence[int]):
        key = tuple(int(value) for value in values)
        result = self._index_arrays.get(key)
        if result is None:
            result = wp.array(key, dtype=wp.int32, device=self.device)
            self._index_arrays[key] = result
        return result

    def token(self, pd: Any, attribute: str) -> int:
        """Intern one runtime attribute once."""
        result = self._tokens.get(attribute)
        if result is None:
            result = int(pd.intern_token(attribute))
            self._tokens[attribute] = result
        return result


@dataclass
class _ReadColumn:
    """One typed column in a batched latest-payload read."""

    attribute: str
    dtype: Any
    lanes: int
    expected_rows: frozenset[int]
    consume: Any
    destination_rows: Any
    kind: str = "runtime attribute"
    phase: int = 0
    accepted_rows: Optional[frozenset[int]] = None
    deferred_if_missing: bool = False
    slices: List["_ReadSlice"] = field(default_factory=list)
    deferred: bool = False


@dataclass(frozen=True)
class _ReadPlan:
    columns: Tuple[_ReadColumn, ...]
    tokens: Tuple[int, ...]
    by_token: Dict[int, _ReadColumn]


@dataclass
class _ReadSlice:
    """Fixed group layout plus binding-owned device indices and staging."""

    generation: int
    ordinal: int
    source_size: int
    count: int
    prim_rows_host: Tuple[int, ...]
    query_rows_host: Tuple[int, ...]
    source_rows: Any
    destination_rows: Any
    staging: Any


@dataclass(frozen=True)
class _PendingRead:
    """Borrowed group data retained until its device work completes."""

    column: _ReadColumn
    layout: _ReadSlice
    source: Any
    source_device: Any
    source_rows: Any
    destination_rows: Any
    cuda_stream: int
    cuda_wait_event: int


def _wait_for_cuda_sync(item: _PendingRead, stream: Any, keepalive: List[Any]) -> None:
    if not item.cuda_stream and not item.cuda_wait_event:
        return
    if not item.source_device.is_cuda:
        raise OvstageContractError(
            f"runtime attribute {item.column.attribute!r} has CUDA synchronization metadata on CPU data"
        )
    if item.cuda_stream:
        producer_stream = wp.Stream(
            item.source_device,
            cuda_stream=None if item.cuda_stream == 1 else item.cuda_stream,
        )
        keepalive.append(producer_stream)
        if stream is not None:
            stream.wait_stream(producer_stream)
        else:
            boundary = producer_stream.record_event()
            keepalive.append(boundary)
            wp.synchronize_event(boundary)
    if item.cuda_wait_event:
        event = wp.Event(device=item.source_device, cuda_event=item.cuda_wait_event)
        keepalive.append(event)
        if stream is not None:
            stream.wait_event(event)
        else:
            wp.synchronize_event(event)


def _build_runtime_transport(model: Any, body_paths: Sequence[str], joint_labels: Sequence[str]) -> _RuntimeTransport:
    """Resolve immutable runtime mappings and allocate per-binding device buffers."""
    import newton  # noqa: PLC0415

    if not joint_labels or model.joint_count == 0:
        runtime = _RuntimeTransport(model, (), (), (), ())
        return _finalize_runtime_transport(runtime, body_paths)

    joint_type = model.joint_type.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    target_q_start = model.joint_target_q_start.numpy()
    target_mode = model.joint_target_mode.numpy()
    supported = {
        int(newton.JointType.REVOLUTE): ("angular", True),
        int(newton.JointType.PRISMATIC): ("linear", False),
    }
    body_derived = {
        int(newton.JointType.BALL),
        int(newton.JointType.FREE),
        int(newton.JointType.DISTANCE),
    }
    articulation_by_joint = {}
    if model.articulation_count:
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        for articulation in range(model.articulation_count):
            start = int(articulation_start[articulation])
            end = int(articulation_end[articulation])
            articulation_by_joint.update((joint, articulation) for joint in range(start, end))
    groups: Dict[str, _PendingJointChannel] = {}
    derived_articulations = set()
    derived_q_indices: List[int] = []
    derived_qd_indices: List[int] = []

    for joint, label in enumerate(joint_labels):
        joint_kind = int(joint_type[joint])
        articulation = articulation_by_joint.get(joint)
        if joint_kind in body_derived and articulation is not None:
            derived_articulations.add(articulation)
            derived_q_indices.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
            derived_qd_indices.extend(range(int(qd_start[joint]), int(qd_start[joint + 1])))
        spec = supported.get(joint_kind)
        if spec is None:
            continue

        instance, rotational = spec
        mode = int(target_mode[int(qd_start[joint])])
        group = groups.setdefault(instance, _PendingJointChannel(rotational))
        group.add(
            label,
            int(q_start[joint]),
            int(qd_start[joint]),
            int(target_q_start[joint]),
            position_target=mode
            in (int(newton.JointTargetMode.POSITION), int(newton.JointTargetMode.POSITION_VELOCITY)),
            velocity_target=mode
            in (int(newton.JointTargetMode.VELOCITY), int(newton.JointTargetMode.POSITION_VELOCITY)),
        )

    channels = tuple(
        pending.finish(instance, model.device) for instance, pending in sorted(groups.items())
    )
    runtime = _RuntimeTransport(
        model,
        channels,
        tuple(sorted(derived_articulations)),
        tuple(derived_q_indices),
        tuple(derived_qd_indices),
    )
    return _finalize_runtime_transport(runtime, body_paths)


def _derive_joint_state(runtime: _RuntimeTransport, state: Any, joint_q: Any, joint_qd: Any) -> None:
    """Update only coordinates whose canonical ovstage state is on bodies."""
    if not runtime.derived_articulation_indices_host:
        return
    import newton  # noqa: PLC0415

    newton.eval_ik(
        runtime.model,
        state,
        runtime.ik_joint_q,
        runtime.ik_joint_qd,
        indices=runtime.derived_articulation_indices,
    )
    wp.launch(
        _scatter_scalar_kernel,
        dim=len(runtime.derived_q_indices_host),
        inputs=[
            runtime.ik_joint_q,
            runtime.derived_q_indices,
            runtime.derived_q_indices,
            wp.float32(1.0),
            joint_q,
        ],
        device=runtime.device,
    )
    wp.launch(
        _scatter_scalar_kernel,
        dim=len(runtime.derived_qd_indices_host),
        inputs=[
            runtime.ik_joint_qd,
            runtime.derived_qd_indices,
            runtime.derived_qd_indices,
            wp.float32(1.0),
            joint_qd,
        ],
        device=runtime.device,
    )


def _finalize_runtime_transport(runtime: _RuntimeTransport, body_paths: Sequence[str]) -> _RuntimeTransport:
    runtime.paths, runtime.channel_rows = _combined_runtime_paths(body_paths, runtime)
    runtime.paths = tuple(runtime.paths)
    return runtime


def _capture_body_scales(
    stage: Any,
    pd: Any,
    runtime: _RuntimeTransport,
    body_paths: Sequence[str],
    ordinal: int,
) -> None:
    """Keep the initial signed world scale in one reusable Warp buffer."""
    from . import _build

    if not body_paths:
        return
    with _stage._path_list_query(stage, pd, body_paths) as query:
        rows = _stage.read_fixed(stage, pd, query, WORLD_MATRIX, ordinal)
    missing = [body_paths[i] for i in range(len(body_paths)) if i not in rows]
    if missing:
        raise OvstageContractError(f"rigid body has no world transform: {missing[0]}")
    matrices = np.stack(
        [np.asarray(rows[i], dtype=np.float64).reshape(4, 4) for i in range(len(body_paths))]
    )
    _, _, scales = _build._decode_pose(matrices.reshape(-1, 16))
    runtime.body_scale = wp.array(
        scales.astype(np.float32), dtype=wp.vec3, device=runtime.device
    )


@wp.kernel
def _encode_body_for_ovstage_kernel(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_scale: wp.array(dtype=wp.vec3),
    pose_out: wp.array(dtype=wp.float64),
    linear_out: wp.array(dtype=wp.float32),
    angular_out: wp.array(dtype=wp.float32),
) -> None:
    """Encode the canonical ovstage body columns."""
    i = wp.tid()
    t = body_q[i]
    p = wp.transform_get_translation(t)
    qf = wp.normalize(wp.transform_get_rotation(t))
    scale = body_scale[i]

    # Cast to float64 for bit-exact match with the CPU encoder.
    x = wp.float64(qf[0])
    y = wp.float64(qf[1])
    z = wp.float64(qf[2])
    w = wp.float64(qf[3])

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    one = wp.float64(1.0)
    two = wp.float64(2.0)
    zero = wp.float64(0.0)

    rcv00 = one - two * (yy + zz)
    rcv01 = two * (xy - wz)
    rcv02 = two * (xz + wy)
    rcv10 = two * (xy + wz)
    rcv11 = one - two * (xx + zz)
    rcv12 = two * (yz - wx)
    rcv20 = two * (xz - wy)
    rcv21 = two * (yz + wx)
    rcv22 = one - two * (xx + yy)

    # GfMatrix4d uses row-vector transforms, so store transpose(rcv).
    matrix = i * 16
    pose_out[matrix + 0] = wp.float64(scale[0]) * rcv00
    pose_out[matrix + 1] = wp.float64(scale[0]) * rcv10
    pose_out[matrix + 2] = wp.float64(scale[0]) * rcv20
    pose_out[matrix + 3] = zero
    pose_out[matrix + 4] = wp.float64(scale[1]) * rcv01
    pose_out[matrix + 5] = wp.float64(scale[1]) * rcv11
    pose_out[matrix + 6] = wp.float64(scale[1]) * rcv21
    pose_out[matrix + 7] = zero
    pose_out[matrix + 8] = wp.float64(scale[2]) * rcv02
    pose_out[matrix + 9] = wp.float64(scale[2]) * rcv12
    pose_out[matrix + 10] = wp.float64(scale[2]) * rcv22
    pose_out[matrix + 11] = zero
    pose_out[matrix + 12] = wp.float64(p[0])
    pose_out[matrix + 13] = wp.float64(p[1])
    pose_out[matrix + 14] = wp.float64(p[2])
    pose_out[matrix + 15] = one

    qd = body_qd[i]
    vector = i * 3
    linear_out[vector + 0] = qd[0]
    linear_out[vector + 1] = qd[1]
    linear_out[vector + 2] = qd[2]
    deg_per_rad = wp.float32(57.29577951308232)
    angular_out[vector + 0] = qd[3] * deg_per_rad
    angular_out[vector + 1] = qd[4] * deg_per_rad
    angular_out[vector + 2] = qd[5] * deg_per_rad


@wp.kernel
def _decode_pose_kernel(
    source: wp.array(dtype=wp.float64),
    source_rows: wp.array(dtype=wp.int32),
    body_rows: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
) -> None:
    """Scatter row-vector GfMatrix4d values into Newton body transforms."""
    i = wp.tid()
    src = source_rows[i] * 16
    dst = body_rows[i]

    # GfMatrix4d stores a row-vector S*R transform.  Normalize its rows to
    # remove scale, then transpose into Newton's column-vector rotation.
    r00 = wp.float32(source[src + 0]); r01 = wp.float32(source[src + 1]); r02 = wp.float32(source[src + 2])
    r10 = wp.float32(source[src + 4]); r11 = wp.float32(source[src + 5]); r12 = wp.float32(source[src + 6])
    r20 = wp.float32(source[src + 8]); r21 = wp.float32(source[src + 9]); r22 = wp.float32(source[src + 10])
    s0 = wp.sqrt(r00 * r00 + r01 * r01 + r02 * r02)
    s1 = wp.sqrt(r10 * r10 + r11 * r11 + r12 * r12)
    s2 = wp.sqrt(r20 * r20 + r21 * r21 + r22 * r22)
    if s0 < 1.0e-12:
        s0 = 1.0
    if s1 < 1.0e-12:
        s1 = 1.0
    if s2 < 1.0e-12:
        s2 = 1.0
    rotation = wp.mat33f(
        r00 / s0, r10 / s1, r20 / s2,
        r01 / s0, r11 / s1, r21 / s2,
        r02 / s0, r12 / s1, r22 / s2,
    )
    q = wp.normalize(wp.quat_from_matrix(rotation))
    p = wp.vec3(
        wp.float32(source[src + 12]),
        wp.float32(source[src + 13]),
        wp.float32(source[src + 14]),
    )
    body_q[dst] = wp.transform(p, q)


@wp.kernel
def _scatter_vec3_kernel(
    source: wp.array(dtype=wp.float32),
    source_rows: wp.array(dtype=wp.int32),
    body_rows: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.vec3),
) -> None:
    i = wp.tid()
    src = source_rows[i] * 3
    destination[body_rows[i]] = wp.vec3(source[src], source[src + 1], source[src + 2])


@wp.kernel
def _decode_velocity_kernel(
    body_rows: wp.array(dtype=wp.int32),
    linear: wp.array(dtype=wp.vec3),
    angular_degrees: wp.array(dtype=wp.vec3),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
) -> None:
    i = wp.tid()
    body = body_rows[i]
    omega = angular_degrees[body] / wp.float32(57.29577951308232)
    body_qd[body] = wp.spatial_vector(
        linear[body][0], linear[body][1], linear[body][2], omega[0], omega[1], omega[2]
    )


@wp.kernel
def _encode_joint_state_for_ovstage_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    q_indices: wp.array(dtype=wp.int32),
    qd_indices: wp.array(dtype=wp.int32),
    scale: wp.float32,
    position_out: wp.array(dtype=wp.float32),
    velocity_out: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()
    position_out[i] = joint_q[q_indices[i]] * scale
    velocity_out[i] = joint_qd[qd_indices[i]] * scale


@wp.kernel
def _gather_scalar_output_kernel(
    source: wp.array(dtype=wp.float32),
    source_indices: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()
    destination[i] = source[source_indices[i]]


@wp.kernel
def _gather_axis_output_kernel(
    source: wp.array(dtype=wp.float32),
    source_indices: wp.array(dtype=wp.int32),
    angular_mask: wp.array(dtype=wp.bool),
    body_order_sign: wp.array(dtype=wp.float32),
    angular_scale: wp.float32,
    apply_body_order_sign: wp.bool,
    destination: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()
    value = source[source_indices[i]]
    if angular_mask[i]:
        value *= angular_scale
    if apply_body_order_sign:
        value *= body_order_sign[i]
    destination[i] = value


@wp.kernel
def _gather_axis_limit_output_kernel(
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    source_indices: wp.array(dtype=wp.int32),
    angular_mask: wp.array(dtype=wp.bool),
    body_order_sign: wp.array(dtype=wp.float32),
    angular_scale: wp.float32,
    destination: wp.array(dtype=wp.vec2),
) -> None:
    i = wp.tid()
    source = source_indices[i]
    scale = angular_scale if angular_mask[i] else wp.float32(1.0)
    low = lower[source]
    high = upper[source]
    if body_order_sign[i] < 0.0:
        low = -upper[source]
        high = -lower[source]
    low = -wp.float32(_FLOAT32_MAX) if low <= -wp.float32(_NEWTON_LIMIT_MAX) else low * scale
    high = wp.float32(_FLOAT32_MAX) if high >= wp.float32(_NEWTON_LIMIT_MAX) else high * scale
    destination[i] = wp.vec2(low, high)


@wp.kernel
def _gather_padded_shape_output_kernel(
    source: wp.array(dtype=wp.float32),
    source_indices: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()
    source_index = source_indices[i]
    if source_index < 0:
        destination[i] = 0.0
    else:
        destination[i] = source[source_index]


@wp.kernel
def _extract_transform_position_kernel(
    source: wp.array(dtype=wp.transformf),
    destination: wp.array(dtype=wp.vec3),
) -> None:
    i = wp.tid()
    destination[i] = wp.transform_get_translation(source[i])


@wp.kernel
def _extract_transform_position_indexed_kernel(
    source: wp.array(dtype=wp.transformf),
    source_indices: wp.array(dtype=wp.uint32),
    destination: wp.array(dtype=wp.vec3),
) -> None:
    i = wp.tid()
    destination[i] = wp.transform_get_translation(source[source_indices[i]])


@wp.kernel
def _extract_transform_orientation_kernel(
    source: wp.array(dtype=wp.transformf),
    destination: wp.array(dtype=wp.quatf),
) -> None:
    i = wp.tid()
    destination[i] = wp.transform_get_rotation(source[i])


@wp.kernel
def _extract_transform_orientation_indexed_kernel(
    source: wp.array(dtype=wp.transformf),
    source_indices: wp.array(dtype=wp.uint32),
    destination: wp.array(dtype=wp.quatf),
) -> None:
    i = wp.tid()
    destination[i] = wp.transform_get_rotation(source[source_indices[i]])


@wp.kernel
def _extract_spatial_linear_kernel(
    source: wp.array(dtype=wp.spatial_vectorf),
    destination: wp.array(dtype=wp.vec3),
) -> None:
    i = wp.tid()
    value = source[i]
    destination[i] = wp.vec3(value[0], value[1], value[2])


@wp.kernel
def _extract_spatial_linear_indexed_kernel(
    source: wp.array(dtype=wp.spatial_vectorf),
    source_indices: wp.array(dtype=wp.uint32),
    destination: wp.array(dtype=wp.vec3),
) -> None:
    i = wp.tid()
    value = source[source_indices[i]]
    destination[i] = wp.vec3(value[0], value[1], value[2])


@wp.kernel
def _extract_spatial_angular_kernel(
    source: wp.array(dtype=wp.spatial_vectorf),
    destination: wp.array(dtype=wp.vec3),
) -> None:
    i = wp.tid()
    value = source[i]
    destination[i] = wp.vec3(value[3], value[4], value[5])


@wp.kernel
def _extract_spatial_angular_indexed_kernel(
    source: wp.array(dtype=wp.spatial_vectorf),
    source_indices: wp.array(dtype=wp.uint32),
    destination: wp.array(dtype=wp.vec3),
) -> None:
    i = wp.tid()
    value = source[source_indices[i]]
    destination[i] = wp.vec3(value[3], value[4], value[5])


@wp.kernel
def _scatter_scalar_kernel(
    source: wp.array(dtype=wp.float32),
    source_rows: wp.array(dtype=wp.int32),
    destination_rows: wp.array(dtype=wp.int32),
    scale: wp.float32,
    destination: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()
    destination[destination_rows[i]] = source[source_rows[i]] * scale


def _device_index_tensor(rows: Optional[Tuple[int, ...]], device: Any) -> Any:
    if rows is None:
        return None
    return make_dltensor(wp.array(rows, dtype=wp.uint32, device=device))


def _native_output_dltensor(
    source: Any,
    count: int,
    lanes: int,
    *,
    code: int = DLDataTypeCode.kDLFloat,
    bits: int = 32,
) -> DLTensor:
    """Describe a borrowed Read API buffer without invoking its DLPack producer.

    ``make_dltensor`` calls the producer's ``__dlpack__`` method. Warp performs
    stream negotiation there, which is not safe while the caller is capturing a
    CUDA graph. Read API buffers therefore retain this small descriptor adapter;
    publication and reusable index buffers use ``make_dltensor``.
    """
    is_cuda = bool(getattr(source.device, "is_cuda", False))
    device_type = DLDeviceType.kDLCUDA if is_cuda else DLDeviceType.kDLCPU
    device_id = int(getattr(source.device, "ordinal", 0) or 0) if is_cuda else 0

    tensor = DLTensor()
    tensor.data = ctypes.c_void_p(int(source.ptr))
    tensor.device = DLDevice(device_type, device_id)
    tensor.ndim = 1
    tensor._shape_storage = (ctypes.c_int64 * 1)(count)
    tensor.shape = ctypes.cast(tensor._shape_storage, ctypes.POINTER(ctypes.c_int64))
    tensor.strides = None
    tensor.byte_offset = 0
    tensor.dtype = DLDataType(code=code, bits=bits, lanes=lanes)
    tensor._source = source
    return tensor


@dataclass(frozen=True)
class _JointOutputGroup:
    width: int
    offset: int
    source_row_count: int
    prim_count: int
    prim_list: int
    data_indices_host: Optional[Tuple[int, ...]]
    data_index_tensor: Any


@dataclass(frozen=True)
class _JointOutputSelection:
    source_indices: Any
    groups: Tuple[_JointOutputGroup, ...]
    angular_mask: Any = None
    body_order_sign: Any = None


@dataclass(frozen=True)
class _OutputCatalog:
    """Static path and coordinate metadata for output queries."""

    body_by_path: Mapping[str, int]
    joint_by_path: Mapping[str, int]
    scene_paths: Tuple[str, ...]
    scene_gravity_row: Optional[int]
    articulation_by_path: Mapping[str, int]
    shapes_by_body: Tuple[Tuple[int, ...], ...]
    joint_q_start: Tuple[int, ...]
    joint_qd_start: Tuple[int, ...]
    joint_target_q_start: Tuple[int, ...]
    joint_dof_dim: Tuple[Tuple[int, int], ...]
    joint_is_articulation: Tuple[bool, ...]
    joint_position_supported: Tuple[bool, ...]
    joint_body_order_sign: Tuple[float, ...]
    body_is_articulation_link: Tuple[bool, ...]
    all_paths: Tuple[str, ...]
    names_by_token: Mapping[int, str]


def _query_paths(binding: Any, stage_query: Any) -> List[str]:
    """Materialize one ovstage query while it is still owned by its caller."""
    stage_query.wait()
    result = stage_query.result()
    if not result.total_prim_count:
        return []
    ordinal = _stage._resolve_read_ceiling(binding._stage, None)
    return _stage._query_paths(
        binding._stage,
        binding._pd,
        stage_query,
        ordinal,
        expected_count=result.total_prim_count,
    )


def _publication_write(
    runtime: _RuntimeTransport,
    pd: Any,
    attribute: str,
    output: Any,
    count: int,
    lanes: int,
    *,
    code: int = DLDataTypeCode.kDLFloat,
    bits: int = 32,
    semantic: int = AttributeSemantic.NONE,
) -> WriteDesc:
    event = (
        int(runtime.producer_event.cuda_event)
        if output.device.is_cuda and runtime.producer_event is not None
        else None
    )
    producer = output if lanes == 1 else output.reshape((count, lanes))
    return WriteDesc(
        attribute=runtime.token(pd, attribute),
        tensors=make_dltensor(
            producer,
            dtype=DLDataType(code=code, bits=bits, lanes=lanes),
            shape=[count],
            ndim=1,
        ),
        is_array=False,
        semantic=semantic,
        cuda_event=event,
    )


def _create_publication_batches(binding: Any) -> Tuple[Tuple[int, Tuple[WriteDesc, ...]], ...]:
    runtime = binding._runtime
    pd = binding._pd
    batches = []
    owned_path_lists = []
    try:
        binding._surface_publication = None
        body_paths = list(runtime.model.body_label)
        if body_paths:
            path_list = pd.create_path_list_from_strings(body_paths)
            owned_path_lists.append(path_list)
            batches.append(
                (
                    path_list,
                    (
                        _publication_write(
                            runtime,
                            pd,
                            RESET_XFORM_STACK,
                            runtime.reset_xform_stack_out,
                            len(body_paths),
                            1,
                            code=DLDataTypeCode.kDLBool,
                            bits=8,
                        ),
                        _publication_write(
                            runtime,
                            pd,
                            XFORM,
                            runtime.pose_out,
                            len(body_paths),
                            16,
                            bits=64,
                            semantic=AttributeSemantic.MATRIX,
                        ),
                        _publication_write(
                            runtime,
                            pd,
                            BODY_VELOCITY,
                            runtime.linear_out,
                            len(body_paths),
                            3,
                            semantic=AttributeSemantic.VECTOR,
                        ),
                        _publication_write(
                            runtime,
                            pd,
                            BODY_ANGULAR_VELOCITY,
                            runtime.angular_out,
                            len(body_paths),
                            3,
                            semantic=AttributeSemantic.VECTOR,
                        ),
                    ),
                )
            )
        for channel in runtime.channels:
            path_list = pd.create_path_list_from_strings(channel.paths)
            owned_path_lists.append(path_list)
            batches.append(
                (
                    path_list,
                    (
                        _publication_write(
                            runtime,
                            pd,
                            joint_state(channel.instance, "position"),
                            channel.position_out,
                            len(channel.paths),
                            1,
                        ),
                        _publication_write(
                            runtime,
                            pd,
                            joint_state(channel.instance, "velocity"),
                            channel.velocity_out,
                            len(channel.paths),
                            1,
                        ),
                    ),
                )
            )
        from ._deformables import model_surface_ranges  # noqa: PLC0415

        surfaces = model_surface_ranges(runtime.model)
        if surfaces:
            path_list = pd.create_path_list_from_strings([surface.geometry_path for surface in surfaces])
            owned_path_lists.append(path_list)
            binding._surface_publication = (path_list, surfaces)
        result = tuple(batches)
        _finalize_path_lists(binding, pd, owned_path_lists)
        return result
    except Exception:
        _destroy_path_lists(pd, tuple(owned_path_lists))
        raise


def _joint_output_selection(
    binding: Any,
    selected_joints: Sequence[Tuple[str, int]],
    starts: Sequence[int],
    path_lists: List[int],
) -> Optional[_JointOutputSelection]:
    by_width: Dict[int, List[Tuple[str, int]]] = {}
    for path, joint in selected_joints:
        start = int(starts[joint])
        width = int(starts[joint + 1]) - start
        if width:
            by_width.setdefault(width, []).append((path, start))
    if not by_width:
        return None

    if len(by_width) == 1:
        width, entries = next(iter(by_width.items()))
        source_start = min(start for _, start in entries)
        if all((start - source_start) % width == 0 for _, start in entries):
            paths = [path for path, _ in entries]
            rows = tuple((start - source_start) // width for _, start in entries)
            data_indices_host = rows if rows != tuple(range(len(rows))) else None
            prim_list = binding._pd.create_path_list_from_strings(paths)
            path_lists.append(prim_list)
            return _JointOutputSelection(
                source_indices=None,
                groups=(
                    _JointOutputGroup(
                        width=width,
                        offset=source_start,
                        source_row_count=max(rows) + 1,
                        prim_count=len(paths),
                        prim_list=prim_list,
                        data_indices_host=data_indices_host,
                        data_index_tensor=_device_index_tensor(data_indices_host, binding._runtime.device),
                    ),
                ),
            )

    indices = []
    groups = []
    for width, entries in sorted(by_width.items()):
        offset = len(indices)
        paths = [path for path, _ in entries]
        for _, start in entries:
            indices.extend(range(start, start + width))
        prim_list = binding._pd.create_path_list_from_strings(paths)
        path_lists.append(prim_list)
        groups.append(
            _JointOutputGroup(
                width=width,
                offset=offset,
                source_row_count=len(entries),
                prim_count=len(paths),
                prim_list=prim_list,
                data_indices_host=None,
                data_index_tensor=None,
            )
        )
    return _JointOutputSelection(
        source_indices=wp.array(indices, dtype=wp.int32, device=binding._runtime.device),
        groups=tuple(groups),
    )


def _semantic_joint_output_selection(
    binding: Any,
    selected_joints: Sequence[Tuple[str, int]],
    starts: Sequence[int],
    eligible: Sequence[bool],
    catalog: _OutputCatalog,
    path_lists: List[int],
) -> Optional[_JointOutputSelection]:
    by_width: Dict[int, List[Tuple[str, int, int, int]]] = {}
    for path, joint in selected_joints:
        if not eligible[joint]:
            continue
        start = int(starts[joint])
        width = int(starts[joint + 1]) - start
        linear_width, angular_width = catalog.joint_dof_dim[joint]
        if width != linear_width + angular_width:
            continue
        by_width.setdefault(width, []).append((path, start, linear_width, joint))
    if not by_width:
        return None

    indices = []
    angular_mask = []
    body_order_sign = []
    groups = []
    for width, entries in sorted(by_width.items()):
        offset = len(indices)
        paths = [path for path, _, _, _ in entries]
        for _, start, linear_width, joint in entries:
            indices.extend(range(start, start + width))
            angular_mask.extend((False,) * linear_width)
            angular_mask.extend((True,) * (width - linear_width))
            body_order_sign.extend((catalog.joint_body_order_sign[joint],) * width)
        prim_list = binding._pd.create_path_list_from_strings(paths)
        path_lists.append(prim_list)
        groups.append(
            _JointOutputGroup(
                width=width,
                offset=offset,
                source_row_count=len(entries),
                prim_count=len(paths),
                prim_list=prim_list,
                data_indices_host=None,
                data_index_tensor=None,
            )
        )
    return _JointOutputSelection(
        source_indices=wp.array(indices, dtype=wp.int32, device=binding._runtime.device),
        groups=tuple(groups),
        angular_mask=wp.array(angular_mask, dtype=wp.bool, device=binding._runtime.device),
        body_order_sign=wp.array(body_order_sign, dtype=wp.float32, device=binding._runtime.device),
    )


def create_query(
    binding: Any,
    stage_query: Any = None,
    *,
    paths: Optional[Sequence[str]] = None,
    object_type: Optional[SimObjectType] = None,
    scope: ObjectScope = ObjectScope.ALL,
) -> Query:
    """Compile an ovstage query or explicit paths into immutable Newton indices."""
    if object_type is not None and not isinstance(object_type, SimObjectType):
        raise TypeError("object_type must be a SimObjectType")
    if not isinstance(scope, ObjectScope):
        raise TypeError("scope must be an ObjectScope")
    if scope is not ObjectScope.ALL:
        raise NotImplementedError("ObjectScope.ACTIVE is not supported by ovnewton")
    if stage_query is not None and paths is not None:
        raise TypeError("pass at most one of stage_query or paths")
    if stage_query is None and paths is None and object_type is None:
        raise TypeError("pass stage_query, paths, or object_type")
    strict = paths is not None
    if strict:
        if isinstance(paths, (str, bytes)):
            raise TypeError("paths must be a sequence of absolute prim paths")
        selected_paths = list(paths)
        if any(not isinstance(path, str) or not path.startswith("/") for path in selected_paths):
            raise ValueError("query paths must be absolute prim paths")
        if len(selected_paths) != len(set(selected_paths)):
            raise ValueError("query paths contain duplicates")
    elif stage_query is not None:
        selected_paths = _query_paths(binding, stage_query)
    else:
        selected_paths = list(binding._output_catalog.all_paths)

    runtime = binding._runtime
    catalog = binding._output_catalog

    body_paths = []
    body_indices = []
    scene_paths = []
    articulation_paths = []
    articulation_indices = []
    selected_joints = []
    matched_path_count = 0
    for path in selected_paths:
        matched = False
        body = catalog.body_by_path.get(path)
        body_matches = body is not None and (
            object_type is None
            or (
                object_type is SimObjectType.RIGID_BODY
                and not catalog.body_is_articulation_link[body]
            )
            or (
                object_type is SimObjectType.ARTICULATION_LINK
                and catalog.body_is_articulation_link[body]
            )
        )
        if body_matches:
            body_paths.append(path)
            body_indices.append(body)
            matched = True
        joint = catalog.joint_by_path.get(path)
        joint_matches = joint is not None and (
            object_type is None
            or (
                object_type is SimObjectType.ARTICULATION_JOINT
                and catalog.joint_is_articulation[joint]
            )
        )
        if joint_matches:
            selected_joints.append((path, joint))
            matched = True
        if path in catalog.scene_paths and object_type in (None, SimObjectType.PHYSICS_SCENE):
            scene_paths.append(path)
            matched = True
        articulation = catalog.articulation_by_path.get(path)
        if articulation is not None and object_type in (None, SimObjectType.ARTICULATION):
            articulation_paths.append(path)
            articulation_indices.append(articulation)
            matched = True
        if matched:
            matched_path_count += 1
        elif strict:
            raise ValueError(f"path is not an output-capable bound object: {path}")

    path_lists = []
    try:
        body_index_array = None
        body_index_tensor = None
        body_indices_host = None
        body_prim_list = 0
        if body_paths:
            body_prim_list = binding._pd.create_path_list_from_strings(body_paths)
            path_lists.append(body_prim_list)

        scene_prim_list = 0
        if scene_paths:
            scene_prim_list = binding._pd.create_path_list_from_strings(scene_paths)
            path_lists.append(scene_prim_list)

        articulation_prim_list = 0
        if articulation_paths:
            articulation_prim_list = binding._pd.create_path_list_from_strings(articulation_paths)
            path_lists.append(articulation_prim_list)

        articulation_indices_host = tuple(articulation_indices) or None
        articulation_index_array = (
            wp.array(articulation_indices_host, dtype=wp.uint32, device=runtime.device)
            if articulation_indices_host is not None
            else None
        )

        body_rows = tuple(body_indices)
        if body_rows != tuple(range(len(body_rows))):
            body_indices_host = body_rows
            body_index_array = wp.array(
                body_indices_host,
                dtype=wp.uint32,
                device=runtime.device,
            )
            body_index_tensor = make_dltensor(body_index_array)

        selected_shapes = [catalog.shapes_by_body[body] for body in body_indices]
        shape_width = max((len(shapes) for shapes in selected_shapes), default=0)
        shape_counts = None
        shape_indices = None
        if body_paths:
            shape_counts = wp.array(
                [len(shapes) for shapes in selected_shapes],
                dtype=wp.int32,
                device=runtime.device,
            )
        if shape_width:
            shape_indices = wp.array(
                [
                    shapes[column] if column < len(shapes) else -1
                    for shapes in selected_shapes
                    for column in range(shape_width)
                ],
                dtype=wp.int32,
                device=runtime.device,
            )
        joint_q = _joint_output_selection(
            binding,
            selected_joints,
            catalog.joint_q_start,
            path_lists,
        )
        joint_qd = _joint_output_selection(
            binding,
            selected_joints,
            catalog.joint_qd_start,
            path_lists,
        )
        joint_dof = _joint_output_selection(
            binding,
            [
                (path, joint)
                for path, joint in selected_joints
                if catalog.joint_is_articulation[joint]
            ],
            catalog.joint_qd_start,
            path_lists,
        )
        joint_position = _semantic_joint_output_selection(
            binding,
            selected_joints,
            catalog.joint_q_start,
            catalog.joint_position_supported,
            catalog,
            path_lists,
        )
        joint_velocity = _semantic_joint_output_selection(
            binding,
            selected_joints,
            catalog.joint_qd_start,
            catalog.joint_is_articulation,
            catalog,
            path_lists,
        )
        joint_position_target = _semantic_joint_output_selection(
            binding,
            selected_joints,
            catalog.joint_target_q_start,
            catalog.joint_position_supported,
            catalog,
            path_lists,
        )
        available = []
        if body_paths:
            available.extend(
                name for name, descriptor in _OUTPUT_DESCRIPTORS.items() if descriptor.domain == "body"
            )
            available.append(_SHAPE_COUNT)
            if shape_width:
                available.extend((_FRICTION, _RESTITUTION))
        if scene_paths:
            available.append(_GRAVITY)
        if articulation_paths:
            available.extend(
                name
                for name, descriptor in _OUTPUT_DESCRIPTORS.items()
                if descriptor.domain == "articulation"
            )
        if joint_q is not None:
            available.append(_JOINT_Q)
        if joint_qd is not None:
            available.append(_JOINT_QD)
        if joint_position is not None:
            available.append(_JOINT_POSITION)
        if joint_velocity is not None:
            available.append(_JOINT_VELOCITY)
            available.extend(
                (
                    _JOINT_VELOCITY_TARGET,
                    _JOINT_STIFFNESS,
                    _JOINT_DAMPING,
                    _JOINT_LIMIT,
                    _JOINT_MAX_VELOCITY,
                )
            )
        if joint_position_target is not None:
            available.append(_JOINT_POSITION_TARGET)
        if joint_dof is not None:
            available.extend((_JOINT_MAX_FORCE, _JOINT_ARMATURE, _JOINT_FRICTION))
        attributes = tuple(runtime.token(binding._pd, name) for name in available)
        return Query(
            binding,
            body_index_array=body_index_array,
            body_index_tensor=body_index_tensor,
            body_indices_host=body_indices_host,
            body_prim_list=body_prim_list,
            body_count=len(body_paths),
            scene_prim_list=scene_prim_list,
            scene_count=len(scene_paths),
            shape_counts=shape_counts,
            shape_indices=shape_indices,
            shape_width=shape_width,
            articulation_index_array=articulation_index_array,
            articulation_prim_list=articulation_prim_list,
            articulation_count=len(articulation_paths),
            joint_q=joint_q,
            joint_qd=joint_qd,
            joint_dof=joint_dof,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_position_target=joint_position_target,
            attributes=attributes,
            prim_count=matched_path_count,
            object_type=object_type,
            scope=scope,
            path_lists=tuple(path_lists),
        )
    except Exception:
        _destroy_path_lists(binding._pd, tuple(path_lists))
        raise


def _joint_body_order_signs(
    binding: Any,
    joint_by_path: Mapping[str, int],
    ordinal: int,
) -> Tuple[float, ...]:
    """Map Newton joint directions back to the authored USD body order."""
    model = binding._runtime.model
    signs = [1.0] * model.joint_count
    paths = tuple(joint_by_path)
    if not paths:
        return tuple(signs)

    with _stage._path_list_query(binding._stage, binding._pd, paths) as query:
        columns = _stage.read_columns(
            binding._stage,
            binding._pd,
            query,
            ("physics:body0", "physics:body1"),
            ordinal,
            ragged=("physics:body0", "physics:body1"),
        )

    def target(rows: Mapping[int, Any], row: int) -> Optional[str]:
        values = rows.get(row)
        if values is None or not len(values):
            return None
        if len(values) != 1:
            raise ValueError(f"joint {paths[row]} has more than one body target")
        return binding._pd.path_to_string(int(values[0]))

    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    body_labels = tuple(model.body_label)
    for row, path in enumerate(paths):
        joint = joint_by_path[path]
        parent = int(parents[joint])
        child = int(children[joint])
        parent_path = body_labels[parent] if parent >= 0 else None
        child_path = body_labels[child] if child >= 0 else None
        body0 = target(columns["physics:body0"], row)
        body1 = target(columns["physics:body1"], row)
        if (parent_path, child_path) == (body0, body1):
            signs[joint] = 1.0
        elif (parent_path, child_path) == (body1, body0):
            signs[joint] = -1.0
        else:
            raise ValueError(
                f"joint {path} bodies do not match the connected Newton model"
            )
    return tuple(signs)


def _create_output_catalog(binding: Any, ordinal: int) -> _OutputCatalog:
    import newton

    from . import _parse

    runtime = binding._runtime
    model = runtime.model
    body_by_path = {path: index for index, path in enumerate(model.body_label)}
    q_start = tuple(int(value) for value in model.joint_q_start.numpy())
    qd_start = tuple(int(value) for value in model.joint_qd_start.numpy())
    target_q_start = tuple(int(value) for value in model.joint_target_q_start.numpy())
    dof_dim = tuple(
        (int(linear), int(angular))
        for linear, angular in model.joint_dof_dim.numpy()
    )
    joint_types = tuple(int(value) for value in model.joint_type.numpy())
    joint_articulations = tuple(int(value) for value in model.joint_articulation.numpy())
    joint_children = tuple(int(value) for value in model.joint_child.numpy())
    authored_articulation_roots = set()
    for schema in ("PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"):
        authored_articulation_roots.update(
            _parse._api_paths(binding._stage, binding._pd, ordinal, schema)
        )
    authored_articulations = {
        articulation
        for articulation, label in enumerate(model.articulation_label)
        if label in authored_articulation_roots
    }
    scalar_position_types = {
        int(newton.JointType.PRISMATIC),
        int(newton.JointType.REVOLUTE),
        int(newton.JointType.D6),
    }
    joint_is_articulation = tuple(
        joint_articulations[joint] >= 0
        and joint_types[joint] != int(newton.JointType.FREE)
        for joint in range(model.joint_count)
    )
    # Caller-owned articulations need not use USD root paths as labels. Include
    # their links, but not Newton's generated single-free-joint rigid bodies.
    articulation_joint_counts = Counter(joint_articulations)
    link_articulations = authored_articulations | {
        articulation
        for articulation, joint_type in zip(joint_articulations, joint_types)
        if articulation >= 0
        and (articulation_joint_counts[articulation] > 1 or joint_type != int(newton.JointType.FREE))
    }
    articulation_bodies = {
        joint_children[joint]
        for joint in range(model.joint_count)
        if joint_articulations[joint] in link_articulations
        and joint_children[joint] >= 0
    }
    body_is_articulation_link = tuple(
        body in articulation_bodies for body in range(model.body_count)
    )
    joint_position_supported = tuple(
        joint_is_articulation[joint]
        and joint_types[joint] in scalar_position_types
        and q_start[joint + 1] - q_start[joint] == sum(dof_dim[joint])
        for joint in range(model.joint_count)
    )
    joint_by_path = {
        path: joint
        for joint, path in enumerate(model.joint_label)
        if isinstance(path, str)
        and path.startswith("/")
        and (q_start[joint + 1] > q_start[joint] or qd_start[joint + 1] > qd_start[joint])
        and path not in body_by_path
    }
    joint_body_order_sign = _joint_body_order_signs(
        binding,
        {
            path: joint
            for path, joint in joint_by_path.items()
            if joint_is_articulation[joint]
        },
        ordinal,
    )
    scene_paths = tuple(_parse._type_paths(binding._stage, binding._pd, "PhysicsScene", ordinal)[:1])
    scene_gravity_row = None
    if scene_paths:
        if model.gravity.shape[0] == 1:
            scene_gravity_row = 0
        else:
            # Resolve once at attachment, not through a host read on every read().
            worlds = set(int(world) for world in model.body_world.numpy())
            if len(worlds) == 1:
                world = worlds.pop()
                # Explicit worlds store global (-1) gravity in the final row.
                scene_gravity_row = world if world >= 0 else model.gravity.shape[0] - 1
    articulation_by_path = {}
    if model.articulation_count:
        articulation_start = model.articulation_start.numpy()
        joint_child = model.joint_child.numpy()
        for articulation, label in enumerate(model.articulation_label):
            if articulation in authored_articulations:
                root_joint = int(articulation_start[articulation])
                articulation_by_path[label] = int(joint_child[root_joint])
    shape_body = model.shape_body.numpy()
    shapes_by_body_lists = [[] for _ in range(model.body_count)]
    for shape, body in enumerate(shape_body):
        if int(body) >= 0:
            shapes_by_body_lists[int(body)].append(shape)
    shapes_by_body = tuple(tuple(shapes) for shapes in shapes_by_body_lists)
    names_by_token = {runtime.token(binding._pd, name): name for name in _OUTPUT_ATTRIBUTES}
    return _OutputCatalog(
        body_by_path=MappingProxyType(body_by_path),
        joint_by_path=MappingProxyType(joint_by_path),
        scene_paths=scene_paths,
        scene_gravity_row=scene_gravity_row,
        articulation_by_path=MappingProxyType(articulation_by_path),
        shapes_by_body=shapes_by_body,
        joint_q_start=q_start,
        joint_qd_start=qd_start,
        joint_target_q_start=target_q_start,
        joint_dof_dim=dof_dim,
        joint_is_articulation=joint_is_articulation,
        joint_position_supported=joint_position_supported,
        joint_body_order_sign=joint_body_order_sign,
        body_is_articulation_link=body_is_articulation_link,
        all_paths=tuple(
            dict.fromkeys(
                (*model.body_label, *joint_by_path, *scene_paths, *articulation_by_path)
            )
        ),
        names_by_token=MappingProxyType(names_by_token),
    )


def initialize_output(binding: Any, ordinal: int) -> None:
    binding._output_catalog = _create_output_catalog(binding, ordinal)
    binding._all_query = create_query(binding, paths=binding._output_catalog.all_paths)
    binding._publication_batches = _create_publication_batches(binding)


def _normalize_output_attributes(binding: Any, attributes: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(attributes, (str, bytes)):
        raise TypeError("attributes must be a sequence of output names or tokens")
    normalized = []
    seen = set()
    for attribute in attributes:
        if isinstance(attribute, str):
            name = attribute
        elif isinstance(attribute, Integral) and not isinstance(attribute, bool):
            token = int(attribute)
            name = binding._output_catalog.names_by_token.get(token)
            if name is None:
                raise ValueError(f"attribute token {token} is not a supported output attribute")
        else:
            raise TypeError("output attributes must be strings or interned integer tokens")
        if name not in _OUTPUT_ATTRIBUTES:
            raise ValueError(f"unsupported output attribute {name!r}")
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    return tuple(normalized)


def _native_output_group(
    storage: Any,
    *,
    attribute: int,
    prim_list: int,
    prim_count: int,
    source: Any,
    source_row_count: int,
    width: int,
    code: int = DLDataTypeCode.kDLFloat,
    bits: int = 32,
    data_indices_host: Optional[Tuple[int, ...]] = None,
    data_index_tensor: Any = None,
) -> ReadGroup:
    return ReadGroup(
        storage,
        attribute=attribute,
        prim_list=prim_list,
        prim_count=prim_count,
        tensors=(
            _native_output_dltensor(source, source_row_count, width, code=code, bits=bits),
        ),
        is_array=False,
        data_indices=data_indices_host,
        data_index_tensor=data_index_tensor,
    )


def _joint_output_groups(
    storage: Any,
    attribute: int,
    source: Any,
    selection: _JointOutputSelection,
    *,
    lane_multiplier: int = 1,
) -> List[ReadGroup]:
    groups = []
    for layout in selection.groups:
        start = layout.offset
        end = start + layout.source_row_count * layout.width
        groups.append(
            _native_output_group(
                storage,
                attribute=attribute,
                prim_list=layout.prim_list,
                prim_count=layout.prim_count,
                source=source[start:end],
                source_row_count=layout.source_row_count,
                width=layout.width * lane_multiplier,
                data_indices_host=layout.data_indices_host,
                data_index_tensor=layout.data_index_tensor,
            )
        )
    return groups


def _prepare_joint_output(
    source: Any,
    selection: _JointOutputSelection,
    device: Any,
    stream: Any,
    *,
    angular_scale: Optional[float] = None,
    apply_body_order_sign: bool = False,
) -> Any:
    if selection.source_indices is None and angular_scale is None:
        return source
    output = wp.empty(selection.source_indices.shape[0], dtype=wp.float32, device=device)
    if angular_scale is None:
        wp.launch(
            _gather_scalar_output_kernel,
            dim=output.shape[0],
            inputs=[source, selection.source_indices, output],
            device=device,
            stream=stream,
        )
    else:
        wp.launch(
            _gather_axis_output_kernel,
            dim=output.shape[0],
            inputs=[
                source,
                selection.source_indices,
                selection.angular_mask,
                selection.body_order_sign,
                angular_scale,
                apply_body_order_sign,
                output,
            ],
            device=device,
            stream=stream,
        )
    return output


def _prepare_joint_limit_output(
    lower: Any,
    upper: Any,
    selection: _JointOutputSelection,
    device: Any,
    stream: Any,
) -> Any:
    output = wp.empty(selection.source_indices.shape[0], dtype=wp.vec2, device=device)
    wp.launch(
        _gather_axis_limit_output_kernel,
        dim=output.shape[0],
        inputs=[
            lower,
            upper,
            selection.source_indices,
            selection.angular_mask,
            selection.body_order_sign,
            _DEG_PER_RAD,
            output,
        ],
        device=device,
        stream=stream,
    )
    return output


def _prepare_body_output(
    source: Any,
    descriptor: _OutputDescriptor,
    count: int,
    source_indices: Any,
    device: Any,
    stream: Any,
) -> Any:
    if descriptor.conversion is None:
        return source
    if descriptor.conversion == "transform_orientation":
        output = wp.empty(count, dtype=wp.quatf, device=device)
        kernel = (
            _extract_transform_orientation_kernel
            if source_indices is None
            else _extract_transform_orientation_indexed_kernel
        )
    else:
        output = wp.empty(count, dtype=wp.vec3, device=device)
        kernels = {
            "transform_position": (
                _extract_transform_position_kernel,
                _extract_transform_position_indexed_kernel,
            ),
            "spatial_linear": (
                _extract_spatial_linear_kernel,
                _extract_spatial_linear_indexed_kernel,
            ),
            "spatial_angular": (
                _extract_spatial_angular_kernel,
                _extract_spatial_angular_indexed_kernel,
            ),
        }
        kernel = kernels[descriptor.conversion][source_indices is not None]
    wp.launch(
        kernel,
        dim=count,
        inputs=[source, output] if source_indices is None else [source, source_indices, output],
        device=device,
        stream=stream,
    )
    return output


def _prepare_shape_output(source: Any, query: Query, device: Any, stream: Any) -> Any:
    output = wp.empty(query._body_count * query._shape_width, dtype=wp.float32, device=device)
    wp.launch(
        _gather_padded_shape_output_kernel,
        dim=output.shape[0],
        inputs=[source, query._shape_indices, output],
        device=device,
        stream=stream,
    )
    return output


def _read_native_output(
    binding: Any,
    state: Any,
    control: Any,
    query: Query,
    names: Sequence[str],
) -> ReadResult:
    runtime = binding._runtime
    device = runtime.device
    state_attributes_requested = any(
        _OUTPUT_DESCRIPTORS[name].owner == "state"
        and (
            (_OUTPUT_DESCRIPTORS[name].domain == "body" and query._body_count)
            or (
                _OUTPUT_DESCRIPTORS[name].domain == "articulation"
                and query._articulation_count
            )
            or (name == _JOINT_Q and query._joint_q is not None)
            or (name == _JOINT_QD and query._joint_qd is not None)
            or (name == _JOINT_POSITION and query._joint_position is not None)
            or (name == _JOINT_VELOCITY and query._joint_velocity is not None)
        )
        for name in names
    )
    if state is None and state_attributes_requested:
        raise ValueError("state is required for the requested state output attributes")
    control_attributes_requested = any(
        _OUTPUT_DESCRIPTORS[name].owner == "control"
        and (
            (name == _JOINT_POSITION_TARGET and query._joint_position_target is not None)
            or (name == _JOINT_VELOCITY_TARGET and query._joint_velocity is not None)
        )
        for name in names
    )
    if control is None and control_attributes_requested:
        raise ValueError("control is required for the requested control output attributes")

    body_sources = {}
    if query._body_count:
        for name in names:
            descriptor = _OUTPUT_DESCRIPTORS[name]
            if descriptor.domain != "body":
                continue
            owner = state if descriptor.owner == "state" else runtime.model
            body_sources[name] = _validate_array(
                owner,
                descriptor.source,
                runtime.model.body_count,
                device,
            )

    articulation_sources = {}
    if query._articulation_count:
        for name in names:
            descriptor = _OUTPUT_DESCRIPTORS[name]
            if descriptor.domain != "articulation":
                continue
            articulation_sources[name] = _validate_array(
                state,
                descriptor.source,
                runtime.model.body_count,
                device,
            )

    scene_sources = {}
    if query._scene_count and _GRAVITY in names:
        row = binding._output_catalog.scene_gravity_row
        if row is None:
            raise ValueError("cannot map scene gravity to a single Newton world from the bound bodies")
        source = _validate_array(
            runtime.model,
            "gravity",
            runtime.model.gravity.shape[0],
            device,
        )
        scene_sources[_GRAVITY] = source[row:row + 1]

    shape_sources = {}
    if query._shape_width:
        for name in names:
            descriptor = _OUTPUT_DESCRIPTORS[name]
            if descriptor.domain == "shape" and descriptor.owner == "model":
                # ovstage's DLTensor adapters accept at most 255 lanes per row.
                if query._shape_width > 255:
                    raise ValueError(
                        f"{name} supports at most 255 shapes per selected body; got {query._shape_width}"
                    )
                shape_sources[name] = _validate_array(
                    runtime.model,
                    descriptor.source,
                    runtime.model.shape_count,
                    device,
                )

    joint_q = None
    joint_qd = None
    if query._joint_q is not None and _JOINT_Q in names:
        joint_q = _validate_array(state, "joint_q", int(runtime.model.joint_q.shape[0]), device)
    if query._joint_qd is not None and _JOINT_QD in names:
        joint_qd = _validate_array(state, "joint_qd", int(runtime.model.joint_qd.shape[0]), device)
    joint_position = None
    joint_velocity = None
    if query._joint_position is not None and _JOINT_POSITION in names:
        joint_position = _validate_array(
            state,
            "joint_q",
            int(runtime.model.joint_q.shape[0]),
            device,
        )
    if query._joint_velocity is not None and _JOINT_VELOCITY in names:
        joint_velocity = _validate_array(
            state,
            "joint_qd",
            int(runtime.model.joint_qd.shape[0]),
            device,
        )

    joint_field_selections = {
        _JOINT_POSITION_TARGET: query._joint_position_target,
        _JOINT_VELOCITY_TARGET: query._joint_velocity,
        _JOINT_STIFFNESS: query._joint_velocity,
        _JOINT_DAMPING: query._joint_velocity,
        _JOINT_MAX_VELOCITY: query._joint_velocity,
        _JOINT_MAX_FORCE: query._joint_dof,
        _JOINT_ARMATURE: query._joint_dof,
        _JOINT_FRICTION: query._joint_dof,
    }
    joint_field_sources = {}
    for name in names:
        selection = joint_field_selections.get(name)
        if selection is None:
            continue
        descriptor = _OUTPUT_DESCRIPTORS[name]
        owner = control if descriptor.owner == "control" else runtime.model
        expected = getattr(runtime.model, descriptor.source)
        joint_field_sources[name] = _validate_array(
            owner,
            descriptor.source,
            int(expected.shape[0]),
            device,
        )

    joint_limit_sources = None
    if _JOINT_LIMIT in names and query._joint_velocity is not None:
        joint_limit_sources = (
            _validate_array(
                runtime.model,
                "joint_limit_lower",
                int(runtime.model.joint_limit_lower.shape[0]),
                device,
            ),
            _validate_array(
                runtime.model,
                "joint_limit_upper",
                int(runtime.model.joint_limit_upper.shape[0]),
                device,
            ),
        )

    stream = wp.get_stream(device) if device.is_cuda else None
    body_outputs = {
        name: _prepare_body_output(
            source,
            _OUTPUT_DESCRIPTORS[name],
            query._body_count,
            query._body_index_array,
            device,
            stream,
        )
        for name, source in body_sources.items()
    }
    articulation_outputs = {
        name: _prepare_body_output(
            source,
            _OUTPUT_DESCRIPTORS[name],
            query._articulation_count,
            query._articulation_index_array,
            device,
            stream,
        )
        for name, source in articulation_sources.items()
    }
    shape_outputs = {
        name: _prepare_shape_output(source, query, device, stream)
        for name, source in shape_sources.items()
    }
    joint_q_out = _prepare_joint_output(joint_q, query._joint_q, device, stream) if joint_q is not None else None
    joint_qd_out = _prepare_joint_output(joint_qd, query._joint_qd, device, stream) if joint_qd is not None else None
    joint_position_out = (
        _prepare_joint_output(
            joint_position,
            query._joint_position,
            device,
            stream,
            angular_scale=_DEG_PER_RAD,
            apply_body_order_sign=True,
        )
        if joint_position is not None
        else None
    )
    joint_velocity_out = (
        _prepare_joint_output(
            joint_velocity,
            query._joint_velocity,
            device,
            stream,
            angular_scale=_DEG_PER_RAD,
            apply_body_order_sign=True,
        )
        if joint_velocity is not None
        else None
    )
    joint_field_outputs = {}
    for name, source in joint_field_sources.items():
        conversion = _OUTPUT_DESCRIPTORS[name].conversion
        angular_scale = {
            None: None,
            "angular_degrees": _DEG_PER_RAD,
            "angular_per_degree": 1.0 / _DEG_PER_RAD,
        }[conversion]
        joint_field_outputs[name] = _prepare_joint_output(
            source,
            joint_field_selections[name],
            device,
            stream,
            angular_scale=angular_scale,
            apply_body_order_sign=_OUTPUT_DESCRIPTORS[name].owner == "control",
        )
    joint_limit_out = (
        _prepare_joint_limit_output(
            joint_limit_sources[0],
            joint_limit_sources[1],
            query._joint_velocity,
            device,
            stream,
        )
        if joint_limit_sources is not None
        else None
    )
    event = None
    if stream is not None and (
        body_sources
        or articulation_sources
        or scene_sources
        or shape_outputs
        or (_SHAPE_COUNT in names and query._shape_counts is not None)
        or joint_q_out is not None
        or joint_qd_out is not None
        or joint_position_out is not None
        or joint_velocity_out is not None
        or joint_field_outputs
        or joint_limit_out is not None
    ):
        event = wp.Event(device)
        stream.record_event(event)
    keepalive_pairs = [
        *tuple((body_sources[name], body_outputs[name]) for name in body_sources),
        *tuple(
            (articulation_sources[name], articulation_outputs[name])
            for name in articulation_sources
        ),
        *tuple((shape_sources[name], shape_outputs[name]) for name in shape_sources),
        (joint_q, joint_q_out),
        (joint_qd, joint_qd_out),
        (joint_position, joint_position_out),
        (joint_velocity, joint_velocity_out),
        *tuple(
            (joint_field_sources[name], joint_field_outputs[name])
            for name in joint_field_sources
        ),
    ]
    if joint_limit_sources is not None:
        keepalive_pairs.extend(
            (
                (joint_limit_sources[0], joint_limit_out),
                (joint_limit_sources[1], joint_limit_out),
            )
        )
    storage = _ReadStorage(
        query,
        event=event,
        keepalive=tuple(
            source
            for source, output in keepalive_pairs
            if source is not None and output is not source
        ),
    )
    groups = []
    for name in names:
        descriptor = _OUTPUT_DESCRIPTORS[name]
        if descriptor.domain == "body":
            source = body_outputs.get(name)
            if source is not None:
                derived = descriptor.conversion is not None
                groups.append(
                    _native_output_group(
                        storage,
                        attribute=runtime.token(binding._pd, name),
                        prim_list=query._body_prim_list,
                        prim_count=query._body_count,
                        source=source,
                        source_row_count=(
                            query._body_count if derived else runtime.model.body_count
                        ),
                        width=descriptor.width,
                        data_indices_host=(
                            None if derived else query._body_indices_host
                        ),
                        data_index_tensor=(
                            None if derived else query._body_index_tensor
                        ),
                    )
                )
        elif descriptor.domain == "scene":
            source = scene_sources.get(name)
            if source is not None:
                groups.append(
                    _native_output_group(
                        storage,
                        attribute=runtime.token(binding._pd, name),
                        prim_list=query._scene_prim_list,
                        prim_count=query._scene_count,
                        source=source,
                        source_row_count=source.shape[0],
                        width=descriptor.width,
                    )
                )
        elif descriptor.domain == "articulation":
            source = articulation_outputs.get(name)
            if source is not None:
                groups.append(
                    _native_output_group(
                        storage,
                        attribute=runtime.token(binding._pd, name),
                        prim_list=query._articulation_prim_list,
                        prim_count=query._articulation_count,
                        source=source,
                        source_row_count=query._articulation_count,
                        width=descriptor.width,
                    )
                )
        elif descriptor.domain == "shape":
            if name == _SHAPE_COUNT and query._shape_counts is not None:
                groups.append(
                    _native_output_group(
                        storage,
                        attribute=runtime.token(binding._pd, name),
                        prim_list=query._body_prim_list,
                        prim_count=query._body_count,
                        source=query._shape_counts,
                        source_row_count=query._body_count,
                        width=1,
                        code=DLDataTypeCode.kDLInt,
                    )
                )
            else:
                source = shape_outputs.get(name)
                if source is not None:
                    groups.append(
                        _native_output_group(
                            storage,
                            attribute=runtime.token(binding._pd, name),
                            prim_list=query._body_prim_list,
                            prim_count=query._body_count,
                            source=source,
                            source_row_count=query._body_count,
                            width=query._shape_width,
                        )
                    )
        elif name == _JOINT_Q and joint_q is not None:
            groups.extend(
                _joint_output_groups(
                    storage,
                    runtime.token(binding._pd, name),
                    joint_q_out,
                    query._joint_q,
                )
            )
        elif name == _JOINT_QD and joint_qd is not None:
            groups.extend(
                _joint_output_groups(
                    storage,
                    runtime.token(binding._pd, name),
                    joint_qd_out,
                    query._joint_qd,
                )
            )
        elif name == _JOINT_POSITION and joint_position is not None:
            groups.extend(
                _joint_output_groups(
                    storage,
                    runtime.token(binding._pd, name),
                    joint_position_out,
                    query._joint_position,
                )
            )
        elif name == _JOINT_VELOCITY and joint_velocity is not None:
            groups.extend(
                _joint_output_groups(
                    storage,
                    runtime.token(binding._pd, name),
                    joint_velocity_out,
                    query._joint_velocity,
                )
            )
        elif name in joint_field_outputs:
            groups.extend(
                _joint_output_groups(
                    storage,
                    runtime.token(binding._pd, name),
                    joint_field_outputs[name],
                    joint_field_selections[name],
                )
            )
        elif name == _JOINT_LIMIT and joint_limit_out is not None:
            groups.extend(
                _joint_output_groups(
                    storage,
                    runtime.token(binding._pd, name),
                    joint_limit_out,
                    query._joint_velocity,
                    lane_multiplier=2,
                )
            )
    return ReadResult(groups)


def read_output(
    binding: Any,
    state: Any,
    control: Any,
    *,
    query: Optional[Query],
    attributes: Sequence[Any],
) -> ReadResult:
    """Expose selected Newton state columns through read-only groups."""
    selected_query = binding._all_query if query is None else query
    if not isinstance(selected_query, Query) or selected_query._binding_ref() is not binding:
        raise ValueError("query belongs to a different StageBinding")
    names = _normalize_output_attributes(binding, attributes)
    return _read_native_output(binding, state, control, selected_query, names)


def publish_output(binding: Any, state: Any, ordinal: int) -> None:
    """Encode and publish every canonical ovstage output."""
    with binding._runtime_lock:
        runtime = binding._runtime
        model = runtime.model
        body_q = _validate_array(state, "body_q", model.body_count, runtime.device) if model.body_count else None
        body_qd = _validate_array(state, "body_qd", model.body_count, runtime.device) if model.body_count else None
        joint_q = None
        joint_qd = None
        particle_q = None
        if runtime.channels:
            joint_q = _validate_array(state, "joint_q", int(model.joint_q.shape[0]), runtime.device)
            joint_qd = _validate_array(state, "joint_qd", int(model.joint_qd.shape[0]), runtime.device)
        if binding._surface_publication is not None:
            particle_q = _validate_array(state, "particle_q", model.particle_count, runtime.device)
        stream = wp.get_stream(runtime.device) if runtime.device.is_cuda else None
        if model.body_count:
            wp.launch(
                _encode_body_for_ovstage_kernel,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    runtime.body_scale,
                    runtime.pose_out,
                    runtime.linear_out,
                    runtime.angular_out,
                ],
                device=runtime.device,
                stream=stream,
            )
        for channel in runtime.channels:
            wp.launch(
                _encode_joint_state_for_ovstage_kernel,
                dim=len(channel.paths),
                inputs=[
                    joint_q,
                    joint_qd,
                    channel.q_indices,
                    channel.qd_indices,
                    wp.float32(_DEG_PER_RAD if channel.rotational else 1.0),
                    channel.position_out,
                    channel.velocity_out,
                ],
                device=runtime.device,
                stream=stream,
            )
        if stream is not None:
            stream.record_event(runtime.producer_event)
        operations = []
        with contextlib.ExitStack() as stack:
            try:
                for path_list, writes in binding._publication_batches:
                    query = stack.enter_context(binding._stage.query_from_path_list(path_list))
                    operations.append(binding._stage.write_attributes(query, writes, int(ordinal)))
                if binding._surface_publication is not None:
                    path_list, surfaces = binding._surface_publication
                    tensors = tuple(
                        make_dltensor(
                            particle_q[surface.particle_start : surface.particle_end],
                            dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=32, lanes=3),
                            shape=[surface.particle_end - surface.particle_start],
                            ndim=1,
                        )
                        for surface in surfaces
                    )
                    event = (
                        int(runtime.producer_event.cuda_event)
                        if runtime.device.is_cuda and runtime.producer_event is not None
                        else None
                    )
                    query = stack.enter_context(binding._stage.query_from_path_list(path_list))
                    operations.append(
                        binding._stage.write_attributes(
                            query,
                            (
                                WriteDesc(
                                    attribute=runtime.token(binding._pd, "points"),
                                    tensors=tensors,
                                    is_array=True,
                                    semantic=AttributeSemantic.POINT,
                                    cuda_event=event,
                                ),
                            ),
                            int(ordinal),
                        )
                    )
            except Exception:
                _wait_operations(operations, suppress=True)
                raise
            _wait_operations(operations)


def _tensor_device(tensor: Any):
    """Return the Warp device corresponding to a CPU/CUDA DLTensor."""
    device_type = int(tensor.device.device_type.value)
    if device_type == DLDeviceType.kDLCPU:
        return wp.get_device("cpu")
    if device_type in (DLDeviceType.kDLCUDA, DLDeviceType.kDLCUDAManaged):
        return wp.get_device(f"cuda:{int(tensor.device.device_id)}")
    raise OvstageContractError(f"runtime tensor device type {device_type} is unsupported")


def _tensor_source_view(tensor: Any, *, dtype: Any, lanes: int, attribute: str = "runtime column"):
    """Validate and alias one contiguous ovstage DLTensor as a Warp array."""
    expected_bits = 64 if dtype == wp.float64 else 32
    if (
        int(tensor.dtype.code) != DLDataTypeCode.kDLFloat
        or int(tensor.dtype.bits) != expected_bits
        or int(tensor.dtype.lanes) != lanes
    ):
        raise OvstageContractError(
            f"{attribute!r} has incompatible DLTensor type "
            f"(code={int(tensor.dtype.code)}, bits={int(tensor.dtype.bits)}, lanes={int(tensor.dtype.lanes)}); "
            f"expected float{expected_bits}x{lanes}"
        )
    if int(tensor.ndim) != 1 or not tensor.shape:
        raise OvstageContractError(f"{attribute!r} must be a contiguous one-dimensional DLTensor")
    if tensor.strides:
        raise OvstageContractError(f"{attribute!r} returned a strided DLTensor")

    count = int(tensor.shape[0]) * lanes
    if count and not tensor.data:
        raise OvstageContractError(f"{attribute!r} returned a null data pointer")
    device = _tensor_device(tensor)
    return wp.array(
        ptr=int(tensor.data or 0) + int(tensor.byte_offset),
        shape=(count,),
        dtype=dtype,
        device=device,
        copy=False,
    ), device


def _column_tokens(runtime: _RuntimeTransport, pd: Any, columns: Sequence[_ReadColumn]):
    result = {runtime.token(pd, column.attribute): column for column in columns}
    if len(result) != len(columns):
        raise ValueError("runtime read contains duplicate attribute tokens")
    return result


def _make_read_plan(runtime: _RuntimeTransport, pd: Any, columns: Sequence[_ReadColumn]) -> _ReadPlan:
    by_token = _column_tokens(runtime, pd, columns)
    return _ReadPlan(tuple(columns), tuple(by_token), by_token)


def _group_rows(
    column: _ReadColumn,
    group: Any,
    runtime: _RuntimeTransport,
    source_row_count: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    prim_rows = []
    source_rows = []
    query_rows = []
    accepted_rows = column.expected_rows if column.accepted_rows is None else column.accepted_rows
    for local in range(group.prim_count):
        query_row = int(group.prim_index(local))
        source_row = int(group.data_row_index(local))
        if query_row < 0 or query_row >= len(runtime.paths):
            raise OvstageContractError(
                f"runtime attribute {column.attribute!r} returned query row {query_row} "
                f"outside [0, {len(runtime.paths)})"
            )
        if source_row < 0 or source_row >= source_row_count:
            raise OvstageContractError(
                f"runtime attribute {column.attribute!r} returned data row {source_row} "
                f"outside [0, {source_row_count})"
            )
        prim_rows.append(query_row)
        if query_row in accepted_rows:
            source_rows.append(source_row)
            query_rows.append(query_row)
    if len(prim_rows) != len(set(prim_rows)):
        raise OvstageContractError(f"runtime attribute {column.attribute!r} returned duplicate query rows")
    return tuple(sorted(prim_rows)), tuple(source_rows), tuple(query_rows)


def _capture_group_slice(column: _ReadColumn, group: Any, runtime: _RuntimeTransport):
    if group.is_delete or group.tensor_count != 1:
        raise OvstageContractError(f"{column.kind} {column.attribute!r} is unavailable in the bound layout")
    source, source_device = _tensor_source_view(
        group.tensor(0),
        dtype=column.dtype,
        lanes=column.lanes,
        attribute=column.attribute,
    )
    prim_rows, source_rows, query_rows = _group_rows(
        column,
        group,
        runtime,
        int(source.shape[0]) // column.lanes,
    )
    destination_rows = column.destination_rows(tuple(query_rows))
    layout = _ReadSlice(
        generation=int(group.meta.layout_generation),
        ordinal=int(group.ordinal),
        source_size=int(source.shape[0]),
        count=len(query_rows),
        prim_rows_host=prim_rows,
        query_rows_host=tuple(query_rows),
        source_rows=runtime.indices(source_rows),
        destination_rows=runtime.indices(destination_rows),
        staging=wp.empty(source.shape, dtype=column.dtype, device=runtime.device),
    )
    return layout, source, source_device


def _capture_runtime_read_layout(stage: Any, pd: Any, runtime: _RuntimeTransport, ordinal: int) -> None:
    """Capture fixed read-group layouts and allocate exact staging buffers."""
    from ovstage import OrdinalRange  # noqa: PLC0415

    columns = tuple(_runtime_read_columns(runtime))
    token_to_column = _column_tokens(runtime, pd, columns)
    seen = {column.attribute: set() for column in columns}
    if not runtime.paths:
        runtime.read_plan = _make_read_plan(runtime, pd, columns)
        return
    with _stage._path_list_query(stage, pd, runtime.paths) as query:
        with stage.read_attributes(query, list(token_to_column), OrdinalRange.latest(ordinal)) as read:
            read.wait()
            group = read.fetch_next()
            while group is not None:
                try:
                    column = token_to_column.get(group.attribute)
                    if column is None:
                        raise OvstageContractError(
                            f"runtime layout returned unexpected attribute token {group.attribute}"
                        )
                    _stage._validate_payload_ordinal(group, ordinal, f"runtime attribute {column.attribute!r}")
                    layout, _, _ = _capture_group_slice(column, group, runtime)
                    if any(existing.prim_rows_host == layout.prim_rows_host for existing in column.slices):
                        raise OvstageContractError(
                            f"runtime layout for {column.attribute!r} contains duplicate prim groups"
                        )
                    column.slices.append(layout)
                    seen[column.attribute].update(layout.query_rows_host)
                finally:
                    stage.release_group(group)
                group = read.fetch_next()

    for column in columns:
        missing = sorted(column.expected_rows.difference(seen[column.attribute]))
        if missing:
            if column.deferred_if_missing:
                column.slices.clear()
                column.deferred = True
            else:
                raise OvstageContractError(
                    f"{column.kind} {column.attribute!r} is unavailable at {runtime.paths[missing[0]]}"
                )
        elif not column.slices and not column.expected_rows:
            column.deferred = True
    runtime.read_plan = _make_read_plan(runtime, pd, columns)


def _topology_changed(column: _ReadColumn, detail: str) -> OvstageContractError:
    return OvstageContractError(
        f"runtime layout for {column.attribute!r} changed ({detail}); recreate the StageBinding"
    )


def _read_runtime_columns(
    stage: Any,
    query: Any,
    ordinal: int,
    *,
    runtime: _RuntimeTransport,
    plan: _ReadPlan,
    between_phases: Optional[Any] = None,
) -> None:
    """Consume current payloads below a ceiling against the immutable read plan."""
    from ovstage import OrdinalRange  # noqa: PLC0415

    if not plan.columns:
        return
    columns = plan.columns
    token_to_column = plan.by_token
    group_counts = {column.attribute: 0 for column in columns}
    seen_slice_rows = {column.attribute: set() for column in columns}
    captured_by_rows = {}
    for column in columns:
        by_rows = {layout.prim_rows_host: layout for layout in column.slices}
        if len(by_rows) != len(column.slices):
            raise OvstageContractError(f"runtime layout for {column.attribute!r} contains duplicate prim groups")
        captured_by_rows[column.attribute] = by_rows
    pending_rows = {column.attribute: set() for column in columns if column.deferred}
    pending_slices = {
        column.attribute: list(column.slices) for column in columns if column.deferred
    }
    groups = []
    pending: List[_PendingRead] = []
    keepalive = []
    device_work_started = False
    stream = wp.get_stream(runtime.device) if runtime.device.is_cuda else None
    if stream is not None and stream.device.is_capturing:
        raise RuntimeError("runtime ingress cannot run during CUDA graph capture")
    try:
        with stage.read_attributes(query, plan.tokens, OrdinalRange.latest(ordinal)) as read:
            read.wait()
            group = read.fetch_next()
            while group is not None:
                groups.append(group)
                column = token_to_column.get(group.attribute)
                if column is None:
                    raise OvstageContractError(f"runtime read returned unexpected attribute token {group.attribute}")
                _stage._validate_payload_ordinal(group, ordinal, f"runtime attribute {column.attribute!r}")
                attribute = column.attribute
                layouts = pending_slices.get(attribute, column.slices)
                if group.is_delete and not column.expected_rows:
                    group_counts[attribute] = len(layouts)
                    seen_slice_rows[attribute].update(layout.prim_rows_host for layout in layouts)
                    group = read.fetch_next()
                    continue
                if layouts and attribute not in pending_slices and not group_counts[attribute]:
                    captured = layouts[0]
                    generation = int(group.meta.layout_generation)
                    if (
                        captured.generation == 0
                        and generation != captured.generation
                        and int(group.ordinal) > captured.ordinal
                    ):
                        pending_slices[attribute] = []
                        pending_rows[attribute] = set()
                        layouts = pending_slices[attribute]
                if attribute in pending_slices:
                    layout, source, source_device = _capture_group_slice(column, group, runtime)
                    layouts.append(layout)
                    source_rows = layout.source_rows
                    destination_rows = layout.destination_rows
                else:
                    if group.is_delete:
                        raise _topology_changed(column, "a required group was deleted")
                    if group.tensor_count != 1:
                        raise _topology_changed(column, f"tensor count changed to {group.tensor_count}")
                    source, source_device = _tensor_source_view(
                        group.tensor(0),
                        dtype=column.dtype,
                        lanes=column.lanes,
                        attribute=column.attribute,
                    )
                    prim_rows, source_rows_host, query_rows_host = _group_rows(
                        column,
                        group,
                        runtime,
                        int(source.shape[0]) // column.lanes,
                    )
                    layout = captured_by_rows[attribute].get(prim_rows)
                    if layout is None:
                        raise _topology_changed(column, "prim membership changed")
                    if int(source.shape[0]) != layout.source_size:
                        raise _topology_changed(column, "tensor shape changed")
                    generation = int(group.meta.layout_generation)
                    if generation != layout.generation:
                        raise _topology_changed(column, "layout generation changed")
                    source_rows = runtime.indices(source_rows_host)
                    destination_rows = runtime.indices(column.destination_rows(query_rows_host))
                if layout.prim_rows_host in seen_slice_rows[attribute]:
                    raise _topology_changed(column, "prim membership changed")
                if attribute in pending_rows:
                    pending_rows[attribute].update(layout.query_rows_host)
                seen_slice_rows[attribute].add(layout.prim_rows_host)
                group_counts[attribute] += 1
                if layout.count:
                    pending.append(
                        _PendingRead(
                            column=column,
                            layout=layout,
                            source=source,
                            source_device=source_device,
                            source_rows=source_rows,
                            destination_rows=destination_rows,
                            cuda_stream=int(group.raw.data.cuda_sync.stream or 0),
                            cuda_wait_event=int(group.raw.data.cuda_sync.wait_event or 0),
                        )
                    )
                group = read.fetch_next()

        for column in columns:
            layouts = pending_slices.get(column.attribute, column.slices)
            if column.attribute in pending_rows:
                missing = sorted(column.expected_rows.difference(pending_rows[column.attribute]))
                if missing:
                    if column.deferred:
                        raise OvstageContractError(
                            f"{column.kind} {column.attribute!r} is unavailable at {runtime.paths[missing[0]]}"
                        )
                    raise _topology_changed(column, "prim membership changed")
            if column.expected_rows and group_counts[column.attribute] != len(layouts):
                raise _topology_changed(column, "a read group disappeared")

        stream_scope = wp.ScopedStream(stream) if stream is not None else contextlib.nullcontext()
        with stream_scope:
            current_phase = 0
            for item in sorted(pending, key=lambda value: value.column.phase):
                if item.column.phase != current_phase:
                    if current_phase == 0 and between_phases is not None:
                        device_work_started = True
                        between_phases()
                    current_phase = item.column.phase

                source = item.source
                _wait_for_cuda_sync(item, stream, keepalive)

                if item.source_device != runtime.device:
                    if stream is not None:
                        wp.copy(item.layout.staging, source, stream=stream)
                    else:
                        wp.copy(item.layout.staging, source)
                        if item.source_device.is_cuda:
                            wp.synchronize_stream(item.source_device)
                    source = item.layout.staging

                device_work_started = True
                item.column.consume(
                    source,
                    item.source_rows,
                    item.destination_rows,
                    item.layout.count,
                )

            if current_phase == 0 and between_phases is not None:
                device_work_started = True
                between_phases()

        if stream is not None and device_work_started:
            wp.synchronize_stream(stream)
            device_work_started = False

        for column in columns:
            if column.attribute in pending_slices and group_counts[column.attribute]:
                column.slices[:] = pending_slices[column.attribute]
                column.deferred = False

    finally:
        if stream is not None and device_work_started:
            wp.synchronize_stream(stream)
        for group in groups:
            stage.release_group(group)


def _validate_array(owner: Any, name: str, size: int, device: Any):
    """Return a required Warp array after checking its bound-model shape/device."""
    value = getattr(owner, name, None)
    shape = getattr(value, "shape", ())
    if len(shape) != 1 or int(shape[0]) != size:
        raise ValueError(f"{type(owner).__name__}.{name} is not compatible with the bound model")
    value_device = getattr(value, "device", None)
    if value_device != device:
        raise ValueError(
            f"{type(owner).__name__}.{name} is on {value_device}, but the bound model is on {device}"
        )
    return value


def _validate_control_array(
    value: Any,
    name: str,
    expected_size: int,
    device: Any,
    *,
    required: bool,
) -> None:
    if value is None:
        if required:
            raise ValueError(f"control has no {name} array for live drive targets")
        return
    shape = getattr(value, "shape", ())
    if len(shape) != 1 or int(shape[0]) != expected_size:
        raise ValueError(f"control {name} array is not compatible with the bound model")
    value_device = getattr(value, "device", None)
    if value_device != device:
        raise ValueError(f"control {name} array is on {value_device}, but the bound model is on {device}")


def _validate_body_state(state: Any, model: Any, velocities: bool) -> None:
    if not model.body_count:
        return
    _validate_array(state, "body_q", model.body_count, model.device)
    if velocities:
        _validate_array(state, "body_qd", model.body_count, model.device)


def _validate_joint_state(state: Any, model: Any) -> None:
    if not model.joint_count:
        return
    _validate_array(state, "joint_q", int(model.joint_q.shape[0]), model.device)
    _validate_array(state, "joint_qd", int(model.joint_qd.shape[0]), model.device)


def _wait_operations(operations: Sequence[Any], *, suppress: bool = False) -> None:
    """Wait/release every accepted operation, preserving the first failure."""
    first_error = None
    for operation in operations:
        try:
            operation.wait()
        except Exception as error:  # keep draining: each operation owns input lifetimes
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress:
        raise first_error


def _body_read_columns(runtime: _RuntimeTransport) -> List[_ReadColumn]:
    """Describe body columns in the combined runtime query."""

    def consume_pose(source, source_rows, destination_rows, count):
        wp.launch(
            _decode_pose_kernel,
            dim=count,
            inputs=[source, source_rows, destination_rows, runtime.body_q_in],
            device=runtime.device,
        )

    expected = frozenset(range(runtime.model.body_count))
    columns = [
        _ReadColumn(
            WORLD_MATRIX,
            wp.float64,
            16,
            expected,
            consume_pose,
            destination_rows=lambda rows: rows,
        )
    ]
    linear, angular = runtime.linear_in, runtime.angular_in

    def consume_vector(destination):
        def consume(source, source_rows, destination_rows, count):
            wp.launch(
                _scatter_vec3_kernel,
                dim=count,
                inputs=[source, source_rows, destination_rows, destination],
                device=runtime.device,
            )

        return consume

    columns.extend(
        [
            _ReadColumn(
                BODY_VELOCITY,
                wp.float32,
                3,
                expected,
                consume_vector(linear),
                destination_rows=lambda rows: rows,
            ),
            _ReadColumn(
                BODY_ANGULAR_VELOCITY,
                wp.float32,
                3,
                expected,
                consume_vector(angular),
                destination_rows=lambda rows: rows,
            ),
        ]
    )
    return columns


def _joint_scalar_column(
    runtime: _RuntimeTransport,
    channel: _JointChannel,
    attribute: str,
    destination: Any,
    destination_indices: Sequence[int],
    query_rows: Sequence[int],
    scale: float,
    expected_rows: frozenset[int],
    kind: str,
    accepted_rows: Optional[frozenset[int]] = None,
) -> _ReadColumn:
    destination_by_query = dict(zip(query_rows, destination_indices, strict=True))

    def consume(source, source_rows, destination_rows, count):
        wp.launch(
            _scatter_scalar_kernel,
            dim=count,
            inputs=[source, source_rows, destination_rows, wp.float32(scale), destination],
            device=channel.device,
        )

    return _ReadColumn(
        attribute=attribute,
        dtype=wp.float32,
        lanes=1,
        expected_rows=expected_rows,
        consume=consume,
        destination_rows=lambda rows: tuple(destination_by_query[row] for row in rows),
        kind=kind,
        phase=1,
        accepted_rows=accepted_rows,
        deferred_if_missing=True,
    )


def _joint_read_columns(
    runtime: _RuntimeTransport,
) -> List[_ReadColumn]:
    """Describe joint state/control columns in the combined runtime query."""
    columns: List[_ReadColumn] = []
    for channel in runtime.channels:
        scale = 1.0 / _DEG_PER_RAD if channel.rotational else 1.0
        query_rows = runtime.channel_rows[channel.instance]
        all_rows = frozenset(query_rows)
        columns.append(
            _joint_scalar_column(
                runtime,
                channel,
                joint_state(channel.instance, "position"),
                runtime.joint_q_in,
                channel.q_indices_host,
                query_rows,
                scale,
                all_rows,
                "runtime state attribute",
            )
        )
        columns.append(
            _joint_scalar_column(
                runtime,
                channel,
                joint_state(channel.instance, "velocity"),
                runtime.joint_qd_in,
                channel.qd_indices_host,
                query_rows,
                scale,
                all_rows,
                "runtime state attribute",
            )
        )

        position_attr = joint_drive(channel.instance, "targetPosition")
        velocity_attr = joint_drive(channel.instance, "targetVelocity")
        has_drive = bool(channel.position_target_rows or channel.velocity_target_rows)
        if has_drive and runtime.target_q_in is not None:
            columns.append(
                _joint_scalar_column(
                    runtime,
                    channel,
                    position_attr,
                    runtime.target_q_in,
                    channel.target_q_indices_host,
                    query_rows,
                    scale,
                    frozenset(query_rows[row] for row in channel.position_target_rows),
                    "runtime command attribute",
                    accepted_rows=all_rows,
                )
            )
        if has_drive and runtime.target_qd_in is not None:
            columns.append(
                _joint_scalar_column(
                    runtime,
                    channel,
                    velocity_attr,
                    runtime.target_qd_in,
                    channel.qd_indices_host,
                    query_rows,
                    scale,
                    frozenset(query_rows[row] for row in channel.velocity_target_rows),
                    "runtime command attribute",
                    accepted_rows=all_rows,
                )
            )

    return columns


def _runtime_read_columns(runtime: _RuntimeTransport) -> List[_ReadColumn]:
    return [*_body_read_columns(runtime), *_joint_read_columns(runtime)]


def _combined_runtime_paths(
    body_paths: Sequence[str], runtime: _RuntimeTransport
) -> Tuple[List[str], Dict[str, Tuple[int, ...]]]:
    """Return one stable query order and each joint channel's row indices."""
    paths = list(body_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("runtime body paths must be unique")
    row_by_path = {path: row for row, path in enumerate(paths)}
    channel_rows: Dict[str, Tuple[int, ...]] = {}
    for channel in runtime.channels:
        rows = []
        for path in channel.paths:
            if path in row_by_path:
                raise ValueError(f"runtime path {path!r} is used by more than one body or joint")
            row_by_path[path] = len(paths)
            rows.append(len(paths))
            paths.append(path)
        channel_rows[channel.instance] = tuple(rows)
    return paths, channel_rows


def read_runtime(
    stage: Any,
    pd: Any,
    ordinal: int,
    state: Any,
    *,
    runtime: _RuntimeTransport,
    control: Any,
) -> None:
    """Apply every available channel below one sealed read ceiling."""
    model = runtime.model
    _validate_body_state(state, model, True)
    if model.joint_count:
        _validate_joint_state(state, model)
    if control is None:
        raise TypeError("control is required")
    if not runtime.paths:
        return

    control_target_q = control.joint_target_q
    control_target_qd = control.joint_target_qd
    _validate_control_array(
        control_target_q,
        "position-target",
        int(runtime.target_q_in.shape[0]) if runtime.target_q_in is not None else 0,
        model.device,
        required=runtime.position_target_required,
    )
    _validate_control_array(
        control_target_qd,
        "velocity-target",
        int(runtime.target_qd_in.shape[0]) if runtime.target_qd_in is not None else 0,
        model.device,
        required=runtime.velocity_target_required,
    )

    if model.joint_count:
        wp.copy(runtime.joint_q_in, state.joint_q)
        wp.copy(runtime.joint_qd_in, state.joint_qd)
    if control_target_q is not None:
        wp.copy(runtime.target_q_in, control_target_q)
    if control_target_qd is not None:
        wp.copy(runtime.target_qd_in, control_target_qd)

    plan = runtime.read_plan
    if plan is None:
        raise RuntimeError("runtime read plan was not captured")

    def finish_body_decode() -> None:
        wp.launch(
            _decode_velocity_kernel,
            dim=model.body_count,
            inputs=[runtime.body_rows, runtime.linear_in, runtime.angular_in, runtime.body_qd_in],
            device=runtime.device,
        )

        _derive_joint_state(
            runtime,
            SimpleNamespace(body_q=runtime.body_q_in, body_qd=runtime.body_qd_in),
            runtime.joint_q_in,
            runtime.joint_qd_in,
        )

    with _stage._path_list_query(stage, pd, runtime.paths) as query:
        _read_runtime_columns(
            stage,
            query,
            ordinal,
            runtime=runtime,
            plan=plan,
            between_phases=finish_body_decode,
        )

    # Commit only after every requested column has been validated and decoded.
    wp.copy(state.body_q, runtime.body_q_in)
    wp.copy(state.body_qd, runtime.body_qd_in)
    if model.joint_count:
        wp.copy(state.joint_q, runtime.joint_q_in)
        wp.copy(state.joint_qd, runtime.joint_qd_in)
    if control_target_q is not None:
        wp.copy(control_target_q, runtime.target_q_in)
    if control_target_qd is not None:
        wp.copy(control_target_qd, runtime.target_qd_in)
