# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident state and control transport between Newton and ovstage.

Warp kernels encode and decode on the model device. Inbound DLTensors are aliased
while their groups are alive and copied only when devices differ.
"""

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import warp as wp
from ovstage import (
    AttributeSemantic,
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLTensor,
    WriteDesc,
)

from . import _stage
from ._errors import OvstageContractError
from ._schema_names import BODY_ANGULAR_VELOCITY, BODY_VELOCITY, WORLD_MATRIX, joint_drive, joint_state
from .ovnewton import Query, ReadGroup, ReadResult, _destroy_path_lists, _finalize_path_lists, _ReadStorage

_DEG_PER_RAD = 57.29577951308232

_BODY_Q = "body_q"
_BODY_QD = "body_qd"
_JOINT_Q = "joint_q"
_JOINT_QD = "joint_qd"
_OUTPUT_ATTRIBUTES = (_BODY_Q, _BODY_QD, _JOINT_Q, _JOINT_QD)


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


@wp.kernel
def _encode_body_for_ovstage_kernel(
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    pose_out: wp.array(dtype=wp.float64),
    linear_out: wp.array(dtype=wp.float32),
    angular_out: wp.array(dtype=wp.float32),
) -> None:
    """Encode the canonical ovstage body columns."""
    i = wp.tid()
    t = body_q[i]
    p = wp.transform_get_translation(t)
    qf = wp.normalize(wp.transform_get_rotation(t))

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
    pose_out[matrix + 0] = rcv00
    pose_out[matrix + 1] = rcv10
    pose_out[matrix + 2] = rcv20
    pose_out[matrix + 3] = zero
    pose_out[matrix + 4] = rcv01
    pose_out[matrix + 5] = rcv11
    pose_out[matrix + 6] = rcv21
    pose_out[matrix + 7] = zero
    pose_out[matrix + 8] = rcv02
    pose_out[matrix + 9] = rcv12
    pose_out[matrix + 10] = rcv22
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
def _scatter_scalar_kernel(
    source: wp.array(dtype=wp.float32),
    source_rows: wp.array(dtype=wp.int32),
    destination_rows: wp.array(dtype=wp.int32),
    scale: wp.float32,
    destination: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()
    destination[destination_rows[i]] = source[source_rows[i]] * scale


def _device_dltensor(wa, n: int, lanes: int, code: int, bits: int):
    """View a contiguous Warp array as a DLTensor."""
    is_cuda = bool(getattr(wa.device, "is_cuda", False))
    dev = DLDeviceType.kDLCUDA if is_cuda else DLDeviceType.kDLCPU
    dev_id = int(getattr(wa.device, "ordinal", 0) or 0) if is_cuda else 0

    t = DLTensor()
    t.data = ctypes.c_void_p(int(wa.ptr))
    t.device = DLDevice(dev, dev_id)
    t.ndim = 1
    ss = (ctypes.c_int64 * 1)(n)
    t._ss = ss  # keepalive
    t.shape = ctypes.cast(ss, ctypes.POINTER(ctypes.c_int64))
    t.strides = None
    t.byte_offset = 0
    t.dtype = DLDataType(code=code, bits=bits, lanes=lanes)
    t._wa = wa  # keepalive the Warp array
    return t


def _device_index_tensor(rows: Optional[Tuple[int, ...]], device: Any) -> Any:
    if rows is None:
        return None
    return _device_dltensor(
        wp.array(rows, dtype=wp.uint32, device=device),
        len(rows),
        lanes=1,
        code=DLDataTypeCode.kDLUInt,
        bits=32,
    )


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


@dataclass(frozen=True)
class _OutputCatalog:
    """Static path and coordinate metadata for output queries."""

    body_by_path: Mapping[str, int]
    joint_by_path: Mapping[str, int]
    joint_q_start: Tuple[int, ...]
    joint_qd_start: Tuple[int, ...]
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
    bits: int = 32,
    semantic: int = AttributeSemantic.NONE,
) -> WriteDesc:
    event = int(runtime.producer_event.cuda_event) if runtime.producer_event is not None else None
    return WriteDesc(
        attribute=runtime.token(pd, attribute),
        tensors=_device_dltensor(
            output,
            count,
            lanes=lanes,
            code=DLDataTypeCode.kDLFloat,
            bits=bits,
        ),
        is_array=False,
        semantic=semantic,
        cuda_event=event,
    )


def _create_publication_batches(binding: Any) -> Tuple[Tuple[int, Tuple[WriteDesc, ...]], ...]:
    runtime = binding._runtime
    pd = binding._pd
    batches = []
    try:
        body_paths = list(runtime.model.body_label)
        if body_paths:
            path_list = pd.create_path_list_from_strings(body_paths)
            batches.append(
                (
                    path_list,
                    (
                        _publication_write(
                            runtime,
                            pd,
                            WORLD_MATRIX,
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
        result = tuple(batches)
        _finalize_path_lists(binding, pd, (path_list for path_list, _ in result))
        return result
    except Exception:
        _destroy_path_lists(pd, tuple(path_list for path_list, _ in batches))
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


def create_query(
    binding: Any,
    stage_query: Any = None,
    *,
    paths: Optional[Sequence[str]] = None,
) -> Query:
    """Compile an ovstage query or explicit paths into immutable Newton indices."""
    if (stage_query is None) == (paths is None):
        raise TypeError("pass exactly one of stage_query or paths")
    strict = paths is not None
    if strict:
        if isinstance(paths, (str, bytes)):
            raise TypeError("paths must be a sequence of absolute prim paths")
        selected_paths = list(paths)
        if any(not isinstance(path, str) or not path.startswith("/") for path in selected_paths):
            raise ValueError("query paths must be absolute prim paths")
        if len(selected_paths) != len(set(selected_paths)):
            raise ValueError("query paths contain duplicates")
    else:
        selected_paths = _query_paths(binding, stage_query)

    runtime = binding._runtime
    catalog = binding._output_catalog

    body_paths = []
    body_indices = []
    selected_joints = []
    for path in selected_paths:
        body = catalog.body_by_path.get(path)
        if body is not None:
            body_paths.append(path)
            body_indices.append(body)
            continue
        joint = catalog.joint_by_path.get(path)
        if joint is not None:
            selected_joints.append((path, joint))
            continue
        if strict:
            raise ValueError(f"path is not an output-capable bound body or joint: {path}")

    path_lists = []
    try:
        body_index_tensor = None
        body_indices_host = None
        body_prim_list = 0
        if body_paths:
            body_prim_list = binding._pd.create_path_list_from_strings(body_paths)
            path_lists.append(body_prim_list)

        body_rows = tuple(body_indices)
        if body_rows != tuple(range(len(body_rows))):
            body_indices_host = body_rows
            body_index_tensor = _device_index_tensor(body_indices_host, runtime.device)
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
        available = []
        if body_paths:
            available.extend((_BODY_Q, _BODY_QD))
        if joint_q is not None:
            available.append(_JOINT_Q)
        if joint_qd is not None:
            available.append(_JOINT_QD)
        attributes = tuple(runtime.token(binding._pd, name) for name in available)
        return Query(
            binding,
            body_index_tensor=body_index_tensor,
            body_indices_host=body_indices_host,
            body_prim_list=body_prim_list,
            body_count=len(body_paths),
            joint_q=joint_q,
            joint_qd=joint_qd,
            attributes=attributes,
            prim_count=len(body_paths) + len(selected_joints),
            path_lists=tuple(path_lists),
        )
    except Exception:
        _destroy_path_lists(binding._pd, tuple(path_lists))
        raise


def _create_output_catalog(binding: Any) -> _OutputCatalog:
    runtime = binding._runtime
    model = runtime.model
    body_by_path = {path: index for index, path in enumerate(model.body_label)}
    q_start = tuple(int(value) for value in model.joint_q_start.numpy())
    qd_start = tuple(int(value) for value in model.joint_qd_start.numpy())
    joint_by_path = {
        path: joint
        for joint, path in enumerate(model.joint_label)
        if isinstance(path, str)
        and path.startswith("/")
        and (q_start[joint + 1] > q_start[joint] or qd_start[joint + 1] > qd_start[joint])
        and path not in body_by_path
    }
    names_by_token = {runtime.token(binding._pd, name): name for name in _OUTPUT_ATTRIBUTES}
    return _OutputCatalog(
        body_by_path=MappingProxyType(body_by_path),
        joint_by_path=MappingProxyType(joint_by_path),
        joint_q_start=q_start,
        joint_qd_start=qd_start,
        all_paths=tuple(model.body_label) + tuple(joint_by_path),
        names_by_token=MappingProxyType(names_by_token),
    )


def initialize_output(binding: Any) -> None:
    binding._output_catalog = _create_output_catalog(binding)
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
    data_indices_host: Optional[Tuple[int, ...]] = None,
    data_index_tensor: Any = None,
) -> ReadGroup:
    return ReadGroup(
        storage,
        attribute=attribute,
        prim_list=prim_list,
        prim_count=prim_count,
        tensors=(
            _device_dltensor(
                source,
                source_row_count,
                lanes=width,
                code=DLDataTypeCode.kDLFloat,
                bits=32,
            ),
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
                width=layout.width,
                data_indices_host=layout.data_indices_host,
                data_index_tensor=layout.data_index_tensor,
            )
        )
    return groups


def _prepare_joint_output(source: Any, selection: _JointOutputSelection, device: Any, stream: Any) -> Any:
    if selection.source_indices is None:
        return source
    output = wp.empty(selection.source_indices.shape[0], dtype=wp.float32, device=device)
    wp.launch(
        _gather_scalar_output_kernel,
        dim=output.shape[0],
        inputs=[source, selection.source_indices, output],
        device=device,
        stream=stream,
    )
    return output


def _read_native_output(
    binding: Any,
    state: Any,
    query: Query,
    names: Sequence[str],
) -> ReadResult:
    runtime = binding._runtime
    device = runtime.device
    body_q = None
    body_qd = None

    if query._body_count and _BODY_Q in names:
        body_q = _validate_array(state, "body_q", runtime.model.body_count, device)
    if query._body_count and _BODY_QD in names:
        body_qd = _validate_array(state, "body_qd", runtime.model.body_count, device)

    joint_q = None
    joint_qd = None
    if query._joint_q is not None and _JOINT_Q in names:
        joint_q = _validate_array(state, "joint_q", int(runtime.model.joint_q.shape[0]), device)
    if query._joint_qd is not None and _JOINT_QD in names:
        joint_qd = _validate_array(state, "joint_qd", int(runtime.model.joint_qd.shape[0]), device)

    stream = wp.get_stream(device) if device.is_cuda else None
    joint_q_out = _prepare_joint_output(joint_q, query._joint_q, device, stream) if joint_q is not None else None
    joint_qd_out = _prepare_joint_output(joint_qd, query._joint_qd, device, stream) if joint_qd is not None else None
    event = None
    if stream is not None and (
        body_q is not None or body_qd is not None or joint_q_out is not None or joint_qd_out is not None
    ):
        event = wp.Event(device)
        stream.record_event(event)
    storage = _ReadStorage(
        query,
        event=event,
        keepalive=tuple(
            source
            for source, output in ((joint_q, joint_q_out), (joint_qd, joint_qd_out))
            if source is not None and output is not source
        ),
    )
    groups = []
    for name in names:
        if name in (_BODY_Q, _BODY_QD):
            source, width = (body_q, 7) if name == _BODY_Q else (body_qd, 6)
            if source is not None:
                groups.append(
                    _native_output_group(
                        storage,
                        attribute=runtime.token(binding._pd, name),
                        prim_list=query._body_prim_list,
                        prim_count=query._body_count,
                        source=source,
                        source_row_count=runtime.model.body_count,
                        width=width,
                        data_indices_host=query._body_indices_host,
                        data_index_tensor=query._body_index_tensor,
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
    return ReadResult(groups)


def read_output(
    binding: Any,
    state: Any,
    *,
    query: Optional[Query],
    attributes: Sequence[Any],
) -> ReadResult:
    """Expose selected Newton state columns through read-only groups."""
    selected_query = binding._all_query if query is None else query
    if not isinstance(selected_query, Query) or selected_query._binding_ref() is not binding:
        raise ValueError("query belongs to a different StageBinding")
    names = _normalize_output_attributes(binding, attributes)
    return _read_native_output(binding, state, selected_query, names)


def publish_output(binding: Any, state: Any, ordinal: int) -> None:
    """Encode and publish every canonical ovstage output."""
    with binding._runtime_lock:
        runtime = binding._runtime
        model = runtime.model
        body_q = _validate_array(state, "body_q", model.body_count, runtime.device)
        body_qd = _validate_array(state, "body_qd", model.body_count, runtime.device)
        joint_q = None
        joint_qd = None
        if runtime.channels:
            joint_q = _validate_array(state, "joint_q", int(model.joint_q.shape[0]), runtime.device)
            joint_qd = _validate_array(state, "joint_qd", int(model.joint_qd.shape[0]), runtime.device)
        stream = wp.get_stream(runtime.device) if runtime.device.is_cuda else None
        if model.body_count:
            wp.launch(
                _encode_body_for_ovstage_kernel,
                dim=model.body_count,
                inputs=[body_q, body_qd, runtime.pose_out, runtime.linear_out, runtime.angular_out],
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
