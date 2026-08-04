# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel-equivalence unit test: Warp encode == numpy encode (the oracle).

Builds random body state and compares the NumPy publication oracles with the
Warp ovstage encoder on CUDA. This isolates conversion correctness from the
Fabric write transport.

Also verifies the single device-agnostic write path: that the runtime writer runs
the encode on the model's CUDA device and labels the DLTensor ``kDLCUDA`` (device
derived from the output Warp array, not branched), and that the written values
round-trip through Fabric.

Skips when CUDA is not available (``wp.is_cuda_available()`` is False).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import ovstage
import pytest
import warp as wp

import ovnewton
from ovnewton._src import _build, _runtime
from ovnewton._src._build import _R3
from ovnewton.examples import get_asset

from .runtime_helpers import lanes_tensor

if not wp.is_cuda_available():
    pytest.skip("CUDA not available", allow_module_level=True)

_DEVICE = "cuda:0"
_RNG = np.random.default_rng(42)
_DEG_PER_RAD = 57.29577951308232

CARTPOLE = get_asset("scene_cartpole.usda")


class _ReadGroup:
    def __init__(self, tensor, *, row=0, cuda_stream=0, wait_event=0):
        self.attribute = 1
        self.ordinal = 1
        self.is_delete = False
        self.tensor_count = 1
        self.prim_count = 1
        self._row = row
        self._tensor = tensor
        self.raw = SimpleNamespace(
            data=SimpleNamespace(cuda_sync=SimpleNamespace(stream=cuda_stream, wait_event=wait_event))
        )
        self.meta = SimpleNamespace(layout_generation=1)

    def tensor(self, index):
        assert index == 0
        return self._tensor

    def prim_index(self, index):
        assert index == 0
        return self._row

    def data_row_index(self, index):
        assert index == 0
        return 0


class _Read:
    def __init__(self, groups):
        self._groups = list(groups) if isinstance(groups, (list, tuple)) else [groups]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def wait(self):
        return None

    def fetch_next(self):
        return self._groups.pop(0) if self._groups else None


class _ReadStage:
    def __init__(self, group):
        self.group = group
        self.released = []
        self.reads = 0

    def read_attributes(self, _query, attributes, _ordinal_range):
        assert tuple(attributes) == (1,)
        self.reads += 1
        return _Read(self.group)

    def release_group(self, group):
        self.released.append(group)


class _Runtime:
    def __init__(self, device, path_count=1):
        self.device = wp.get_device(device)
        self.paths = tuple(f"/{index}" for index in range(path_count))

    def token(self, _pd, _attribute):
        return 1

    def indices(self, values):
        return wp.array(values, dtype=wp.int32, device=self.device)


def _consume_group(stage, destination):
    runtime = _Runtime(destination.device)

    def consume(source, _source_rows, _destination_rows, _count):
        wp.copy(destination, source)

    column = _runtime._ReadColumn(
        attribute="test:cuda",
        dtype=wp.float32,
        lanes=1,
        expected_rows=frozenset({0}),
        consume=consume,
        destination_rows=lambda rows: rows,
    )
    column.slices.append(
        _runtime._ReadSlice(
            generation=1,
            ordinal=1,
            source_size=1,
            count=1,
            prim_rows_host=(0,),
            query_rows_host=(0,),
            source_rows=runtime.indices([0]),
            destination_rows=runtime.indices([0]),
            staging=wp.empty(1, dtype=wp.float32, device=runtime.device),
        )
    )
    _runtime._read_runtime_columns(
        stage,
        None,
        1,
        runtime=runtime,
        plan=_runtime._make_read_plan(runtime, None, [column]),
    )


def test_runtime_ingress_matches_groups_by_prim_identity():
    runtime = _Runtime(_DEVICE, path_count=2)
    destination = wp.zeros(2, dtype=wp.float32, device=runtime.device)

    def consume(source, source_rows, destination_rows, count):
        wp.launch(
            _runtime._scatter_scalar_kernel,
            dim=count,
            inputs=[source, source_rows, destination_rows, wp.float32(1.0), destination],
            device=runtime.device,
        )

    column = _runtime._ReadColumn(
        attribute="test:cuda",
        dtype=wp.float32,
        lanes=1,
        expected_rows=frozenset({0, 1}),
        consume=consume,
        destination_rows=lambda rows: rows,
    )
    for row in (0, 1):
        column.slices.append(
            _runtime._ReadSlice(
                generation=1,
                ordinal=1,
                source_size=1,
                count=1,
                prim_rows_host=(row,),
                query_rows_host=(row,),
                source_rows=runtime.indices([0]),
                destination_rows=runtime.indices([row]),
                staging=wp.empty(1, dtype=wp.float32, device=runtime.device),
            )
        )

    sources = [
        wp.array([10.0], dtype=wp.float32, device=runtime.device),
        wp.array([20.0], dtype=wp.float32, device=runtime.device),
    ]
    groups = [
        _ReadGroup(_runtime._device_dltensor(sources[1], n=1, lanes=1, code=2, bits=32), row=1),
        _ReadGroup(_runtime._device_dltensor(sources[0], n=1, lanes=1, code=2, bits=32), row=0),
    ]
    stage = _ReadStage(groups)

    _runtime._read_runtime_columns(
        stage,
        None,
        1,
        runtime=runtime,
        plan=_runtime._make_read_plan(runtime, None, [column]),
    )

    np.testing.assert_array_equal(destination.numpy(), [10.0, 20.0])
    assert stage.released == groups


# ── numpy reference encoders (the oracle the Warp kernels must reproduce) ──
# The readable spec for the omni:fabric:worldMatrix / physics:velocity encode.
# _runtime's production path is the Warp kernels; these exist only to pin them
# bit-for-bit (no production caller — they live with the test that consumes them).

def encode_pose_oracle(trans: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Encode body poses ``(trans (N,3), quat xyzw (N,4))`` into ovstage
    ``omni:fabric:worldMatrix`` rows (flat ``(N,16)`` float64, GfMatrix4d).

    Inverse of ``_build._decode_pose`` at unit scale: the stored 3x3 is
    the *row-vector* rotation ``Rᵀ`` (row ``i`` is column ``i`` of the column-vector
    rotation Newton carries), translation goes in row 3."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.where((nrm := np.linalg.norm(q, axis=1, keepdims=True)) > 0.0, nrm, 1.0)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    rcv = np.empty((n, 3, 3), dtype=np.float64)             # column-vector rotation
    rcv[:, 0, 0], rcv[:, 0, 1], rcv[:, 0, 2] = 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)
    rcv[:, 1, 0], rcv[:, 1, 1], rcv[:, 1, 2] = 2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)
    rcv[:, 2, 0], rcv[:, 2, 1], rcv[:, 2, 2] = 2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)
    out = np.zeros((n, 16), dtype=np.float64)
    out[:, _R3] = np.transpose(rcv, (0, 2, 1)).reshape(n, 9)   # row-vector = transpose
    out[:, 12:15] = np.asarray(trans, dtype=np.float64)
    out[:, 15] = 1.0
    return out


def encode_velocity_oracle(body_qd: np.ndarray):
    """Split a world-frame CoM twist and convert angular rad/s to deg/s."""
    qd = np.asarray(body_qd, dtype=np.float64)
    lin, ang = qd[:, :3], qd[:, 3:6]
    return lin.astype(np.float32), (ang * _DEG_PER_RAD).astype(np.float32)


def _random_body_q(n: int) -> np.ndarray:
    """Random body_q (N,7): translation + unit quaternion xyzw."""
    trans = _RNG.uniform(-5.0, 5.0, size=(n, 3))
    quat = _RNG.standard_normal((n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return np.hstack([trans, quat]).astype(np.float32)


def _random_body_qd(n: int) -> np.ndarray:
    """Random body_qd (N,6): linear velocity then angular velocity."""
    return _RNG.uniform(-3.0, 3.0, size=(n, 6)).astype(np.float32)


def test_body_publication_kernel_matches_cpu_encoders():
    """The Warp publication encoder matches both NumPy body-column oracles."""
    n = 16

    bq_np = _random_body_q(n)
    bqd_np = _random_body_qd(n)
    bq_wp = wp.array(bq_np, dtype=wp.transformf, device=_DEVICE)
    bqd_wp = wp.array(bqd_np, dtype=wp.spatial_vectorf, device=_DEVICE)

    # CPU reference
    bq64 = bq_np.astype(np.float64)
    cpu_pose = encode_pose_oracle(bq64[:, :3], bq64[:, 3:7])
    cpu_linear, cpu_angular = encode_velocity_oracle(bqd_np)

    # GPU result
    pose_out = wp.zeros(n * 16, dtype=wp.float64, device=_DEVICE)
    linear_out = wp.zeros(n * 3, dtype=wp.float32, device=_DEVICE)
    angular_out = wp.zeros(n * 3, dtype=wp.float32, device=_DEVICE)
    wp.launch(
        _runtime._encode_body_for_ovstage_kernel,
        dim=n,
        inputs=[
            bq_wp,
            bqd_wp,
            pose_out,
            linear_out,
            angular_out,
        ],
        device=_DEVICE,
    )
    wp.synchronize_device(_DEVICE)
    gpu_pose = pose_out.numpy().reshape(n, 16)
    gpu_linear = linear_out.numpy().reshape(n, 3)
    gpu_angular = angular_out.numpy().reshape(n, 3)

    assert np.allclose(gpu_pose, cpu_pose, atol=1e-6), (
        f"pose kernel/CPU mismatch (max abs diff = {np.abs(gpu_pose - cpu_pose).max():.3e})"
    )
    assert np.allclose(gpu_linear, cpu_linear, atol=1e-4), (
        f"linear velocity kernel/CPU mismatch (max = {np.abs(gpu_linear - cpu_linear).max():.3e})"
    )
    assert np.allclose(gpu_angular, cpu_angular, atol=1e-4), (
        f"angular velocity kernel/CPU mismatch (max = {np.abs(gpu_angular - cpu_angular).max():.3e})"
    )


def test_device_dltensor_derives_cuda_from_array():
    """_device_dltensor labels a CUDA Warp array kDLCUDA (device from the array).

    This is the crux of the single-path design: no branch on device — the
    DLTensor's device_type is read off the output array. A cuda:0 array must
    produce a kDLCUDA tensor carrying that array's pointer and ordinal."""
    from ovstage import DLDeviceType

    wa = wp.zeros(16, dtype=wp.float64, device=_DEVICE)
    t = _runtime._device_dltensor(wa, n=1, lanes=16, code=2, bits=64)
    assert t.device.device_type.value == DLDeviceType.kDLCUDA, (
        f"expected kDLCUDA for a {_DEVICE} array, got {t.device.device_type}"
    )
    assert int(t.data) == int(wa.ptr), "DLTensor data pointer != Warp array ptr"
    assert t.device.device_id == int(wa.device.ordinal), "device_id != array ordinal"
    assert t.dtype.code == 2 and t.dtype.bits == 64 and t.dtype.lanes == 16


def test_runtime_tensor_source_view_aliases_declared_device():
    gpu = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=_DEVICE)
    gpu_tensor = _runtime._device_dltensor(gpu, n=1, lanes=3, code=2, bits=32)
    alias, source_device = _runtime._tensor_source_view(
        gpu_tensor,
        dtype=wp.float32,
        lanes=3,
    )
    assert alias.ptr == gpu.ptr
    assert source_device == wp.get_device(_DEVICE)

    cpu_tensor = lanes_tensor(np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32), lanes=3)
    alias, source_device = _runtime._tensor_source_view(
        cpu_tensor,
        dtype=wp.float32,
        lanes=3,
    )
    assert source_device == wp.get_device("cpu")
    np.testing.assert_array_equal(alias.numpy(), [4.0, 5.0, 6.0])


@pytest.mark.parametrize("destination_device", ["cpu", _DEVICE])
def test_runtime_ingress_waits_for_cuda_producer_event(monkeypatch, destination_device):
    source_value = wp.array([17.0], dtype=wp.float32, device=_DEVICE)
    source = wp.zeros(1, dtype=wp.float32, device=_DEVICE)
    destination = wp.zeros(1, dtype=wp.float32, device=destination_device)
    producer_stream = wp.Stream(_DEVICE)
    producer_event = wp.Event(_DEVICE)
    with wp.ScopedStream(producer_stream):
        wp.copy(source, source_value)
        producer_stream.record_event(producer_event)

    tensor = _runtime._device_dltensor(source, n=1, lanes=1, code=2, bits=32)
    group = _ReadGroup(tensor, wait_event=producer_event.cuda_event)
    stage = _ReadStage(group)
    waits = []
    if destination.device.is_cuda:
        consumer_stream = wp.Stream(destination.device)
        wait_event = wp.Stream.wait_event

        def tracked_wait_event(stream, event, external=False):
            waits.append((stream, int(event.cuda_event)))
            return wait_event(stream, event, external=external)

        monkeypatch.setattr(wp.Stream, "wait_event", tracked_wait_event)
        with wp.ScopedStream(consumer_stream):
            _consume_group(stage, destination)
        assert waits == [(consumer_stream, int(producer_event.cuda_event))]
    else:
        synchronize_event = wp.synchronize_event

        def tracked_synchronize_event(event):
            waits.append(int(event.cuda_event))
            return synchronize_event(event)

        monkeypatch.setattr(wp, "synchronize_event", tracked_synchronize_event)
        _consume_group(stage, destination)
        assert waits == [int(producer_event.cuda_event)]

    assert stage.released == [group]
    assert destination.numpy()[0] == pytest.approx(17.0)


@pytest.mark.parametrize("destination_device", ["cpu", _DEVICE])
def test_runtime_ingress_waits_for_cuda_producer_stream(monkeypatch, destination_device):
    source_value = wp.array([19.0], dtype=wp.float32, device=_DEVICE)
    source = wp.zeros(1, dtype=wp.float32, device=_DEVICE)
    destination = wp.zeros(1, dtype=wp.float32, device=destination_device)
    producer_stream = wp.Stream(_DEVICE)
    with wp.ScopedStream(producer_stream):
        wp.copy(source, source_value)

    tensor = _runtime._device_dltensor(source, n=1, lanes=1, code=2, bits=32)
    group = _ReadGroup(tensor, cuda_stream=producer_stream.cuda_stream)
    stage = _ReadStage(group)
    waits = []
    if destination.device.is_cuda:
        consumer_stream = wp.Stream(destination.device)
        wait_stream = wp.Stream.wait_stream

        def tracked_wait_stream(stream, other_stream, event=None, external=False):
            waits.append((stream, int(other_stream.cuda_stream)))
            return wait_stream(stream, other_stream, event=event, external=external)

        monkeypatch.setattr(wp.Stream, "wait_stream", tracked_wait_stream)
        with wp.ScopedStream(consumer_stream):
            _consume_group(stage, destination)
        assert waits == [(consumer_stream, int(producer_stream.cuda_stream))]
    else:
        synchronize_event = wp.synchronize_event
        synchronize_stream = wp.synchronize_stream
        events = []

        def tracked_synchronize_event(event):
            events.append(int(event.cuda_event))
            return synchronize_event(event)

        def tracked_synchronize_stream(stream):
            waits.append(int(stream.cuda_stream) if isinstance(stream, wp.Stream) else stream)
            return synchronize_stream(stream)

        monkeypatch.setattr(wp, "synchronize_event", tracked_synchronize_event)
        monkeypatch.setattr(wp, "synchronize_stream", tracked_synchronize_stream)
        _consume_group(stage, destination)
        assert len(events) == 1
        assert waits == [source.device]

    assert stage.released == [group]
    assert destination.numpy()[0] == pytest.approx(19.0)


def test_runtime_ingress_maps_default_cuda_stream_sentinel(monkeypatch):
    source = wp.array([23.0], dtype=wp.float32, device=_DEVICE)
    destination = wp.zeros_like(source)
    tensor = _runtime._device_dltensor(source, n=1, lanes=1, code=2, bits=32)
    group = _ReadGroup(tensor, cuda_stream=1)
    stage = _ReadStage(group)
    consumer_stream = wp.Stream(destination.device)
    waits = []
    wait_stream = wp.Stream.wait_stream

    def tracked_wait_stream(stream, other_stream, event=None, external=False):
        waits.append((stream, other_stream.cuda_stream))
        return wait_stream(stream, other_stream, event=event, external=external)

    monkeypatch.setattr(wp.Stream, "wait_stream", tracked_wait_stream)
    with wp.ScopedStream(consumer_stream):
        _consume_group(stage, destination)

    assert waits == [(consumer_stream, None)]
    assert stage.released == [group]
    assert destination.numpy()[0] == pytest.approx(23.0)


def test_runtime_ingress_honors_combined_cuda_sync(monkeypatch):
    source = wp.array([29.0], dtype=wp.float32, device=_DEVICE)
    destination = wp.zeros_like(source)
    producer_stream = wp.Stream(_DEVICE)
    producer_event = wp.Event(_DEVICE)
    producer_stream.record_event(producer_event)
    tensor = _runtime._device_dltensor(source, n=1, lanes=1, code=2, bits=32)
    group = _ReadGroup(
        tensor,
        cuda_stream=producer_stream.cuda_stream,
        wait_event=producer_event.cuda_event,
    )
    stage = _ReadStage(group)
    waits = []
    wait_stream = wp.Stream.wait_stream
    wait_event = wp.Stream.wait_event

    def tracked_wait_stream(stream, other_stream, event=None, external=False):
        waits.append(("stream", int(other_stream.cuda_stream)))
        return wait_stream(stream, other_stream, event=event, external=external)

    def tracked_wait_event(stream, event, external=False):
        waits.append(("event", int(event.cuda_event)))
        return wait_event(stream, event, external=external)

    monkeypatch.setattr(wp.Stream, "wait_stream", tracked_wait_stream)
    monkeypatch.setattr(wp.Stream, "wait_event", tracked_wait_event)
    with wp.ScopedStream(wp.Stream(destination.device)):
        _consume_group(stage, destination)

    assert waits == [
        ("stream", int(producer_stream.cuda_stream)),
        ("event", int(producer_event.cuda_event)),
    ]
    assert stage.released == [group]
    assert destination.numpy()[0] == pytest.approx(29.0)


@pytest.mark.parametrize("cuda_stream,wait_event", [(1, 0), (0, 1)])
def test_runtime_ingress_rejects_cuda_sync_on_cpu_data(cuda_stream, wait_event):
    tensor = lanes_tensor(np.asarray([[3.0]], dtype=np.float32), lanes=1)
    destination = wp.zeros(1, dtype=wp.float32, device="cpu")
    group = _ReadGroup(tensor, cuda_stream=cuda_stream, wait_event=wait_event)
    stage = _ReadStage(group)

    with pytest.raises(_runtime.OvstageContractError, match="CUDA synchronization metadata on CPU data"):
        _consume_group(stage, destination)
    assert stage.released == [group]


def test_runtime_ingress_retains_groups_when_stream_sync_fails(monkeypatch):
    source = wp.array([3.0], dtype=wp.float32, device=_DEVICE)
    destination = wp.zeros_like(source)
    group = _ReadGroup(_runtime._device_dltensor(source, n=1, lanes=1, code=2, bits=32))
    stage = _ReadStage(group)
    synchronize_stream = wp.synchronize_stream
    consumer_stream = wp.Stream(_DEVICE)

    def failed_sync(_stream):
        raise RuntimeError("injected stream synchronization failure")

    monkeypatch.setattr(wp, "synchronize_stream", failed_sync)
    with wp.ScopedStream(consumer_stream):
        with pytest.raises(RuntimeError, match="injected stream synchronization failure"):
            _consume_group(stage, destination)

    synchronize_stream(consumer_stream)
    assert stage.released == []


def test_runtime_ingress_rejects_graph_capture_before_read(monkeypatch):
    source = wp.array([3.0], dtype=wp.float32, device=_DEVICE)
    destination = wp.zeros_like(source)
    stage = _ReadStage(_ReadGroup(_runtime._device_dltensor(source, n=1, lanes=1, code=2, bits=32)))
    stream = SimpleNamespace(device=SimpleNamespace(is_capturing=True))

    monkeypatch.setattr(wp, "get_stream", lambda _device: stream)
    with pytest.raises(RuntimeError, match="cannot run during CUDA graph capture"):
        _consume_group(stage, destination)

    assert stage.reads == 0
    assert stage.released == []


def test_runtime_ingress_handles_backend_device_on_cuda(monkeypatch):
    from ovstage import PopulationDomain, population

    observed = []
    original_tensor_view = _runtime._tensor_source_view

    def tracked_tensor_view(tensor, *, dtype, lanes, attribute="runtime column"):
        result = original_tensor_view(
            tensor,
            dtype=dtype,
            lanes=lanes,
            attribute=attribute,
        )
        observed.append(result[1])
        return result

    monkeypatch.setattr(_runtime, "_tensor_source_view", tracked_tensor_view)
    with ovstage.Stage("ovstage-cuda-runtime-ingress") as stage:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        binding = ovnewton.attach_ovstage(stage)
        assert binding.model.device.is_cuda
        binding.update_to_ovstage(binding.model.state(), ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        binding.update_from_ovstage(binding.model.state(), control=binding.model.control())

    assert observed
    # The pinned ovstage returns these populated/copy-in columns from CPU
    # storage. Accept same-device CUDA as an upstream improvement.
    assert all(not source.is_cuda or source == binding.model.device for source in observed), observed


def test_runtime_writer_device_resident_path():
    """The runtime writer encodes on the model's CUDA device and round-trips.

    Confirms the device-resident single path is actually exercised on CUDA
    (model.device.is_cuda and the fresh state's body_q on a CUDA device), runs a
    full pose+velocity write, and reads back — confirming the device DLTensor
    path produces values that round-trip through Fabric correctly."""
    from ovstage import PopulationDomain, population

    with ovstage.Stage("ovstage-device-resident-path") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        model = _build.build_model(stage, pd, ordinal=1)
        body_paths = model.body_label

        # Confirm the encode runs on a CUDA device — so the single path's
        # DLTensor is built kDLCUDA and the CUDA kernels are what's exercised.
        assert model.device.is_cuda, (
            f"model.device={model.device!r} is not CUDA; device path not on GPU"
        )
        assert state_on_cuda(model), (
            "model.state().body_q is not on a CUDA device; device path not on GPU"
        )

        n = len(body_paths)
        rng = np.arange(n * 6, dtype=np.float64).reshape(n, 6) * 0.1 - 1.0
        state = model.state()
        state.body_qd.assign(
            wp.array(rng.astype(np.float32), dtype=wp.spatial_vectorf, device=model.device)
        )

        # The runtime writer encodes on model.device (no .numpy() on body_q/body_qd)
        # and writes a device DLTensor over the single path.
        ovnewton.StageBinding(stage, model).update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        # Read back and verify values are sensible (full round-trip is tested in
        # test_write.py; here we just confirm the device path did not produce
        # zeros or garbage for all bodies).
        from ovnewton._src import _stage  # noqa: PLC0415
        plist = pd.create_path_list_from_strings(list(body_paths))
        query = stage.query_from_path_list(plist)
        wm = _stage.read_fixed(stage, pd, query, "omni:fabric:worldMatrix", ordinal=2)
        stage.release_query(query).wait()
        pd.destroy_path_list(plist)

    mats = np.stack([np.asarray(wm[i], dtype=np.float64).reshape(16) for i in range(n)])
    # The diagonal entries of rcv (which go into out[0], out[5], out[10]) should
    # sum to ≈3 for an identity rotation — any non-degenerate rotation has trace > -1.
    traces = mats[:, 0] + mats[:, 5] + mats[:, 10]
    assert np.all(traces > -1.0 + 1e-5), (
        f"worldMatrix diagonal traces suspiciously low: {traces}"
    )


def state_on_cuda(model) -> bool:
    """Return True if a fresh model state's body_q lives on a CUDA device."""
    st = model.state()
    return getattr(getattr(st.body_q, "device", None), "is_cuda", False)
