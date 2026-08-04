# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ovnewton — connect the Newton physics engine to ovstage.

Public entry point: :func:`attach_ovstage`. It either builds a Newton
:class:`newton.Model` from a populated ovstage (model construction lives in
:mod:`._build`, with scene parsing in :mod:`._parse`) or binds a caller-supplied
model, and returns a :class:`StageBinding` that publishes stepped state back to
the stage, resyncs from it, or exposes selected Newton state (:mod:`._runtime`).
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

from ._schema_names import JOINT_LOCAL_POS_0, WORLD_MATRIX


@dataclass(frozen=True)
class ReadGroupMeta:
    """Metadata attached to a generated output group."""

    attribute_write_floor_ordinal: int = 0
    layout_generation: int = 0


@dataclass(frozen=True)
class ReadGroupCudaSync:
    """CUDA dependency attached to a generated output group."""

    stream: int = 0
    wait_event: int = 0


def _destroy_path_lists(path_dictionary: Any, path_lists: Tuple[int, ...]) -> None:
    for path_list in path_lists:
        try:
            path_dictionary.destroy_path_list(path_list)
        except Exception:
            pass


def _finalize_path_lists(owner: Any, path_dictionary: Any, path_lists: Iterable[int]) -> None:
    weakref.finalize(owner, _destroy_path_lists, path_dictionary, tuple(path_lists))


class Query:
    """Reusable selection of bound Newton bodies and joints.

    Create this object with :meth:`StageBinding.query`. ``attributes`` lists the
    fields that can be read, and ``prim_count`` is the number of selected bodies
    and joints.
    """

    def __init__(
        self,
        binding: Any,
        *,
        body_index_tensor: Any,
        body_indices_host: Optional[Tuple[int, ...]],
        body_prim_list: int,
        body_count: int,
        joint_q: Any,
        joint_qd: Any,
        attributes: Tuple[int, ...],
        prim_count: int,
        path_lists: Tuple[int, ...],
    ) -> None:
        self.attributes = attributes
        self.prim_count = prim_count
        self._binding_ref = weakref.ref(binding)
        self._body_index_tensor = body_index_tensor
        self._body_indices_host = body_indices_host
        self._body_prim_list = body_prim_list
        self._body_count = body_count
        self._joint_q = joint_q
        self._joint_qd = joint_qd
        _finalize_path_lists(self, binding._pd, path_lists)


class _ReadStorage:
    """Objects that must outlive every view of one read."""

    def __init__(
        self,
        query: Query,
        *,
        event: Any = None,
        keepalive: Sequence[Any] = (),
    ) -> None:
        self.query = query
        self.event = event
        self.keepalive = tuple(keepalive)
        self.cuda_sync = ReadGroupCudaSync(
            wait_event=int(event.cuda_event) if event is not None else 0
        )


class ReadGroup:
    """Read-only Newton output group following the ovstage read protocol.

    ``attribute`` identifies the Newton state field. ``data_row_index(i)`` maps
    path ``i`` in ``prim_list`` to its row in the tensor.

    Keeping the group alive also keeps its data alive. On CUDA,
    ``cuda_sync.wait_event`` identifies when the data is ready.
    """

    ordinal = 0
    is_delete = False
    prim_offset = 0
    has_prim_index_map = False
    has_data_index_map = False
    semantic = 0
    meta = ReadGroupMeta()

    def __init__(
        self,
        storage: _ReadStorage,
        *,
        attribute: int,
        prim_list: int,
        prim_count: int,
        tensors: Tuple[Any, ...],
        is_array: bool,
        data_indices: Optional[Tuple[int, ...]] = None,
        data_index_tensor: Any = None,
    ) -> None:
        self.attribute = attribute
        self.is_array = is_array
        self.prim_list = prim_list
        self.prim_count = prim_count
        self.tensor_count = len(tensors)
        self.data_count = prim_count
        self.has_data_index_map = data_indices is not None
        self.cuda_sync = storage.cuda_sync
        self._tensors = tensors
        self._data_indices = data_indices
        self._data_index_tensor = data_index_tensor
        self._storage = storage

    def prim_index(self, local: int) -> int:
        """Return the prim index for row ``local``."""
        if not 0 <= local < self.prim_count:
            raise IndexError(f"prim index {local} out of range [0, {self.prim_count})")
        return local

    def data_row_index(self, local: int) -> int:
        """Return the tensor row for prim ``local``."""
        if not 0 <= local < self.data_count:
            raise IndexError(f"data row index {local} out of range [0, {self.data_count})")
        return local if self._data_indices is None else self._data_indices[local]

    def tensor(self, index: int) -> Any:
        """Return the DLTensor at ``index``."""
        if not 0 <= index < self.tensor_count:
            raise IndexError(f"tensor index {index} out of range [0, {self.tensor_count})")
        return self._tensors[index]

    def array(self, index: int):
        """Return a read-only NumPy view of a CPU tensor without copying it."""
        from ovstage import dltensor_to_numpy  # noqa: PLC0415

        result = dltensor_to_numpy(self.tensor(index))
        result.setflags(write=False)
        return result

    def dlpack(self, index: int, *, readonly: bool = True) -> Any:
        """Return a read-only DLPack view and keep its source data alive."""
        from ovstage import ManagedDLTensor  # noqa: PLC0415

        tensor = self.tensor(index)
        if not readonly:
            raise ValueError("Newton output groups are read-only")
        return ManagedDLTensor(tensor, manager_ctx=self, readonly=True)

    def data_index_tensor(self) -> Any:
        """Return the optional raw DLTensor row map."""
        return self._data_index_tensor

    def data_index_array(self):
        """Return the optional read-only CPU row map."""
        from ovstage import dltensor_to_numpy  # noqa: PLC0415

        if self._data_index_tensor is None:
            return None
        result = dltensor_to_numpy(self._data_index_tensor)
        result.setflags(write=False)
        return result

    def data_index_dlpack(self, *, readonly: bool = True) -> Any:
        """Return the optional read-only DLPack row map."""
        from ovstage import ManagedDLTensor  # noqa: PLC0415

        if self._data_index_tensor is None:
            return None
        if not readonly:
            raise ValueError("Newton output groups are read-only")
        return ManagedDLTensor(self._data_index_tensor, manager_ctx=self, readonly=True)


class ReadResult:
    """Read-only output groups from one read.

    A result may refer to arrays from the state passed to
    :meth:`StageBinding.read`; it is not always a separate snapshot. Results do
    not need to be closed or released.
    """

    def __init__(self, groups: Sequence[ReadGroup]) -> None:
        self.groups = tuple(groups)


def _validate_labels(stage, pd, labels, ordinal):
    """Raise unless every body label resolves to a world-matrix row."""
    if not labels:
        return
    from . import _stage  # noqa: PLC0415

    with _stage._path_list_query(stage, pd, list(labels)) as query:
        wm = _stage.read_fixed(stage, pd, query, WORLD_MATRIX, ordinal)
    missing = [label for i, label in enumerate(labels) if i not in wm]
    if missing:
        raise ValueError("body labels did not all resolve in the ovstage — model not built "
                         "from the same USD? missing labels: %s" % missing[:3])


def _validate_joint_labels(stage, pd, model, labels, ordinal):
    """Raise unless every absolute joint label resolves exactly once."""
    import newton  # noqa: PLC0415

    joint_types = model.joint_type.numpy()
    no_runtime_path = {int(newton.JointType.FREE), int(newton.JointType.FIXED)}
    invalid = [
        label
        for i, label in enumerate(labels)
        if int(joint_types[i]) not in no_runtime_path
        and (not isinstance(label, str) or not label.startswith("/"))
    ]
    if invalid:
        raise ValueError("non-fixed joint labels must be absolute ovstage prim paths")
    absolute = [label for i, label in enumerate(labels)
                if isinstance(label, str) and label.startswith("/")
                and int(joint_types[i]) != int(newton.JointType.FREE)]
    if len(absolute) != len(set(absolute)):
        raise ValueError("joint labels contain duplicate prim paths")
    if not absolute:
        return
    from . import _stage  # noqa: PLC0415

    with _stage._path_list_query(stage, pd, absolute) as query:
        local_pos = _stage.read_fixed(stage, pd, query, JOINT_LOCAL_POS_0, ordinal)
    missing = [label for i, label in enumerate(absolute) if i not in local_pos]
    if missing:
        raise ValueError("joint labels did not all resolve in the ovstage — model not built "
                         "from the same USD? missing labels: %s" % missing[:3])


class StageBinding:
    """Correspondence between a Newton model and an ovstage, via the model's
    per-index `body_label` / `joint_label` (= prim paths). Publishes stepped state
    to the stage, resyncs it, and exposes selected native state. Stage updates
    are serialized because they reuse device buffers. Output reads may borrow
    the supplied state; callers must order state mutation after consumers.
    Recreate the binding after changing stage topology or runtime layouts."""

    def __init__(self, stage, model, *, ordinal: int = 1):
        import ovstage  # noqa: PLC0415

        body_labels = list(model.body_label)
        if len(body_labels) != model.body_count:
            raise ValueError(f"expected {model.body_count} body labels, got {len(body_labels)}")
        joint_labels = list(model.joint_label)
        if len(joint_labels) != model.joint_count:
            raise ValueError(f"expected {model.joint_count} joint labels, got {len(joint_labels)}")

        from . import _runtime, _stage  # noqa: PLC0415

        ordinal = _stage._resolve_read_ceiling(stage, ordinal)
        self._stage = stage
        self._pd = ovstage.PathDictionary(stage)
        self._runtime_lock = threading.Lock()
        _validate_labels(stage, self._pd, body_labels, ordinal)
        _validate_joint_labels(stage, self._pd, model, joint_labels, ordinal)
        self._runtime = _runtime._build_runtime_transport(
            model,
            body_labels,
            joint_labels,
        )
        _runtime._capture_runtime_read_layout(stage, self._pd, self._runtime, ordinal)
        _runtime._derive_joint_state(self._runtime, model, model.joint_q, model.joint_qd)
        _runtime.initialize_output(self)

    @property
    def model(self):
        """The connected :class:`newton.Model`."""
        return self._runtime.model

    def query(
        self,
        stage_query: Any = None,
        *,
        paths: Optional[Sequence[str]] = None,
    ) -> Query:
        """Prepare a reusable body/joint selection from a stage query or paths.

        Pass either ``stage_query`` or ``paths``, but not both. Paths must be
        unique absolute stage paths. Read results keep body order and preserve
        joint order within each returned group. Joints with different numbers
        of values are returned in separate groups. A stage query may also match
        unrelated prims; those prims are ignored. The returned selection remains
        valid after the stage query is closed.

        Create a new selection after changing the model or stage structure.
        """
        from . import _runtime  # noqa: PLC0415

        with self._runtime_lock:
            return _runtime.create_query(self, stage_query, paths=paths)

    def read(
        self,
        state: Any,
        *,
        query: Optional[Query] = None,
        attributes: Sequence[str | int],
    ) -> ReadResult:
        """Expose selected Newton state arrays as read-only groups.

        ``state`` must provide the requested arrays. Their sizes must match the
        connected model, and they must use the same device. If ``query`` is
        omitted, the read includes every available body and joint.
        ``attributes`` accepts ``body_q``, ``body_qd``, ``joint_q``, and
        ``joint_qd`` as names or token IDs.

        Body groups and compatible single-width joint groups refer directly to
        arrays from ``state`` and may provide a row map. Other joint selections
        are gathered once per requested array and split by coordinate width.
        Use ``data_row_index(i)`` or ``data_index_dlpack()`` to locate logical
        prim ``i``. Do not mutate borrowed state until every reader has
        finished.

        CPU data is ready when this method returns. CUDA work does not block the
        CPU; a reader on another CUDA stream must wait for
        ``group.cuda_sync.wait_event``.
        """
        from . import _runtime  # noqa: PLC0415

        return _runtime.read_output(self, state, query=query, attributes=attributes)

    def update_to_ovstage(self, state, *, ordinal):
        """Write the current Newton state to ovstage at ``ordinal``.

        This writes body poses, body velocities, and supported one-axis joint
        positions and velocities. The writes finish before this method returns.
        Call ``stage.advance_write_floor(ordinal)`` to make them visible to
        readers.

        Raises :class:`ValueError` if an array has the wrong size or device.
        """
        from . import _runtime  # noqa: PLC0415

        _runtime.publish_output(self, state, ordinal)

    def update_from_ovstage(self, state, control, *, ordinal=None):
        """Apply all available state and drive targets from ovstage.

        If ``ordinal`` is omitted, this reads the latest values made visible by
        the stage. Otherwise, it reads values at or below that ordinal. The
        method reads all supported state and drive fields; callers cannot select
        a subset.

        Raises :class:`ovnewton.OvstageContractError` if the stage cannot provide
        one consistent set of values.
        """
        from . import _runtime, _stage  # noqa: PLC0415

        with self._runtime_lock:
            read_ordinal = _stage._resolve_read_ceiling(self._stage, ordinal)
            _runtime.read_runtime(
                self._stage,
                self._pd,
                ordinal=read_ordinal,
                state=state,
                runtime=self._runtime,
                control=control,
            )


def attach_ovstage(ovstage_stage, *, model=None, ordinal: int = 1):
    """Bind a Newton model to an ovstage.

    `model=None` builds the model from the ovstage — bodies, mass, joints and
    gravity taken from the populated stage (primitive and triangle-mesh
    colliders are supported).
    `model=<newton.Model>` skips the parse and binds the supplied model, using its
    `body_label` / `joint_label` as the body / joint -> prim-path mapping (the model
    must have been built from the same USD that populated this ovstage).
    `ordinal` bounds the populated payloads used to build or validate the binding
    and defaults to the conventional initial population ordinal, 1. Ovstage does
    not retain historical payloads. Returns a `StageBinding`."""
    if model is not None:
        return StageBinding(ovstage_stage, model, ordinal=ordinal)
    import ovstage  # noqa: PLC0415

    from . import _build  # noqa: PLC0415

    with ovstage.PathDictionary(ovstage_stage) as pd:
        model = _build.build_model(ovstage_stage, pd, ordinal=ordinal)
    return StageBinding(ovstage_stage, model, ordinal=ordinal)
