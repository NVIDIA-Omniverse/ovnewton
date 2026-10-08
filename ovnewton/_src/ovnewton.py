# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ovnewton — connect the Newton physics engine to ovstage.

Public entry points: :func:`add_ovstage` populates a caller-owned Newton
:class:`newton.ModelBuilder`; :func:`attach_ovstage` binds a finalized model and
returns a :class:`StageBinding` that publishes stepped state back to the stage,
resyncs from it, or exposes selected Newton output (:mod:`._runtime`).
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import newton
    import numpy as np
    import numpy.typing as npt
    import ovstage
    import warp as wp

    from ._runtime import _JointOutputSelection

from ._schema_names import JOINT_LOCAL_POS_0, WORLD_MATRIX


class SimObjectType(IntEnum):
    """Physics object type selected by an output query."""

    RIGID_BODY = 0
    ARTICULATION_LINK = 1
    ARTICULATION_JOINT = 2
    ARTICULATION = 9
    PHYSICS_SCENE = 11


class ObjectScope(IntEnum):
    """Membership rule for an output query."""

    ALL = 0
    ACTIVE = 1


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


@dataclass(frozen=True)
class OvstageImportResult:
    """Information needed to finalize a builder populated from ovstage.

    ``has_orphan_joints`` is true when the USD intentionally leaves joints
    outside an articulation. Preserve that topology with
    ``builder.finalize(skip_validation_joints=result.has_orphan_joints)``.
    """

    has_orphan_joints: bool = False


def _destroy_path_lists(path_dictionary: Any, path_lists: Tuple[int, ...]) -> None:
    for path_list in path_lists:
        try:
            path_dictionary.destroy_path_list(path_list)
        except Exception:
            pass


def _finalize_path_lists(owner: Any, path_dictionary: Any, path_lists: Iterable[int]) -> None:
    weakref.finalize(owner, _destroy_path_lists, path_dictionary, tuple(path_lists))


class Query:
    """Reusable selection of bound Newton scene objects.

    This object is returned by :meth:`StageBinding.query`. ``attributes`` lists
    the output fields that can be read, and ``prim_count`` is the number
    of selected bodies, articulation roots, joints, and the effective physics scene.
    """

    attributes: tuple[int, ...]
    prim_count: int
    object_type: SimObjectType | None
    scope: ObjectScope

    def __init__(
        self,
        binding: StageBinding,
        *,
        body_index_array: wp.array | None,
        body_index_tensor: ovstage.DLTensor | None,
        body_indices_host: Optional[Tuple[int, ...]],
        body_prim_list: int,
        body_count: int,
        scene_prim_list: int,
        scene_count: int,
        shape_counts: wp.array | None,
        shape_indices: wp.array | None,
        shape_width: int,
        articulation_index_array: wp.array | None,
        articulation_prim_list: int,
        articulation_count: int,
        joint_q: _JointOutputSelection | None,
        joint_qd: _JointOutputSelection | None,
        joint_dof: _JointOutputSelection | None,
        joint_position: _JointOutputSelection | None,
        joint_velocity: _JointOutputSelection | None,
        joint_position_target: _JointOutputSelection | None,
        attributes: Tuple[int, ...],
        prim_count: int,
        object_type: SimObjectType | None,
        scope: ObjectScope,
        path_lists: Tuple[int, ...],
    ) -> None:
        self.attributes = attributes
        self.prim_count = prim_count
        self.object_type = object_type
        self.scope = scope
        self._binding_ref = weakref.ref(binding)
        self._body_index_array = body_index_array
        self._body_index_tensor = body_index_tensor
        self._body_indices_host = body_indices_host
        self._body_prim_list = body_prim_list
        self._body_count = body_count
        self._scene_prim_list = scene_prim_list
        self._scene_count = scene_count
        self._shape_counts = shape_counts
        self._shape_indices = shape_indices
        self._shape_width = shape_width
        self._articulation_index_array = articulation_index_array
        self._articulation_prim_list = articulation_prim_list
        self._articulation_count = articulation_count
        self._joint_q = joint_q
        self._joint_qd = joint_qd
        self._joint_dof = joint_dof
        self._joint_position = joint_position
        self._joint_velocity = joint_velocity
        self._joint_position_target = joint_position_target
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

    This object is returned in :attr:`ReadResult.groups` and is not constructed
    directly.

    ``attribute`` identifies the Newton output field. ``data_row_index(i)`` maps
    path ``i`` in ``prim_list`` to its row in the tensor.

    ``object_type`` is copied from the query. It is ``None`` for a query that
    combines more than one object type.

    Keeping the group alive also keeps its data alive. On CUDA,
    ``cuda_sync.wait_event`` identifies when the data is ready.
    """

    attribute: int
    object_type: SimObjectType | None
    is_array: bool
    prim_list: int
    prim_count: int
    tensor_count: int
    data_count: int
    cuda_sync: ReadGroupCudaSync
    ordinal: int = 0
    is_delete: bool = False
    prim_offset: int = 0
    has_prim_index_map: bool = False
    has_data_index_map: bool = False
    semantic: int = 0
    meta: ReadGroupMeta = ReadGroupMeta()

    def __init__(
        self,
        storage: _ReadStorage,
        *,
        attribute: int,
        prim_list: int,
        prim_count: int,
        tensors: tuple[ovstage.DLTensor, ...],
        is_array: bool,
        data_indices: Optional[Tuple[int, ...]] = None,
        data_index_tensor: ovstage.DLTensor | None = None,
    ) -> None:
        self.attribute = attribute
        self.object_type = storage.query.object_type
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

    def tensor(self, index: int) -> ovstage.DLTensor:
        """Return the DLTensor at ``index``."""
        if not 0 <= index < self.tensor_count:
            raise IndexError(f"tensor index {index} out of range [0, {self.tensor_count})")
        return self._tensors[index]

    def array(self, index: int) -> npt.NDArray[np.generic]:
        """Return a read-only NumPy view of a CPU tensor without copying it."""
        from ovstage import dltensor_to_numpy  # noqa: PLC0415

        result = dltensor_to_numpy(self.tensor(index))
        result.setflags(write=False)
        return result

    def dlpack(self, index: int, *, readonly: bool = True) -> ovstage.ManagedDLTensor:
        """Return a read-only DLPack view and keep its source data alive."""
        from ovstage import ManagedDLTensor  # noqa: PLC0415

        tensor = self.tensor(index)
        if not readonly:
            raise ValueError("Newton output groups are read-only")
        return ManagedDLTensor(tensor, manager_ctx=self, readonly=True)

    def data_index_tensor(self) -> ovstage.DLTensor | None:
        """Return the optional raw DLTensor row map."""
        return self._data_index_tensor

    def data_index_array(self) -> npt.NDArray[np.generic] | None:
        """Return the optional read-only CPU row map."""
        from ovstage import dltensor_to_numpy  # noqa: PLC0415

        if self._data_index_tensor is None:
            return None
        result = dltensor_to_numpy(self._data_index_tensor)
        result.setflags(write=False)
        return result

    def data_index_dlpack(self, *, readonly: bool = True) -> ovstage.ManagedDLTensor | None:
        """Return the optional read-only DLPack row map."""
        from ovstage import ManagedDLTensor  # noqa: PLC0415

        if self._data_index_tensor is None:
            return None
        if not readonly:
            raise ValueError("Newton output groups are read-only")
        return ManagedDLTensor(self._data_index_tensor, manager_ctx=self, readonly=True)


class ReadResult:
    """Read-only output groups from one read.

    A result may refer to arrays from the state or model used by
    :meth:`StageBinding.read` instead of a copy. Its metadata does not identify
    a simulation step. Results do not need to be closed or released. This object
    is returned by :meth:`StageBinding.read` and is not constructed directly.
    """

    groups: tuple[ReadGroup, ...]

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
    """Connect a Newton model to an ovstage through the model's body and joint labels.

    Writes simulation state to ovstage, reads stage changes into Newton, and lets
    applications read selected Newton state or model arrays. Writes run one at a
    time because they reuse buffers. A read may borrow arrays from its supplied
    state or from the bound model; do not change those arrays until every reader
    has finished. Recreate the binding after changing the model or stage
    structure.
    """

    def __init__(self, stage: ovstage.Stage, model: newton.Model, *, ordinal: int = 1) -> None:
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
        _runtime._capture_body_scales(stage, self._pd, self._runtime, body_labels, ordinal)
        _runtime._capture_runtime_read_layout(stage, self._pd, self._runtime, ordinal)
        _runtime._derive_joint_state(self._runtime, model, model.joint_q, model.joint_qd)
        _runtime.initialize_output(self, ordinal)

    @property
    def model(self) -> newton.Model:
        """The connected :class:`newton.Model`."""
        return self._runtime.model

    def query(
        self,
        stage_query: ovstage.Query | None = None,
        *,
        paths: Optional[Sequence[str]] = None,
        object_type: SimObjectType | None = None,
        scope: ObjectScope = ObjectScope.ALL,
    ) -> Query:
        """Prepare a reusable output selection from a stage query or paths.

        Pass either ``stage_query`` or ``paths``, but not both. When
        ``object_type`` is provided, both may be omitted to select every object
        of that type. Paths must be unique absolute paths. Bound bodies,
        articulation roots, the effective physics scene, and authored joints
        use stage prim paths; generated Newton joints may use model-only labels.
        Read results keep body order and preserve joint order within each
        returned group. Joints with different numbers of values are returned in
        separate groups. A stage query may also match unrelated prims; those
        prims are ignored. ``ObjectScope.ALL`` is supported. Active and
        changed-since membership are not yet supported. The returned selection
        remains valid after the stage query is closed.

        Create a new selection after changing the model or stage structure.
        """
        from . import _runtime  # noqa: PLC0415

        with self._runtime_lock:
            return _runtime.create_query(
                self,
                stage_query,
                paths=paths,
                object_type=object_type,
                scope=scope,
            )

    def read(
        self,
        state: newton.State | None = None,
        *,
        control: newton.Control | None = None,
        query: Optional[Query] = None,
        attributes: Sequence[str | int],
    ) -> ReadResult:
        """Expose selected Newton output arrays as read-only groups.

        ``state`` must provide any requested state arrays. Their sizes must
        match the connected model, and they must use the same device. It may be
        omitted when only model attributes are requested. ``control`` supplies
        current joint targets and is required only when targets are requested.
        If ``query`` is omitted, the read includes every available output object.
        ``attributes`` accepts the body pose, velocity, optional acceleration,
        mass-property, shape-material, articulation-root, and scene fields
        documented in :doc:`output-reads`, plus shared per-axis joint state,
        targets, properties, and the Newton-native ``joint_q`` and ``joint_qd``
        fields. Names or token IDs are accepted. Native fields expose their
        Newton arrays directly. Split and converted fields are produced on the
        model device.

        Native body groups and compatible single-width joint groups refer
        directly to arrays from ``state`` or the connected model and may
        provide a row map. Other groups use device-side extraction or gathers.
        Use ``data_row_index(i)`` or ``data_index_dlpack()`` to locate logical
        prim ``i``. Do not mutate borrowed state or model arrays until every
        reader has finished.

        CPU data is ready when this method returns. On CUDA, call this method
        on the Warp stream that last wrote the requested state or model arrays,
        or first make the current stream wait for that work. CUDA work does
        not block the CPU; a reader on another CUDA stream must wait for
        ``group.cuda_sync.wait_event``.
        """
        from . import _runtime  # noqa: PLC0415

        return _runtime.read_output(
            self,
            state,
            control,
            query=query,
            attributes=attributes,
        )

    def update_to_ovstage(self, state: newton.State, *, ordinal: int) -> None:
        """Write the current Newton state to ovstage at ``ordinal``.

        Body world poses are written through ``omni:xform`` with
        ``omni:resetXformStack`` enabled. This also writes body velocities and
        supported one-axis joint positions and velocities. The writes finish
        before this method returns. Call
        ``stage.advance_write_floor(ordinal)`` to make them visible to readers.

        Raises :class:`ValueError` if an array has the wrong size or device.
        """
        from . import _runtime  # noqa: PLC0415

        _runtime.publish_output(self, state, ordinal)

    def update_from_ovstage(
        self,
        state: newton.State,
        control: newton.Control,
        *,
        ordinal: int | None = None,
    ) -> None:
        """Apply all available state and drive targets from ovstage.

        Body poses include transforms authored on ancestor prims.
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


def add_ovstage(
    builder: newton.ModelBuilder,
    ovstage_stage: ovstage.Stage,
    *,
    ordinal: int = 1,
) -> OvstageImportResult:
    """Add one populated ovstage's model data to a caller-owned Newton builder.

    The current implementation supports one ovstage per builder. ``builder``
    must not contain model entities, but its defaults and custom attribute
    definitions may be configured. This function applies the stage's up axis
    and gravity, adds the stage-backed entities, and returns without finalizing
    the builder or creating a :class:`StageBinding`. Callers may perform
    solver-specific preparation afterwards.

    Supported surface cloth retains its publication mappings through Newton's
    custom model attributes. For VBD cloth, call
    ``builder.color(include_bending=True)`` before finalization, then bind the
    finalized model with ``attach_ovstage(stage, model=model)``.

    ``ordinal`` bounds the populated payloads read from the stage. The returned
    metadata reports whether preserving rootless joints requires skipping
    Newton's articulation-membership validation during finalization.
    """
    import ovstage  # noqa: PLC0415

    from . import _build  # noqa: PLC0415

    with ovstage.PathDictionary(ovstage_stage) as pd:
        has_orphan_joints = _build.add_ovstage(builder, ovstage_stage, pd, ordinal=ordinal)
    return OvstageImportResult(has_orphan_joints=has_orphan_joints)


def attach_ovstage(
    ovstage_stage: ovstage.Stage,
    *,
    model: newton.Model | None = None,
    ordinal: int = 1,
) -> StageBinding:
    """Create a :class:`StageBinding` for an ovstage and a Newton model.

    When ``model`` is supplied, this function binds it without importing or
    finalizing anything. Its ``body_label`` and ``joint_label`` values must
    match prim paths in the ovstage.

    When ``model`` is ``None``, this function creates a
    :class:`newton.ModelBuilder`, adds the ovstage contents, finalizes the
    model, and binds it. Call :func:`add_ovstage` directly when the application
    needs to configure the builder before finalization.

    ``ordinal`` bounds the populated payloads used to build or validate the
    binding and defaults to the conventional initial population ordinal, 1.
    Ovstage does not retain historical payloads.
    """
    if model is not None:
        return StageBinding(ovstage_stage, model, ordinal=ordinal)
    import ovstage  # noqa: PLC0415

    from . import _build  # noqa: PLC0415

    with ovstage.PathDictionary(ovstage_stage) as pd:
        model = _build.build_model(ovstage_stage, pd, ordinal=ordinal)
    return StageBinding(ovstage_stage, model, ordinal=ordinal)
