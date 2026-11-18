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
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Tuple

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

    #: Rigid bodies outside the articulation-link selection.
    RIGID_BODY = 0
    #: Bodies classified as links in a Newton articulation.
    ARTICULATION_LINK = 1
    #: Articulation joints with readable coordinates/DOFs; excludes free joints.
    ARTICULATION_JOINT = 2
    #: Authored PhysicsArticulationRootAPI or NewtonArticulationRootAPI prims.
    ARTICULATION = 9
    #: The effective PhysicsScene prim, for gravity reads.
    PHYSICS_SCENE = 11


class ObjectScope(IntEnum):
    """Membership rule for an output query."""

    #: All matching bound objects; reusable while the binding topology is fixed.
    ALL = 0
    #: Reserved; requesting this scope raises NotImplementedError.
    ACTIVE = 1


@dataclass(frozen=True)
class ReadColumnMeta:
    """Metadata attached to a generated output column."""

    attribute_write_floor_ordinal: int = 0
    layout_generation: int = 0


@dataclass(frozen=True)
class ReadColumnCudaSync:
    """CUDA dependency attached to a generated output column."""

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


def _destroy_path_lists(path_dictionary: Any, path_lists: Tuple[ovstage.PrimPathList, ...]) -> None:
    for path_list in path_lists:
        try:
            path_dictionary.destroy_path_list(path_list)
        except Exception:
            pass


def _finalize_path_lists(owner: Any, path_dictionary: Any, path_lists: Iterable[ovstage.PrimPathList]) -> None:
    weakref.finalize(owner, _destroy_path_lists, path_dictionary, tuple(path_lists))


class Query:
    """Reusable selection of bound Newton scene objects.

    Returned by :meth:`StageBinding.query`. ``attributes`` contains interned
    name tokens for fields supported by the selected topology; resolve them
    with ``ovstage.PathDictionary.token_to_string()``. It does not check the
    state/control supplied later: optional acceleration buffers, targets or an
    unambiguous gravity-world mapping may still be unavailable at read time.
    ``prim_count`` counts selected bodies, articulation roots, joints and scenes.

    Reuse this selection only with its originating binding and unchanged
    topology, labels, joint axes and shape membership. Structural changes
    require a new binding and queries; no automatic invalidation token is
    provided. See :doc:`output-reads` for field formats and availability.
    """

    attributes: tuple[ovstage.Token, ...]
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
        body_prim_list: Any,
        body_count: int,
        scene_prim_list: Any,
        scene_count: int,
        shape_counts: wp.array | None,
        shape_indices: wp.array | None,
        shape_width: int,
        articulation_index_array: wp.array | None,
        articulation_prim_list: Any,
        articulation_count: int,
        joint_q: _JointOutputSelection | None,
        native_joint_dofs: _JointOutputSelection | None,
        joint_dof: _JointOutputSelection | None,
        joint_position: _JointOutputSelection | None,
        joint_velocity: _JointOutputSelection | None,
        joint_position_target: _JointOutputSelection | None,
        attributes: Tuple[ovstage.Token, ...],
        prim_count: int,
        object_type: SimObjectType | None,
        scope: ObjectScope,
        path_lists: Tuple[ovstage.PrimPathList, ...],
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
        self._native_joint_dofs = native_joint_dofs
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
        self.cuda_sync = ReadColumnCudaSync(
            wait_event=int(event.cuda_event) if event is not None else 0
        )


class ReadColumn:
    """One read-only attribute aligned to its group's logical prim rows.

    This object is returned in :attr:`ReadGroup.columns` and is not constructed
    directly.

    ``attribute`` is an interned field-name token. ``data_row_index(i)`` maps
    logical prim ``i`` in the group to this column's storage row; the tensor can
    include unselected rows. Each column has one packed DLTensor with shape
    ``(storage_rows,)`` and ``dtype.lanes`` scalar components per row. Columns in
    different groups can have the same attribute with different widths. Values
    are float32 except int32 ``shapeCount``; an optional uint32 data-index tensor
    provides the row map on the payload device.

    Keeping the column alive retains its data, query and readiness event, but
    does not freeze borrowed values. Prevent source writes until consumers
    finish. CPU data is ready on return from :meth:`StageBinding.read`; a CUDA
    consumer on another stream must wait for ``cuda_sync.wait_event``. Retain
    the column through asynchronous consumer completion. ``ordinal``, ``semantic``
    and ``meta`` are zero placeholders, not sample or representation identifiers.
    """

    attribute: ovstage.Token
    is_array: bool
    tensor_count: int
    data_count: int
    cuda_sync: ReadColumnCudaSync
    ordinal: int = 0
    is_delete: bool = False
    has_data_index_map: bool = False
    semantic: int = 0
    meta: ReadColumnMeta = ReadColumnMeta()

    def __init__(
        self,
        storage: _ReadStorage,
        *,
        attribute: ovstage.Token,
        data_count: int,
        tensor: ovstage.DLTensor,
        data_indices: Optional[Tuple[int, ...]] = None,
        data_index_tensor: ovstage.DLTensor | None = None,
    ) -> None:
        self.attribute = attribute
        self.is_array = False
        self.tensor_count = 1
        self.data_count = data_count
        self.has_data_index_map = data_indices is not None
        self.cuda_sync = storage.cuda_sync
        self._tensors = (tensor,)
        self._data_indices = data_indices
        self._data_index_tensor = data_index_tensor
        self._storage = storage

    def data_row_index(self, local: int) -> int:
        """Return the tensor row for prim ``local``."""
        if not 0 <= local < self.data_count:
            raise IndexError(f"data row index {local} out of range [0, {self.data_count})")
        return local if self._data_indices is None else self._data_indices[local]

    def tensor(self, index: int) -> ovstage.DLTensor:
        """Return a raw packed DLTensor; retain this column while using it.

        Use :meth:`data_row_index` to map selected prims to storage rows.
        The descriptor's shape counts storage rows, not selected objects.
        """
        if not 0 <= index < self.tensor_count:
            raise IndexError(f"tensor index {index} out of range [0, {self.tensor_count})")
        return self._tensors[index]

    def array(self, index: int) -> npt.NDArray[np.generic]:
        """Return a read-only NumPy view of CPU storage without copying it.

        Lane components are expanded into scalar dimensions. Retain this column
        while using the view and apply :meth:`data_row_index` for selected prims.
        CUDA storage must be consumed on device or explicitly copied to CPU.
        """
        from ovstage import dltensor_to_numpy  # noqa: PLC0415

        result = dltensor_to_numpy(self.tensor(index))
        result.setflags(write=False)
        return result

    def dlpack(self, index: int, *, readonly: bool = True) -> ovstage.ManagedDLTensor:
        """Return a managed DLPack export retaining this column and its storage.

        Only ``readonly=True`` is supported. Treat the export as read-only even
        if a consumer permits writes. Exporting does not copy, normalize values
        or wait for CUDA completion; follow :attr:`cuda_sync` before consuming.
        Borrowed values remain live until the application copies them.
        """
        from ovstage import ManagedDLTensor  # noqa: PLC0415

        tensor = self.tensor(index)
        if not readonly:
            raise ValueError("Newton output columns are read-only")
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
            raise ValueError("Newton output columns are read-only")
        return ManagedDLTensor(self._data_index_tensor, manager_ctx=self, readonly=True)


class ReadGroup:
    """Shared logical prim rows and their read-only attribute columns.

    Resolve ``prim_list`` through the stage's path dictionary. Row ``i`` names
    the same prim in every column, but each column has its own tensor layout
    and ``data_row_index(i)``. Columns are ordered by the requested attributes.
    ``object_type`` is the query's filter, or ``None`` for an unfiltered query.

    Unlike an OVStage read, this result does not pin stage storage and needs no
    explicit release. Keeping a group, column or managed DLPack export alive
    retains the required Newton storage, query and CUDA event. Borrowed values
    remain live; retain results and prevent source writes until consumers finish.
    Construct groups through :meth:`StageBinding.read`, not directly.
    """

    prim_list: ovstage.PrimPathList
    prim_count: int
    object_type: SimObjectType | None
    columns: tuple[ReadColumn, ...]
    prim_offset: int = 0
    has_prim_index_map: bool = False

    def __init__(
        self,
        storage: _ReadStorage,
        *,
        prim_list: ovstage.PrimPathList,
        prim_count: int,
        columns: Sequence[ReadColumn],
    ) -> None:
        self.prim_list = prim_list
        self.prim_count = prim_count
        self.object_type = storage.query.object_type
        self.columns = tuple(columns)
        self._storage = storage

    @property
    def column_count(self) -> int:
        """Number of attribute columns sharing these prim rows."""
        return len(self.columns)

    def prim_index(self, local: int) -> int:
        """Return the prim index for row ``local``."""
        if not 0 <= local < self.prim_count:
            raise IndexError(f"prim index {local} out of range [0, {self.prim_count})")
        return local


class ReadResult:
    """Read-only output groups from one read.

    A result may refer to arrays from the state or model used by
    :meth:`StageBinding.read` instead of a copy. Borrowing depends on the
    selection and layout; derived and gathered columns own per-read buffers.
    Retention preserves allocation lifetime, not a snapshot of borrowed values.
    Keep source arrays unchanged and retain the result until consumers finish.

    Its metadata does not identify a simulation step. Results need no explicit
    close or release. Managed DLPack exports retain their columns independently.
    This object is returned by :meth:`StageBinding.read` and is not constructed
    directly. See :doc:`output-reads` for storage and synchronization rules.
    """

    groups: tuple[ReadGroup, ...]

    def __init__(self, groups: Sequence[ReadGroup]) -> None:
        self.groups = tuple(groups)


def _read_body_matrices(stage, pd, labels, ordinal):
    """Read owned world-matrix rows, requiring every body label to resolve."""
    if not labels:
        return {}
    from . import _stage  # noqa: PLC0415

    with _stage._path_list_query(stage, pd, list(labels)) as query:
        wm = _stage.read_fixed(stage, pd, query, WORLD_MATRIX, ordinal)
    missing = [label for i, label in enumerate(labels) if i not in wm]
    if missing:
        raise ValueError("body labels did not all resolve in the ovstage — model not built "
                         "from the same USD? missing labels: %s" % missing[:3])
    return wm


def _validate_joint_labels(stage, pd, model, labels, ordinal, excluded=()):
    """Raise unless every absolute joint label resolves exactly once."""
    import newton  # noqa: PLC0415

    joint_types = model.joint_type.numpy()
    no_runtime_path = {int(newton.JointType.FREE), int(newton.JointType.FIXED)}
    invalid = [
        label
        for i, label in enumerate(labels)
        if i not in excluded and int(joint_types[i]) not in no_runtime_path
        and (not isinstance(label, str) or not label.startswith("/"))
    ]
    if invalid:
        raise ValueError("non-fixed joint labels must be absolute ovstage prim paths")
    absolute = [label for i, label in enumerate(labels)
                if i not in excluded and isinstance(label, str) and label.startswith("/")
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

        from . import _curves, _runtime, _stage  # noqa: PLC0415

        self._curves = _curves.model_curve_ranges(model)
        generated_bodies = {body for curve in self._curves for body in range(curve.body_start, curve.body_end)}
        generated_joints = frozenset(
            joint for curve in self._curves for joint in range(curve.joint_start, curve.joint_end)
        )
        body_indices = tuple(index for index in range(model.body_count) if index not in generated_bodies)
        body_labels = [body_labels[index] for index in body_indices]
        ordinal = _stage._resolve_read_ceiling(stage, ordinal)
        self._stage = stage
        self._pd = ovstage.PathDictionary(stage)
        self._runtime_lock = threading.Lock()
        body_matrices = _read_body_matrices(stage, self._pd, body_labels, ordinal)
        _validate_joint_labels(stage, self._pd, model, joint_labels, ordinal, generated_joints)
        self._runtime = _runtime._build_runtime_transport(
            model,
            body_labels,
            joint_labels,
            body_indices=body_indices,
            excluded_joints=generated_joints,
        )
        _runtime._capture_body_scales(self._runtime, body_matrices)
        _runtime.initialize_output(self, ordinal)
        _runtime._capture_runtime_read_layout(stage, self._pd, self._runtime, ordinal)
        _runtime._derive_joint_state(self._runtime, model, model.joint_q, model.joint_qd)
        _runtime.initialize_runtime_selection(self)

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
        returned group. Each column has one width; columns with the same ordered
        prim rows share a group. A stage query may also match unrelated prims;
        those prims are ignored. ``ObjectScope.ALL`` is supported. Active and
        changed-since membership are not yet supported. The returned selection
        remains valid after the stage query is closed.

        ``Query.attributes`` lists candidate field-name tokens for this
        topology, not sample-time availability. Optional state/control buffers
        are validated by :meth:`read`. Keep this selection with its originating
        binding. Structural changes require a new binding and new selections;
        no automatic topology-invalidation token is provided.
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
        attributes: Sequence[str | int | ovstage.Token],
    ) -> ReadResult:
        """Expose selected Newton output as prim groups with attribute columns.

        ``state`` must provide any requested state arrays. Their sizes must
        match the connected model, and they must use the same device. It may be
        omitted when only model or control attributes are requested. ``control``
        supplies current joint targets and commanded effort, and is required
        when either is requested for selected joints.
        If ``query`` is omitted, the read includes every available output object.
        ``attributes`` accepts the fields listed in :doc:`read-fields`, which
        :func:`ovnewton.read_fields` also returns as :class:`ReadField`
        declarations: body pose, velocity, optional acceleration,
        mass-property, shape-material, articulation-root and scene fields,
        Newton joint fields such as ``joint_q``, ``joint_qd`` and
        ``joint_f``, and standard per-axis joint fields such as
        ``jointPosition``.
        ``joint_f`` reads commanded generalized forces from ``Control`` in
        velocity-DOF order, not measured solver forces. Names or interned
        token IDs are accepted; duplicate requests are collapsed.
        An unknown field is an error; a known field with no compatible selected
        objects returns no column. An empty attribute list returns an empty
        result. Discovery does not validate optional state buffers; requesting
        unavailable acceleration, for example, raises ``ValueError``. No solver
        work or state reconstruction is performed.

        Newton fields have snake_case names, which are Newton array names.
        They keep Newton's units,
        frames and component order. ``joint_q`` uses generalized coordinates
        (including XYZW quaternions), while ``joint_qd`` uses velocity DOFs;
        their widths can differ. Standard fields have camelCase names. On
        angular joint axes, positions, velocities, targets and limits
        are in degrees or degrees/s, gains are per degree, and efforts are not
        converted; these fields follow the authored body order. Body angular
        velocity stays in radians/s.
        Reads preserve model length units and do not normalize to SI.

        Each group identifies shared prim rows; its columns hold the requested
        attributes, independent row mappings and CUDA readiness.
        Native body columns and compatible single-width joint columns refer
        directly to arrays from ``state``, ``control`` or the connected model
        and may include unselected storage rows plus a row map. A joint
        selection borrows only when it has one width and compatible
        source-offset alignment; mixed widths or incompatible offsets require
        a device gather. Split body/root fields, materials and converted joint
        fields allocate output on the model device. See :doc:`output-reads`
        for the per-field contract. Use each column's ``data_row_index(i)`` or
        ``data_index_dlpack()`` to locate logical prim ``i``. Do not mutate
        borrowed state, control or model arrays until every reader has
        finished.

        CPU data is ready when this method returns. On CUDA, call this method
        on the Warp stream that last wrote the requested state, control or
        model arrays, or first make the current stream wait for that work.
        CUDA work does not block the CPU; a reader on another CUDA stream must
        wait for ``column.cuda_sync.wait_event``. Retain the result, group or
        column through consumer completion. The event reports producer
        readiness, not consumer completion.
        Borrowed results are live views, so state-buffer reuse, control writes
        such as ``Control.clear()`` or model edits require coordination with
        every reader; copy explicitly for history.
        Results need no explicit release, and their zero-valued ordinal metadata
        does not identify a simulation tick.
        """
        from . import _runtime  # noqa: PLC0415

        return _runtime.read_output(
            self,
            state,
            control,
            query=query,
            attributes=attributes,
        )

    def write(
        self,
        values: Mapping[str, wp.array[Any]],
        *,
        control: newton.Control,
        query: Query,
    ) -> None:
        """Replace selected native scalar joint targets without stage writes.

        Supported names are ``joint_target_q`` and ``joint_target_qd``.
        Values are one-dimensional float32 Warp arrays on the model device.
        The query must select only scalar articulation joints. Rows follow the
        corresponding public read's logical prim order, not its storage rows.
        Values use Newton's native units and signs, including radians, with
        no USD conversion.

        All input shapes, dtypes and finite values are checked before mutation.
        Payloads stay on the device; only a validation flag is read on the host.
        Inputs are copied and may be reused after return. Unselected values
        survive. Calls are synchronous and cannot run during CUDA graph capture.
        Use the producer's stream or establish a stream dependency first.
        The application owns solver configuration and step ordering. No state,
        forces, model properties or topology are changed.
        """
        from . import _runtime  # noqa: PLC0415

        with self._runtime_lock:
            _runtime.write_runtime(self, control, query=query, values=values)

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
        attributes: Sequence[str] | None = None,
    ) -> None:
        """Apply sealed state and drive targets from ovstage.

        With ``attributes=None`` (the default), read all supported body/joint
        state and drive targets. Body poses include ancestor transforms.
        Select any combination of ``body_q``, ``body_qd``, ``jointPosition``,
        ``jointVelocity``, ``jointPositionTarget``, and ``jointVelocityTarget``
        to update only those fields. Joint fields cover existing single-DOF
        channels; other coordinates and unselected arrays stay unchanged.
        Explicit selections do not run forward/inverse kinematics or reconcile
        body and joint state. Use the default full resync for body-derived joint
        coordinates. An empty selection is a no-op; unknown names are rejected.

        Joint angular values convert from stage degrees to Newton radians;
        linear values retain the stage's length units. If ``ordinal`` is
        omitted, read the latest sealed values. An explicit ordinal is a
        consistency ceiling, not a request for historical payloads.

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
                attributes=attributes,
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

    Populate USD inputs with :func:`ovnewton.population_desc` (or an equivalent
    descriptor) so unsupported authored API names survive schema population.
    This function cannot detect names discarded before import.

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

    For imports, populate USD inputs with :func:`ovnewton.population_desc` or
    an equivalent descriptor to preserve unsupported authored API names.

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
