# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse composed physics and topology from ovstage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ovstage import Filter, FilterOp, Predicate, instancing

from . import _stage
from ._errors import InvalidPhysicsError, OvstageContractError, UnsupportedPhysicsError, log_diagnostic
from ._schema_names import (
    COLLISION_ENABLED,
    JOINT_LOCAL_POS_0,
    NEWTON_CONTACT_ADHESION,
    NEWTON_CONTACT_DAMPING,
    NEWTON_CONTACT_FRICTION_GAIN,
    NEWTON_CONTACT_GAP,
    NEWTON_CONTACT_MARGIN,
    NEWTON_CONTACT_STIFFNESS,
    NEWTON_GRAVITY_ENABLED,
    NEWTON_HYDROELASTIC_ENABLED,
    NEWTON_HYDROELASTIC_STIFFNESS,
    NEWTON_JOINT_ARMATURE,
    NEWTON_JOINT_DAMPING,
    NEWTON_JOINT_FRICTION,
    NEWTON_JOINT_LIMIT_DAMPING,
    NEWTON_JOINT_LIMIT_STIFFNESS,
    NEWTON_JOINT_VELOCITY_LIMIT,
    NEWTON_MASS_MODEL,
    NEWTON_MAX_HULL_VERTICES,
    NEWTON_ROLLING_FRICTION,
    NEWTON_SDF_MAX_RESOLUTION,
    NEWTON_SDF_NARROW_BAND_INNER,
    NEWTON_SDF_NARROW_BAND_OUTER,
    NEWTON_SDF_PADDING,
    NEWTON_SDF_TARGET_VOXEL_SIZE,
    NEWTON_SDF_TEXTURE_FORMAT,
    NEWTON_SHELL_THICKNESS,
    NEWTON_TORSIONAL_FRICTION,
    USD_PARENT,
    USD_PATH,
    USD_PRIM_TYPE,
    USD_SCHEMAS,
    WORLD_MATRIX,
    joint_drive,
    joint_limit,
    joint_state,
)


@dataclass(frozen=True)
class _DofDesc:
    axis: Tuple[float, float, float]
    rotational: bool
    limit_lower: float
    limit_upper: float
    drive_enabled: bool
    drive_target: Optional[float]
    drive_velocity: Optional[float]
    drive_stiffness: Optional[float]
    drive_damping: Optional[float]
    drive_force_limit: Optional[float]
    damping: Optional[float]
    armature: Optional[float]
    friction: Optional[float]
    velocity_limit: Optional[float]
    broadcast_limit_stiffness: Optional[float]
    broadcast_limit_damping: Optional[float]
    initial_position: Optional[float]
    initial_velocity: Optional[float]


@dataclass(frozen=True)
class _JointDesc:
    prim_type: str
    body0: Optional[str]
    body1: Optional[str]
    local_pos0: Tuple[float, float, float]
    local_rot0: Tuple[float, float, float, float]
    local_pos1: Tuple[float, float, float]
    local_rot1: Tuple[float, float, float, float]
    dofs: Tuple[_DofDesc, ...]
    collision_enabled: bool = False
    excluded_from_articulation: bool = False
    min_distance: float = -1.0
    max_distance: float = -1.0


@dataclass(frozen=True)
class _Hierarchy:
    parents: Dict[str, Optional[str]]
    incomplete_parents: frozenset[str] = frozenset()

    def _parent(self, path: str) -> Optional[str]:
        if path in self.incomplete_parents:
            raise OvstageContractError(f"{USD_PARENT!r} is missing for nested prim {path}")
        return self.parents[path]

    def ancestors(self, path: str, *, include_self: bool = False) -> Tuple[str, ...]:
        if path not in self.parents:
            raise OvstageContractError(f"prim is absent from the populated hierarchy: {path}")
        lineage: List[str] = []
        seen: set[str] = set()
        node = path if include_self else self._parent(path)
        while node is not None:
            if node in seen:
                raise OvstageContractError(f"populated hierarchy contains a cycle at {node}")
            seen.add(node)
            lineage.append(node)
            if not node:
                break
            if node not in self.parents:
                raise OvstageContractError(f"populated hierarchy references an unknown parent: {node}")
            node = self._parent(node)
        return tuple(lineage)

    def contains(self, root: str, path: str) -> bool:
        if root == "/":
            return path in self.parents
        return root in self.ancestors(path, include_self=True)

    def nearest(self, path: str, candidates: Collection[str]) -> Optional[str]:
        return next((node for node in self.ancestors(path, include_self=True) if node in candidates), None)


# USD physics joint types ovnewton consumes.
JOINT_TYPES = [
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "PhysicsFixedJoint",
    "PhysicsSphericalJoint",
    "PhysicsDistanceJoint",
    "PhysicsJoint",
]
# Gprim types mapped to Newton shapes.
ANALYTIC_SHAPE_TYPES = ["Cube", "Sphere", "Capsule", "Cylinder", "Cone"]
SHAPE_TYPES = [*ANALYTIC_SHAPE_TYPES, "Plane", "Mesh"]
MESH_APPROXIMATIONS = {
    "convexDecomposition",
    "convexHull",
    "boundingSphere",
    "boundingCube",
    "meshSimplification",
}

_D6_AXES = (
    ("transX", (1.0, 0.0, 0.0), False),
    ("transY", (0.0, 1.0, 0.0), False),
    ("transZ", (0.0, 0.0, 1.0), False),
    ("rotX", (1.0, 0.0, 0.0), True),
    ("rotY", (0.0, 1.0, 0.0), True),
    ("rotZ", (0.0, 0.0, 1.0), True),
)
_DRIVE_FIELDS = ("drive_target", "drive_velocity", "drive_stiffness", "drive_damping", "drive_force_limit")
_NEWTON_JOINT_API_FIELDS = {
    "armature": NEWTON_JOINT_ARMATURE,
    "damping": NEWTON_JOINT_DAMPING,
    "friction": NEWTON_JOINT_FRICTION,
    "velocity_limit": NEWTON_JOINT_VELOCITY_LIMIT,
    "broadcast_limit_stiffness": NEWTON_JOINT_LIMIT_STIFFNESS,
    "broadcast_limit_damping": NEWTON_JOINT_LIMIT_DAMPING,
}
_AXIS_VECTORS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
_STAGE_METADATA_PATH = "/"
_STAGE_METERS_PER_UNIT = "usd-metadata:metersPerUnit"
_STAGE_UP_AXIS = "usd-metadata:upAxis"
_USD_GENERATED_PROTOTYPE_PREFIX = "__Prototype_"


def read_stage_units(stage: Any, pd: Any, ordinal: int) -> Tuple[float, str]:
    """Return the populated stage ``metersPerUnit`` and ``upAxis`` metadata."""
    with _stage._path_list_query(stage, pd, [_STAGE_METADATA_PATH]) as query:
        columns = _stage.read_columns(stage, pd, query, [_STAGE_METERS_PER_UNIT, _STAGE_UP_AXIS], ordinal)
    meters_rows = columns[_STAGE_METERS_PER_UNIT]
    up_rows = columns[_STAGE_UP_AXIS]
    meters = meters_rows.get(0)
    up = _token_value(pd, up_rows.get(0))
    if meters is None or not len(meters) or not np.isfinite(meters[0]) or float(meters[0]) <= 0.0:
        raise OvstageContractError("population stage metadata has no valid metersPerUnit")
    if up not in ("Y", "Z"):
        raise OvstageContractError(f"population stage metadata has invalid upAxis {up!r}")
    return float(meters[0]), up


def scene_gravity_vector(
    stage: Any,
    pd: Any,
    ordinal: int,
    *,
    meters_per_unit: float,
    up_axis: str,
) -> Optional[Tuple[float, float, float]]:
    """Resolve the first ``PhysicsScene`` gravity into stage coordinates."""
    paths = _type_paths(stage, pd, "PhysicsScene", ordinal)
    if not paths:
        return None
    path = paths[0]
    with _stage._path_list_query(stage, pd, [path]) as query:
        columns = _stage.read_columns(
            stage,
            pd,
            query,
            ["physics:gravityMagnitude", "physics:gravityDirection", NEWTON_GRAVITY_ENABLED],
            ordinal,
        )
    magnitude_rows = columns["physics:gravityMagnitude"]
    direction_rows = columns["physics:gravityDirection"]
    enabled_rows = columns[NEWTON_GRAVITY_ENABLED]

    enabled_row = enabled_rows.get(0)
    if (
        path in _api_paths(stage, pd, ordinal, "NewtonSceneAPI")
        and enabled_row is not None
        and len(enabled_row)
        and not bool(enabled_row[0])
    ):
        return (0.0, 0.0, 0.0)

    magnitude_row = magnitude_rows.get(0)
    direction_row = direction_rows.get(0)
    if magnitude_row is None or not len(magnitude_row):
        raise OvstageContractError(f"PhysicsScene has no gravity magnitude at {path}")
    if direction_row is None or len(direction_row) < 3:
        raise OvstageContractError(f"PhysicsScene has no gravity direction at {path}")

    magnitude = float(magnitude_row[0])
    if not np.isfinite(magnitude) and not np.isneginf(magnitude):
        raise InvalidPhysicsError(f"invalid gravity magnitude at {path}")
    if magnitude < 0.0:
        magnitude = 9.81 / meters_per_unit

    direction = np.asarray(direction_row[:3], dtype=np.float64)
    if not np.all(np.isfinite(direction)):
        raise InvalidPhysicsError(f"invalid gravity direction at {path}")
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        direction = -np.asarray(_AXIS_VECTORS[up_axis], dtype=np.float64)
    else:
        direction /= length
    return tuple(float(component) for component in direction * magnitude)


def read_hierarchy(stage: Any, pd: Any, ordinal: int) -> _Hierarchy:
    """Read one immutable prim-parent snapshot."""
    with stage.query() as query:
        expected_count = _stage._count(stage, query)
        columns = _stage._read_path_columns(stage, pd, query, [USD_PATH, USD_PARENT], ordinal)

    path_values = columns[USD_PATH]
    if len(path_values) != expected_count:
        raise OvstageContractError(
            f"stage has {expected_count} prims but {USD_PATH!r} materialized {len(path_values)} rows"
        )
    parent_values = columns[USD_PARENT]
    missing_nested = sorted(
        path
        for path in path_values.keys() - parent_values.keys()
        if path.count("/") > 1 and not _is_generated_prototype_path(path)
    )
    parents = {path: parent_values.get(path) or None for path in path_values}
    known = set(parents)
    unknown = sorted({parent for parent in parents.values() if parent and parent not in known})
    if unknown:
        raise OvstageContractError(f"{USD_PARENT!r} references an unknown prim: {unknown[0]}")
    return _Hierarchy(parents, frozenset(missing_nested))


def _token_value(pd: Any, row: Optional[np.ndarray]) -> Optional[str]:
    """Resolve one token-valued column entry from its interned token id."""
    if row is None or not len(row):
        return None
    token = int(row[0])
    return None if token == 0 else pd.token_to_string(token)


def _paths_by_prim_type(
    stage: Any,
    pd: Any,
    paths: Sequence[str],
    ordinal: int,
) -> Dict[str, List[str]]:
    """Group materialized paths by their readable ``usd-prim-type`` value."""
    if not paths:
        return {}
    with _stage._path_list_query(stage, pd, paths) as query:
        rows = _stage.read_columns(stage, pd, query, [USD_PRIM_TYPE], ordinal)[USD_PRIM_TYPE]
    grouped: Dict[str, List[str]] = {}
    for index, path in enumerate(paths):
        row = rows.get(index)
        if row is None or not len(row):
            raise OvstageContractError(f"prim has no readable {USD_PRIM_TYPE!r} value: {path}")
        prim_type = _token_value(pd, row) or ""
        grouped.setdefault(prim_type, []).append(path)
    return grouped


def _token_axis(pd: Any, row: Optional[np.ndarray], *, path: Optional[str] = None) -> Optional[str]:
    """Resolve an ``axis`` token to ``"X"``/``"Y"``/``"Z"`` when valid."""
    tok = _token_value(pd, row)
    if tok is not None and tok not in ("X", "Y", "Z"):
        location = f" at {path}" if path else ""
        raise InvalidPhysicsError(f"invalid physics axis token {tok!r}{location}")
    return tok


def _row_scalar(rows: Dict[int, np.ndarray], index: int, default: Any = None) -> Any:
    value = rows.get(index)
    return float(value[0]) if value is not None and len(value) else default


def _row_int(rows: Dict[int, np.ndarray], index: int, default: Any = None) -> Any:
    value = rows.get(index)
    return int(value[0]) if value is not None and len(value) else default


def _row_tuple(rows: Dict[int, np.ndarray], index: int, width: int, default: Any = None) -> Any:
    value = rows.get(index)
    return tuple(float(x) for x in value[:width]) if value is not None and len(value) >= width else default


def _dof_desc(
    rows: Dict[str, Dict[int, np.ndarray]],
    common_rows: Dict[str, Dict[int, np.ndarray]],
    index: int,
    axis: Tuple[float, float, float],
    rotational: bool,
    lower: float,
    upper: float,
    *,
    force_limit_default: Optional[float] = None,
) -> _DofDesc:
    drive_enabled = any(index in rows[name] for name in _DRIVE_FIELDS)
    force_limit = _row_scalar(rows["drive_force_limit"], index, force_limit_default)
    if force_limit is not None and force_limit >= 1.0e30:
        force_limit = float("inf")
    velocity_limit = _row_scalar(common_rows["velocity_limit"], index)
    if velocity_limit is not None and velocity_limit >= 1.0e30:
        velocity_limit = None
    return _DofDesc(
        axis=axis,
        rotational=rotational,
        limit_lower=lower,
        limit_upper=upper,
        drive_enabled=drive_enabled,
        drive_target=_row_scalar(rows["drive_target"], index, 0.0) if drive_enabled else None,
        drive_velocity=_row_scalar(rows["drive_velocity"], index, 0.0) if drive_enabled else None,
        drive_stiffness=_row_scalar(rows["drive_stiffness"], index, 0.0) if drive_enabled else None,
        drive_damping=_row_scalar(rows["drive_damping"], index, 0.0) if drive_enabled else None,
        drive_force_limit=force_limit if drive_enabled else None,
        damping=_row_scalar(common_rows["damping"], index),
        armature=_row_scalar(common_rows["armature"], index),
        friction=_row_scalar(common_rows["friction"], index),
        velocity_limit=velocity_limit,
        broadcast_limit_stiffness=_row_scalar(common_rows["broadcast_limit_stiffness"], index),
        broadcast_limit_damping=_row_scalar(common_rows["broadcast_limit_damping"], index),
        initial_position=_row_scalar(rows["joint_state_position"], index),
        initial_velocity=_row_scalar(rows["joint_state_velocity"], index),
    )


def _filtered_paths(
    stage: Any,
    pd: Any,
    predicates: Sequence[Predicate],
    ordinal: int,
) -> List[str]:
    """Materialize matching paths and release the discovery query."""
    with stage.query(filter=Filter(list(predicates))) as query:
        query.wait()
        result = stage.fetch_query_result(query)
        if not result.total_prim_count:
            return []
        return _stage._query_paths(
            stage,
            pd,
            query,
            ordinal,
            expected_count=result.total_prim_count,
        )


def _type_paths(stage: Any, pd: Any, prim_type: str, ordinal: int) -> List[str]:
    """Materialize paths for one prim type."""
    return _filtered_paths(
        stage,
        pd,
        [Predicate(USD_PRIM_TYPE, FilterOp.IN, [prim_type])],
        ordinal,
    )


def _native_instance_paths(stage: Any, hierarchy: _Hierarchy, paths: Sequence[str]) -> List[str]:
    found: set[str] = set()
    for prototype_root in instancing.get_prototype_roots(stage):
        prototype_root = prototype_root.rstrip("/")
        instance_roots = [root.rstrip("/") for root in instancing.get_instance_roots(stage, prototype_root)]
        for path in paths:
            if not hierarchy.contains(prototype_root, path):
                continue
            suffix = path[len(prototype_root) :]
            found.update(root + suffix for root in instance_roots)
            if not instance_roots:
                found.add(path)
    return sorted(found)


def _is_generated_prototype_path(path: str) -> bool:
    root = path.split("/", 2)[1] if path.startswith("/") else ""
    suffix = root.removeprefix(_USD_GENERATED_PROTOTYPE_PREFIX)
    return suffix != root and suffix.isdigit()


def validate_unsupported_point_instancers(
    stage: Any,
    pd: Any,
    ordinal: int,
    *,
    hierarchy: _Hierarchy,
) -> None:
    """Reject PointInstancers whose prototypes contain physics."""
    point_instancers = _type_paths(stage, pd, "PointInstancer", ordinal)
    if not point_instancers:
        return
    physics_paths: set[str] = set()
    for schema in ("PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI", "NewtonSiteAPI"):
        physics_paths.update(_api_paths(stage, pd, ordinal, schema))
    with _stage._path_list_query(stage, pd, point_instancers) as query:
        rows = _stage.read_columns(stage, pd, query, ["prototypes"], ordinal, ragged=["prototypes"])["prototypes"]
    prototypes = {i: [int(value) for value in row.tolist()] for i, row in rows.items()}
    for index, path in enumerate(point_instancers):
        for target in prototypes.get(index, ()):
            root = pd.path_to_string(target).rstrip("/")
            if root and any(hierarchy.contains(root, candidate) for candidate in physics_paths):
                raise UnsupportedPhysicsError(f"physics PointInstancer is unsupported at {path}")


def validate_supported_joints(stage: Any, pd: Any, ordinal: int) -> None:
    """Reject joint schemas whose semantics this importer cannot preserve."""
    for schema in ("MjcJointAPI", "MjcEqualityConnectAPI", "MjcEqualityWeldAPI", "MjcEqualityJointAPI"):
        paths = _filtered_paths(
            stage,
            pd,
            [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, [schema])],
            ordinal,
        )
        if paths:
            raise UnsupportedPhysicsError(f"unsupported {schema} properties at {paths[0]}")


def read_bodies(
    stage: Any,
    pd: Any,
    ordinal: int,
    *,
    hierarchy: _Hierarchy,
) -> Tuple[List[int], List[str]]:
    """Return rigid-body IDs and strings in stable interned-ID order.

    Identity comes from readable ``usd-path``, so standalone bodies are
    included without deriving them from joints.
    """
    paths = _filtered_paths(
        stage,
        pd,
        [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, ["PhysicsRigidBodyAPI"])],
        ordinal,
    )
    instanced = _native_instance_paths(stage, hierarchy, paths)
    if instanced:
        raise UnsupportedPhysicsError(
            f"native USD rigid-body instancing is unsupported at {instanced[0]}; "
            "flatten physics instances before attachment"
        )
    bodies = sorted({pd.intern_path(path): path for path in paths}.items())
    return [body_id for body_id, _ in bodies], [path for _, path in bodies]


def _api_paths(stage: Any, pd: Any, ordinal: int, schema: str) -> set[str]:
    return set(
        _filtered_paths(
            stage,
            pd,
            [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, [schema])],
            ordinal,
        )
    )


def _mass_api_paths(stage: Any, pd: Any, ordinal: int) -> set[str]:
    """Set of prim paths that apply ``PhysicsMassAPI``."""
    return _api_paths(stage, pd, ordinal, "PhysicsMassAPI")


def _newton_mass_api_paths(stage: Any, pd: Any, ordinal: int) -> set[str]:
    """Set of prim paths that apply ``NewtonMassAPI``."""
    return _api_paths(stage, pd, ordinal, "NewtonMassAPI")


def read_articulation_self_collisions(stage: Any, pd: Any, ordinal: int) -> Dict[str, bool]:
    """Return authored articulation roots and their self-collision policy."""
    roots: set[str] = set()
    for schema in ("PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"):
        roots.update(
            _filtered_paths(
                stage,
                pd,
                [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, [schema])],
                ordinal,
            )
        )
    if not roots:
        return {}
    paths = sorted(roots)
    with _stage._path_list_query(stage, pd, paths) as query:
        rows = _stage.read_columns(stage, pd, query, ["newton:selfCollisionEnabled"], ordinal)[
            "newton:selfCollisionEnabled"
        ]
    return {path: bool(rows[i][0]) if i in rows and len(rows[i]) else True for i, path in enumerate(paths)}


def read_mimics(stage: Any, pd: Any, ordinal: int) -> List[Dict[str, Any]]:
    """Return enabled ``NewtonMimicAPI`` constraints."""
    paths = sorted(
        _filtered_paths(
            stage,
            pd,
            [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, ["NewtonMimicAPI"])],
            ordinal,
        )
    )
    if not paths:
        return []
    with _stage._path_list_query(stage, pd, paths) as query:
        columns = _stage.read_columns(
            stage,
            pd,
            query,
            (USD_PRIM_TYPE, "newton:mimicEnabled", "newton:mimicJoint", "newton:mimicCoef0", "newton:mimicCoef1"),
            ordinal,
            ragged=("newton:mimicJoint",),
        )

    mimics = []
    for i, path in enumerate(paths):
        enabled_row = columns["newton:mimicEnabled"].get(i)
        enabled = bool(enabled_row[0]) if enabled_row is not None and len(enabled_row) else True
        if not enabled:
            continue
        prim_type = _token_value(pd, columns[USD_PRIM_TYPE].get(i))
        if prim_type is None:
            raise OvstageContractError(f"prim has no readable {USD_PRIM_TYPE!r} value: {path}")
        targets = columns["newton:mimicJoint"].get(i)
        if targets is None or not len(targets):
            raise InvalidPhysicsError(f"NewtonMimicAPI at {path} has no newton:mimicJoint target")
        if len(targets) != 1:
            raise InvalidPhysicsError(f"NewtonMimicAPI at {path} has multiple newton:mimicJoint targets")
        coef0_row = columns["newton:mimicCoef0"].get(i)
        coef1_row = columns["newton:mimicCoef1"].get(i)
        coef0 = float(coef0_row[0]) if coef0_row is not None and len(coef0_row) else 0.0
        coef1 = float(coef1_row[0]) if coef1_row is not None and len(coef1_row) else 1.0
        if not np.isfinite(coef0) or not np.isfinite(coef1):
            raise InvalidPhysicsError(f"NewtonMimicAPI at {path} has non-finite coefficients")
        mimics.append(
            {
                "path": path,
                "leader": pd.path_to_string(int(targets[0])),
                "coef0": coef0,
                "coef1": coef1,
                "rotational": prim_type == "PhysicsRevoluteJoint",
            }
        )
    return mimics


def read_collision_groups(stage: Any, pd: Any, ordinal: int) -> List[Dict[str, Any]]:
    """Return collision-group membership and filter rules from the population surface."""
    paths = _type_paths(stage, pd, "PhysicsCollisionGroup", ordinal)
    if not paths:
        return []
    with _stage._path_list_query(stage, pd, paths) as query:
        columns = _stage.read_columns(
            stage,
            pd,
            query,
            [
                "collection:colliders:includes",
                "collection:colliders:excludes",
                "physics:filteredGroups",
                "physics:invertFilteredGroups",
                "physics:mergeGroup",
            ],
            ordinal,
            ragged=["collection:colliders:includes", "collection:colliders:excludes", "physics:filteredGroups"],
        )

    def targets(attribute: str, index: int) -> List[str]:
        return [pd.path_to_string(int(target)) for target in columns[attribute].get(index, [])]

    groups = []
    for i, path in enumerate(paths):
        inverted = columns["physics:invertFilteredGroups"].get(i)
        groups.append({
            "path": path,
            "includes": targets("collection:colliders:includes", i),
            "excludes": targets("collection:colliders:excludes", i),
            "filtered_groups": targets("physics:filteredGroups", i),
            "inverted": bool(inverted[0]) if inverted is not None and len(inverted) else False,
            "merge_group": _token_value(pd, columns["physics:mergeGroup"].get(i)),
        })
    return groups


def read_filtered_pairs(stage: Any, pd: Any, ordinal: int) -> List[Tuple[str, str]]:
    """Return authored ``PhysicsFilteredPairsAPI`` path pairs."""
    paths = _filtered_paths(
        stage,
        pd,
        [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, ["PhysicsFilteredPairsAPI"])],
        ordinal,
    )
    if not paths:
        return []
    with _stage._path_list_query(stage, pd, paths) as query:
        rows = _stage.read_columns(
            stage,
            pd,
            query,
            ["physics:filteredPairs"],
            ordinal,
            ragged=["physics:filteredPairs"],
        )["physics:filteredPairs"]
    pairs = {i: [int(x) for x in row.tolist()] for i, row in rows.items()}
    return [(path, pd.path_to_string(target)) for i, path in enumerate(paths) for target in pairs.get(i, [])]


# Authored geometry attributes per gprim type (all readable authored attrs).
_GEOM_ATTRS = ("extent", "size", "radius", "height")


def _read_colliders_of_type(
    stage: Any,
    pd: Any,
    shape: str,
    paths: Sequence[str],
    ordinal: int,
    *,
    newton_mass_api: set[str],
    newton_collision_api: set[str],
    newton_mesh_api: set[str],
    newton_sdf_api: set[str],
) -> List[Dict[str, Any]]:
    if not paths:
        return []

    with _stage._path_list_query(stage, pd, paths) as q:
        n = len(paths)
        mesh_attrs = ("points", "faceVertexCounts", "faceVertexIndices") if shape == "Mesh" else ()
        ragged_attrs = ("material:binding:physics", *mesh_attrs)
        columns = _stage.read_columns(
            stage,
            pd,
            q,
            (
                WORLD_MATRIX,
                *_GEOM_ATTRS,
                *mesh_attrs,
                "material:binding:physics",
                "axis",
                "orientation",
                "physics:approximation",
                COLLISION_ENABLED,
                NEWTON_MASS_MODEL,
                NEWTON_SHELL_THICKNESS,
                NEWTON_CONTACT_GAP,
                NEWTON_CONTACT_MARGIN,
                NEWTON_MAX_HULL_VERTICES,
                NEWTON_SDF_MAX_RESOLUTION,
                NEWTON_SDF_TARGET_VOXEL_SIZE,
                NEWTON_SDF_NARROW_BAND_INNER,
                NEWTON_SDF_NARROW_BAND_OUTER,
                NEWTON_SDF_TEXTURE_FORMAT,
                NEWTON_SDF_PADDING,
                NEWTON_HYDROELASTIC_ENABLED,
                NEWTON_HYDROELASTIC_STIFFNESS,
            ),
            ordinal,
            ragged=ragged_attrs,
        )
        world_matrices = columns[WORLD_MATRIX]
        cols = {a: columns[a] for a in _GEOM_ATTRS}
        mesh_cols = {a: columns[a] for a in mesh_attrs}
        mats = {i: [int(x) for x in row.tolist()] for i, row in columns["material:binding:physics"].items()}
        axis_rows = columns["axis"]
        orientation_rows = columns["orientation"]
        approximation_rows = columns["physics:approximation"]
        enabled_rows = columns[COLLISION_ENABLED]
        mass_model_rows = columns[NEWTON_MASS_MODEL]
        shell_thickness_rows = columns[NEWTON_SHELL_THICKNESS]
        contact_gap_rows = columns[NEWTON_CONTACT_GAP]
        contact_margin_rows = columns[NEWTON_CONTACT_MARGIN]
        max_hull_vertices_rows = columns[NEWTON_MAX_HULL_VERTICES]
        sdf_max_resolution_rows = columns[NEWTON_SDF_MAX_RESOLUTION]
        sdf_target_voxel_size_rows = columns[NEWTON_SDF_TARGET_VOXEL_SIZE]
        sdf_narrow_band_inner_rows = columns[NEWTON_SDF_NARROW_BAND_INNER]
        sdf_narrow_band_outer_rows = columns[NEWTON_SDF_NARROW_BAND_OUTER]
        sdf_texture_format_rows = columns[NEWTON_SDF_TEXTURE_FORMAT]
        sdf_padding_rows = columns[NEWTON_SDF_PADDING]
        hydroelastic_enabled_rows = columns[NEWTON_HYDROELASTIC_ENABLED]
        hydroelastic_stiffness_rows = columns[NEWTON_HYDROELASTIC_STIFFNESS]
        colliders = []
        for i in range(n):
            path = paths[i]
            mat = mats.get(i)
            axis = _token_axis(pd, axis_rows.get(i), path=path)
            orientation = _token_value(pd, orientation_rows.get(i))
            if shape == "Mesh" and orientation not in (None, "rightHanded", "leftHanded"):
                raise InvalidPhysicsError(f"invalid mesh orientation {orientation!r} at {path}")
            approximation = _token_value(pd, approximation_rows.get(i))
            if shape == "Mesh" and approximation not in (None, "none", *MESH_APPROXIMATIONS):
                log_diagnostic(
                    "mesh-approximation-ignored",
                    f"unsupported physics:approximation={approximation!r}; the raw mesh was retained",
                    path=path,
                )
                approximation = None
            enabled = enabled_rows.get(i)
            collision_enabled = bool(enabled[0]) if enabled is not None and len(enabled) else True
            max_hull_vertices = _row_int(max_hull_vertices_rows, i) if path in newton_mesh_api else None
            if max_hull_vertices == -1:
                max_hull_vertices = None
            colliders.append(
                {
                    "path": path,
                    "type": shape,
                    "world_matrix": world_matrices.get(i),
                    "extent": cols["extent"].get(i),
                    "size": cols["size"].get(i),
                    "radius": cols["radius"].get(i),
                    "height": cols["height"].get(i),
                    "material": pd.path_to_string(mat[0]) if mat else None,
                    "axis": axis,
                    "orientation": orientation,
                    "points": mesh_cols.get("points", {}).get(i),
                    "face_counts": mesh_cols.get("faceVertexCounts", {}).get(i),
                    "face_indices": mesh_cols.get("faceVertexIndices", {}).get(i),
                    "approximation": approximation,
                    "collision_enabled": collision_enabled,
                    "mass_model": _token_value(pd, mass_model_rows.get(i)) if path in newton_mass_api else None,
                    "shell_thickness": _row_scalar(shell_thickness_rows, i) if path in newton_mass_api else None,
                    "contact_gap": _row_scalar(contact_gap_rows, i) if path in newton_collision_api else None,
                    "contact_margin": _row_scalar(contact_margin_rows, i) if path in newton_collision_api else None,
                    "sdf_api": path in newton_sdf_api,
                    "newton_mesh_api": path in newton_mesh_api,
                    "max_hull_vertices": max_hull_vertices,
                    "sdf_max_resolution": (
                        _row_int(sdf_max_resolution_rows, i) if path in newton_sdf_api else None
                    ),
                    "sdf_target_voxel_size": (
                        _row_scalar(sdf_target_voxel_size_rows, i) if path in newton_sdf_api else None
                    ),
                    "sdf_narrow_band_inner": (
                        _row_scalar(sdf_narrow_band_inner_rows, i) if path in newton_sdf_api else None
                    ),
                    "sdf_narrow_band_outer": (
                        _row_scalar(sdf_narrow_band_outer_rows, i) if path in newton_sdf_api else None
                    ),
                    "sdf_texture_format": (
                        _token_value(pd, sdf_texture_format_rows.get(i)) if path in newton_sdf_api else None
                    ),
                    "sdf_padding": _row_scalar(sdf_padding_rows, i) if path in newton_sdf_api else None,
                    "hydroelastic_enabled": (
                        bool(_row_int(hydroelastic_enabled_rows, i, False)) if path in newton_sdf_api else None
                    ),
                    "hydroelastic_stiffness": (
                        _row_scalar(hydroelastic_stiffness_rows, i) if path in newton_sdf_api else None
                    ),
                }
            )
        return colliders


def read_colliders(
    stage: Any,
    pd: Any,
    ordinal: int,
    *,
    newton_mass_api: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Read all ``PhysicsCollisionAPI`` colliders.

    Each result includes its own materialized ``path`` plus type, transform,
    geometry, material, mass inputs, and axis values. The hierarchy snapshot lets
    the builder assign each collider to its nearest rigid-body ancestor and retain
    colliders with no rigid ancestor as static world geometry.

    One schema query discovers collider identity through ``usd-path``, then one
    ``usd-prim-type`` column classifies every path."""
    all_paths = set(
        _filtered_paths(
            stage,
            pd,
            [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, ["PhysicsCollisionAPI"])],
            ordinal,
        )
    )

    paths_by_type = _paths_by_prim_type(stage, pd, sorted(all_paths), ordinal)
    native_instances = sorted(
        path
        for prim_type, paths in paths_by_type.items()
        if prim_type.endswith("Instance")
        for path in paths
    )
    if native_instances:
        raise UnsupportedPhysicsError(
            "native USD physics instancing cannot preserve composed collider properties at "
            f"{native_instances[0]}; flatten physics instances before attachment"
        )
    found: List[Dict[str, Any]] = []
    handled_paths: set[str] = set()
    if newton_mass_api is None:
        newton_mass_api = _newton_mass_api_paths(stage, pd, ordinal)
    newton_sdf_api = _api_paths(stage, pd, ordinal, "NewtonSDFCollisionAPI")
    newton_mesh_api = _api_paths(stage, pd, ordinal, "NewtonMeshCollisionAPI")
    newton_collision_api = _api_paths(stage, pd, ordinal, "NewtonCollisionAPI")
    newton_collision_api.update(newton_sdf_api)
    for shape in SHAPE_TYPES:
        shape_colliders = _read_colliders_of_type(
            stage,
            pd,
            shape,
            paths_by_type.get(shape, ()),
            ordinal,
            newton_mass_api=newton_mass_api,
            newton_collision_api=newton_collision_api,
            newton_mesh_api=newton_mesh_api,
            newton_sdf_api=newton_sdf_api,
        )
        found.extend(shape_colliders)
        handled_paths.update(c["path"] for c in shape_colliders)
    # add_usd warns and skips CollisionAPI on Xform containers (common on a
    # rigid-body parent whose descendant Mesh carries the actual collider).
    container_paths = set(paths_by_type.get("Xform", ()))
    for path in sorted(container_paths):
        log_diagnostic(
            "collision-container-ignored",
            "PhysicsCollisionAPI on an Xform container has no collision geometry and was ignored",
            path=path,
            level=logging.INFO,
        )
    unsupported_paths = sorted(all_paths - handled_paths - container_paths)
    if unsupported_paths:
        raise UnsupportedPhysicsError(
            "PhysicsCollisionAPI is applied to an unsupported prim type at "
            f"{unsupported_paths[0]}; supported collider types: {', '.join(SHAPE_TYPES)}"
        )
    return found


def read_sites(stage: Any, pd: Any, ordinal: int) -> List[Dict[str, Any]]:
    """Read supported body-attached ``NewtonSiteAPI`` analytic shapes."""
    all_paths = _api_paths(stage, pd, ordinal, "NewtonSiteAPI")
    if not all_paths:
        return []
    collider_paths = _api_paths(stage, pd, ordinal, "PhysicsCollisionAPI")
    conflicts = sorted(all_paths & collider_paths)
    if conflicts:
        raise InvalidPhysicsError(f"prim has both NewtonSiteAPI and PhysicsCollisionAPI at {conflicts[0]}")

    paths = sorted(all_paths)
    with _stage._path_list_query(stage, pd, paths) as query:
        columns = _stage.read_columns(
            stage,
            pd,
            query,
            (USD_PRIM_TYPE, WORLD_MATRIX, *_GEOM_ATTRS, "axis"),
            ordinal,
        )
    prim_types = [_token_value(pd, columns[USD_PRIM_TYPE].get(i)) for i in range(len(paths))]
    unsupported_paths = [
        path
        for path, prim_type in zip(paths, prim_types, strict=True)
        if prim_type not in ANALYTIC_SHAPE_TYPES
    ]
    if unsupported_paths:
        raise UnsupportedPhysicsError(
            "unsupported NewtonSiteAPI prim type at "
            f"{unsupported_paths[0]}; supported site types: {', '.join(ANALYTIC_SHAPE_TYPES)}"
        )
    sites = [
        {
            "path": path,
            "type": prim_types[i],
            "world_matrix": columns[WORLD_MATRIX].get(i),
            "extent": columns["extent"].get(i),
            "size": columns["size"].get(i),
            "radius": columns["radius"].get(i),
            "height": columns["height"].get(i),
            "axis": _token_axis(pd, columns["axis"].get(i), path=path),
        }
        for i, path in enumerate(paths)
    ]
    return sorted(sites, key=lambda site: ANALYTIC_SHAPE_TYPES.index(site["type"]))


def _newton_material_value(
    rows: Dict[int, np.ndarray],
    index: int,
    *,
    attribute: str,
    path: str,
    negative_infinity_is_default: bool = False,
    minimum: Optional[float] = None,
) -> Optional[float]:
    value = _row_scalar(rows, index)
    if value is None or (negative_infinity_is_default and np.isneginf(value)):
        return None
    if np.isfinite(value) and (minimum is None or value >= minimum):
        return value
    requirement = "finite" if minimum is None else f"finite and at least {minimum:g}"
    log_diagnostic(
        "newton-material-value-ignored",
        f"{attribute} must be {requirement}; using the builder default",
        path=path,
    )
    return None


def read_physics_materials(stage: Any, pd: Any, ordinal: int) -> Dict[str, Dict[str, Optional[float]]]:
    """Read applied ``PhysicsMaterialAPI`` values keyed by material prim path."""
    newton_paths = _api_paths(stage, pd, ordinal, "NewtonMaterialAPI")
    physics_paths = _filtered_paths(
        stage,
        pd,
        [Predicate(USD_SCHEMAS, FilterOp.CONTAINS, ["PhysicsMaterialAPI"])],
        ordinal,
    )
    paths = sorted(set(physics_paths) | newton_paths)
    if not paths:
        return {}

    with _stage._path_list_query(stage, pd, paths) as q:
        n = len(paths)
        columns = _stage.read_columns(
            stage,
            pd,
            q,
            (
                "physics:dynamicFriction",
                "physics:restitution",
                "physics:density",
                NEWTON_TORSIONAL_FRICTION,
                NEWTON_ROLLING_FRICTION,
                NEWTON_CONTACT_STIFFNESS,
                NEWTON_CONTACT_DAMPING,
                NEWTON_CONTACT_FRICTION_GAIN,
                NEWTON_CONTACT_ADHESION,
            ),
            ordinal,
        )
        rows = {
            "mu": columns["physics:dynamicFriction"],
            "restitution": columns["physics:restitution"],
            "density": columns["physics:density"],
        }
        newton_rows = {
            "mu_torsional": (NEWTON_TORSIONAL_FRICTION, False, 0.0),
            "mu_rolling": (NEWTON_ROLLING_FRICTION, False, 0.0),
            "ke": (NEWTON_CONTACT_STIFFNESS, True, None),
            "kd": (NEWTON_CONTACT_DAMPING, True, None),
            "kf": (NEWTON_CONTACT_FRICTION_GAIN, True, None),
            "ka": (NEWTON_CONTACT_ADHESION, True, None),
        }

        materials: Dict[str, Dict[str, Optional[float]]] = {}
        for i in range(n):
            path = paths[i]
            material = {name: _row_scalar(values, i) for name, values in rows.items()}
            if path in newton_paths:
                material.update(
                    {
                        name: _newton_material_value(
                            columns[attribute],
                            i,
                            attribute=attribute,
                            path=path,
                            negative_infinity_is_default=sentinel,
                            minimum=minimum,
                        )
                        for name, (attribute, sentinel, minimum) in newton_rows.items()
                    }
                )
            materials[path] = material
        return materials


# Authored joint frame attributes common to every supported joint type.
_JOINT_FRAME_ATTRS = (
    JOINT_LOCAL_POS_0,
    "physics:localRot0",
    "physics:localPos1",
    "physics:localRot1",
)


def _read_joints_of_type(
    stage: Any,
    pd: Any,
    jtype: str,
    paths: Sequence[str],
    ordinal: int,
    newton_joint_api: Collection[str],
) -> List[Tuple[str, _JointDesc]]:
    if not paths:
        return []

    with _stage._path_list_query(stage, pd, paths) as q:
        n = len(paths)
        single_dof = jtype in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
        instance = "angular" if jtype == "PhysicsRevoluteJoint" else "linear"
        fixed_attrs = [
            *_JOINT_FRAME_ATTRS,
            COLLISION_ENABLED,
            "physics:jointEnabled",
            "physics:excludeFromArticulation",
            "physics:breakForce",
            "physics:breakTorque",
        ]
        if jtype == "PhysicsDistanceJoint":
            fixed_attrs.extend(("physics:minDistance", "physics:maxDistance"))
        if single_dof:
            fixed_attrs.extend(
                (
                    "physics:lowerLimit",
                    "physics:upperLimit",
                    "physics:axis",
                    joint_state(instance, "position"),
                    joint_state(instance, "velocity"),
                    *(
                        joint_drive(instance, name)
                        for name in (
                            "targetPosition",
                            "targetVelocity",
                            "stiffness",
                            "damping",
                            "maxForce",
                        )
                    ),
                )
            )

        dof_attrs: Dict[str, str] = {}
        if single_dof or jtype == "PhysicsJoint":
            dof_attrs = dict(_NEWTON_JOINT_API_FIELDS)
            fixed_attrs.extend(dof_attrs.values())

        d6_attrs: Dict[str, Dict[str, str]] = {}
        if jtype == "PhysicsJoint":
            for name, _axis, rotational in _D6_AXES:
                d6_attrs[name] = {
                    "low": joint_limit(name, "low"),
                    "high": joint_limit(name, "high"),
                    "drive_target": joint_drive(name, "targetPosition"),
                    "drive_velocity": joint_drive(name, "targetVelocity"),
                    "drive_stiffness": joint_drive(name, "stiffness"),
                    "drive_damping": joint_drive(name, "damping"),
                    "drive_force_limit": joint_drive(name, "maxForce"),
                    "joint_state_position": joint_state(name, "position"),
                    "joint_state_velocity": joint_state(name, "velocity"),
                }
                fixed_attrs.extend(d6_attrs[name].values())

        relationship_attrs = ("physics:body0", "physics:body1")
        columns = _stage.read_columns(
            stage,
            pd,
            q,
            tuple(dict.fromkeys((*fixed_attrs, *relationship_attrs))),
            ordinal,
            ragged=relationship_attrs,
        )
        b0 = {i: [int(x) for x in row.tolist()] for i, row in columns["physics:body0"].items()}
        b1 = {i: [int(x) for x in row.tolist()] for i, row in columns["physics:body1"].items()}
        cols = {attr: columns[attr] for attr in _JOINT_FRAME_ATTRS}
        axis_rows = columns["physics:axis"] if single_dof else {}
        collision_rows = columns[COLLISION_ENABLED]
        enabled_rows = columns["physics:jointEnabled"]
        excluded_rows = columns["physics:excludeFromArticulation"]
        break_force_rows = columns["physics:breakForce"]
        break_torque_rows = columns["physics:breakTorque"]
        dof_rows = {name: columns[attr] for name, attr in dof_attrs.items()}
        d6_rows = {axis: {name: columns[attr] for name, attr in attrs.items()} for axis, attrs in d6_attrs.items()}
        single_rows = (
            {
                "low": columns["physics:lowerLimit"],
                "high": columns["physics:upperLimit"],
                "drive_target": columns[joint_drive(instance, "targetPosition")],
                "drive_velocity": columns[joint_drive(instance, "targetVelocity")],
                "drive_stiffness": columns[joint_drive(instance, "stiffness")],
                "drive_damping": columns[joint_drive(instance, "damping")],
                "drive_force_limit": columns[joint_drive(instance, "maxForce")],
                "joint_state_position": columns[joint_state(instance, "position")],
                "joint_state_velocity": columns[joint_state(instance, "velocity")],
            }
            if single_dof
            else {}
        )

        def path_for(rel: Dict[int, List[int]], i: int):
            targets = rel.get(i)
            return pd.path_to_string(targets[0]) if targets else None

        joints = []
        for i in range(n):
            path = paths[i]
            if dof_attrs and path in newton_joint_api:
                missing = [attribute for attribute in _NEWTON_JOINT_API_FIELDS.values() if i not in columns[attribute]]
                if missing:
                    raise UnsupportedPhysicsError(
                        f"NewtonJointAPI is not completely populated at {path}; missing "
                        + ", ".join(missing)
                        + "; use an ovstage build with Newton schemas 0.4.0"
                    )
            for rel_name, rel in (("body0", b0), ("body1", b1)):
                if len(rel.get(i, [])) > 1:
                    raise InvalidPhysicsError(f"joint {path} has multiple physics:{rel_name} targets")
            collision_enabled = collision_rows.get(i)
            collisions = (
                bool(collision_enabled[0]) if collision_enabled is not None and len(collision_enabled) else False
            )
            enabled = enabled_rows.get(i)
            if enabled is not None and len(enabled) and not bool(enabled[0]):
                log_diagnostic(
                    "disabled-joint-skipped",
                    "physics:jointEnabled=false; the joint was omitted, matching add_usd's default policy",
                    path=path,
                    level=logging.INFO,
                )
                continue
            excluded = excluded_rows.get(i)
            excluded_from_articulation = bool(excluded[0]) if excluded is not None and len(excluded) else False
            for name, rows in (("breakForce", break_force_rows), ("breakTorque", break_torque_rows)):
                value = rows.get(i)
                if value is not None and len(value) and float(value[0]) < 1.0e30:
                    log_diagnostic(
                        "joint-break-threshold-ignored",
                        f"finite physics:{name} has no Newton model representation and was ignored",
                        path=path,
                    )
            dofs = []
            for name, axis, rotational in _D6_AXES if jtype == "PhysicsJoint" else ():
                rows = d6_rows[name]
                low = _row_scalar(rows["low"], i)
                high = _row_scalar(rows["high"], i)
                if low is None or high is None or low >= high:
                    continue
                dofs.append(_dof_desc(rows, dof_rows, i, axis, rotational, low, high))
            if single_dof:
                rotational = jtype == "PhysicsRevoluteJoint"
                axis = _AXIS_VECTORS[_token_axis(pd, axis_rows.get(i), path=path) or "X"]
                dofs.append(
                    _dof_desc(
                        single_rows,
                        dof_rows,
                        i,
                        axis,
                        rotational,
                        _row_scalar(single_rows["low"], i, float("-inf")),
                        _row_scalar(single_rows["high"], i, float("inf")),
                        force_limit_default=float("inf"),
                    )
                )
            joints.append(
                (
                    path,
                    _JointDesc(
                        prim_type=jtype,
                        body0=path_for(b0, i),
                        body1=path_for(b1, i),
                        local_pos0=_row_tuple(cols[JOINT_LOCAL_POS_0], i, 3, (0.0, 0.0, 0.0)),
                        local_rot0=_row_tuple(cols["physics:localRot0"], i, 4, (0.0, 0.0, 0.0, 1.0)),
                        local_pos1=_row_tuple(cols["physics:localPos1"], i, 3, (0.0, 0.0, 0.0)),
                        local_rot1=_row_tuple(cols["physics:localRot1"], i, 4, (0.0, 0.0, 0.0, 1.0)),
                        dofs=tuple(dofs),
                        collision_enabled=collisions,
                        excluded_from_articulation=excluded_from_articulation,
                        min_distance=_row_scalar(columns.get("physics:minDistance", {}), i, -1.0),
                        max_distance=_row_scalar(columns.get("physics:maxDistance", {}), i, -1.0),
                    ),
                )
            )
        return joints


def read_joints(
    stage: Any,
    pd: Any,
    ordinal: int,
    *,
    hierarchy: _Hierarchy,
) -> List[Tuple[str, _JointDesc]]:
    """Read supported joints and reject every recognized unsupported joint schema."""
    validate_supported_joints(stage, pd, ordinal)
    paths = _filtered_paths(
        stage,
        pd,
        [Predicate(USD_PRIM_TYPE, FilterOp.IN, JOINT_TYPES)],
        ordinal,
    )
    # Physics joint prototypes are not included in ovstage's public instancing graph.
    instanced = {path for path in paths if _is_generated_prototype_path(path)}
    instanced.update(_native_instance_paths(stage, hierarchy, [path for path in paths if path not in instanced]))
    if instanced:
        raise UnsupportedPhysicsError(
            f"native USD joint instancing is unsupported at {min(instanced)}; "
            "flatten physics instances before attachment"
        )
    paths_by_type = _paths_by_prim_type(stage, pd, paths, ordinal)
    newton_joint_api = _api_paths(stage, pd, ordinal, "NewtonJointAPI")
    return [
        joint
        for jtype in JOINT_TYPES
        for joint in _read_joints_of_type(
            stage,
            pd,
            jtype,
            paths_by_type.get(jtype, ()),
            ordinal,
            newton_joint_api,
        )
    ]
