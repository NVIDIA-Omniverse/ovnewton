# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared inputs, publication ranges, USD parsing, and validation for deformable bodies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from . import _stage
from ._errors import InvalidPhysicsError, OvstageContractError, UnsupportedPhysicsError
from ._parse import _api_paths, _row_scalar, _token_value, read_applied_schemas, read_hierarchy, read_stage_units
from ._schema_names import WORLD_MATRIX


class DeformableFamily(Enum):
    CURVES = "curves"
    SURFACE = "surface"
    VOLUME = "volume"


@dataclass(frozen=True)
class ParticleBuildRange:
    """An owned particle range and its fixed USD geometry frame."""

    geometry_path: str
    particle_start: int
    particle_end: int
    world_transform: tuple[float, ...]


@dataclass(frozen=True)
class ElementField:
    values: np.ndarray
    element_type: str


@dataclass(frozen=True)
class CurvesPayload:
    curve_vertex_counts: np.ndarray
    normals: np.ndarray
    curve_type: str
    wrap: str


@dataclass(frozen=True)
class SurfacePayload:
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray
    orientation: str


@dataclass(frozen=True)
class VolumePayload:
    tet_vertex_indices: np.ndarray
    orientation: str


DeformablePayload = Union[CurvesPayload, SurfacePayload, VolumePayload]


@dataclass(frozen=True)
class DeformableInput:
    family: DeformableFamily
    body_path: str
    geometry_path: str
    points: np.ndarray
    velocities: np.ndarray
    world_matrix: np.ndarray
    body_enabled: bool
    kinematic_enabled: bool
    mass: Optional[float]
    density: Optional[float]
    material_path: Optional[str]
    material: Mapping[str, Optional[float]]
    masses: ElementField
    thicknesses: Optional[ElementField]
    payload: DeformablePayload
    meters_per_unit: float = 1.0


_FAMILY_APIS = (
    (DeformableFamily.CURVES, "PhysicsCurvesDeformableSimAPI"),
    (DeformableFamily.SURFACE, "PhysicsSurfaceDeformableSimAPI"),
    (DeformableFamily.VOLUME, "PhysicsVolumeDeformableSimAPI"),
)
_DEFORMABLE_SCHEMAS = (
    "PhysicsDeformableBodyAPI",
    *[schema for _, schema in _FAMILY_APIS],
    "PhysicsCurvesDeformableMaterialAPI",
    "PhysicsSurfaceDeformableMaterialAPI",
    "PhysicsVolumeDeformableMaterialAPI",
)
_REST_ATTRIBUTES = ("physics:restShapePoints", "physics:restNormals", "physics:restTetVertexIndices")
_COMMON_ATTRIBUTES = (
    *_REST_ATTRIBUTES,
    "points",
    "velocities",
    "physics:collisionEnabled",
    WORLD_MATRIX,
    "material:binding:physics",
    "physics:masses",
    "physics:masses:elementType",
    "physics:thicknesses",
    "physics:thicknesses:elementType",
)
_FAMILY_ATTRIBUTES = {
    DeformableFamily.CURVES: ("curveVertexCounts", "normals", "type", "wrap"),
    DeformableFamily.SURFACE: ("faceVertexCounts", "faceVertexIndices", "orientation"),
    DeformableFamily.VOLUME: ("tetVertexIndices", "orientation"),
}
_RAGGED_ATTRIBUTES = frozenset({
    *_REST_ATTRIBUTES,
    "points",
    "velocities",
    "material:binding:physics",
    "physics:masses",
    "physics:thicknesses",
    "curveVertexCounts",
    "normals",
    "faceVertexCounts",
    "faceVertexIndices",
    "tetVertexIndices",
})
_BODY_ATTRIBUTES = (
    "physics:collisionEnabled",
    "physics:bodyEnabled",
    "physics:kinematicEnabled",
    "physics:mass",
    "physics:density",
)
_MATERIAL_ATTRIBUTES = (
    "physics:density",
    "physics:youngsModulus",
    "physics:poissonsRatio",
    "physics:surfaceStretchStiffness",
    "physics:surfaceShearStiffness",
    "physics:surfaceBendStiffness",
    "physics:curvesStretchStiffness",
    "physics:curvesShearStiffness",
    "physics:curvesBendStiffness",
    "physics:curvesTwistStiffness",
)
_POPULATION_ATTRIBUTES = tuple(
    dict.fromkeys(
        (
            *_COMMON_ATTRIBUTES,
            *_BODY_ATTRIBUTES,
            *_MATERIAL_ATTRIBUTES,
            *(name for attrs in _FAMILY_ATTRIBUTES.values() for name in attrs),
        )
    )
)


def population_desc() -> Any:
    """Select the canonical deformable properties ovnewton reads."""
    from ovstage import population  # noqa: PLC0415

    return population.Desc(
        selectors=(
            population.Selector(
                prim_predicate=population.PrimPredicate.has_schema(*_DEFORMABLE_SCHEMAS),
                property_predicate=population.PropertyPredicate.has_name(*_POPULATION_ATTRIBUTES),
            ),
            # OpenUSD preserves unknown applied API names in composed apiSchemas
            # metadata even when their schema plugin is unavailable. This fallback
            # lets ovnewton read explicitly authored draft-schema properties.
            population.Selector(
                prim_predicate=population.PrimPredicate.has_metadata("apiSchemas"),
                property_predicate=population.PropertyPredicate.has_name(*_POPULATION_ATTRIBUTES),
                prim_metadata_paths=("apiSchemas",),
            ),
        )
    )


def _deformable_schema_paths(
    stage: Any,
    pd: Any,
    ordinal: int,
    paths: Sequence[str],
) -> Mapping[str, set[str]]:
    """Index registered and raw applied-schema names by prim path."""
    result = {schema: set() for schema in _DEFORMABLE_SCHEMAS}
    for path, schemas in read_applied_schemas(stage, pd, ordinal, paths).items():
        for schema in schemas & result.keys():
            result[schema].add(path)
    return result


def _vectors(name: str, row: Optional[np.ndarray], width: int, *, required: bool, path: str) -> np.ndarray:
    if row is None or not len(row):
        if required:
            raise OvstageContractError(f"deformable geometry has no {name} at {path}")
        return np.empty((0, width), dtype=np.float32)
    values = np.asarray(row)
    if values.size % width:
        raise OvstageContractError(f"{name} has {values.size} scalar values; expected a multiple of {width} at {path}")
    if not np.all(np.isfinite(values)):
        raise InvalidPhysicsError(f"deformable {name} must be finite at {path}")
    return values.reshape(-1, width)


def _element_field(
    name: str,
    values: Sequence[float] | np.ndarray,
    element_type: str,
    counts: Mapping[str, int],
    *,
    path: str,
) -> ElementField:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(array):
        if element_type:
            raise InvalidPhysicsError(f"empty {name} requires an empty element type at {path}")
        return ElementField(array, "")
    if element_type not in counts:
        raise UnsupportedPhysicsError(f"{name} uses unsupported element type {element_type!r} at {path}")
    expected = counts[element_type]
    if len(array) != expected:
        raise InvalidPhysicsError(
            f"{name} has {len(array)} values; {element_type} count {expected} is required at {path}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise InvalidPhysicsError(f"{name} values must be finite and positive at {path}")
    return ElementField(array, element_type)


def _path_value(pd: Any, row: Optional[np.ndarray], path: str) -> Optional[str]:
    if row is None or not len(row):
        return None
    if len(row) != 1:
        raise UnsupportedPhysicsError(f"deformable physics binding requires one material target at {path}")
    value = _stage._path_to_string(pd, int(row[0]))
    return value or None


def _material_values(stage: Any, pd: Any, path: Optional[str], ordinal: int) -> Mapping[str, Optional[float]]:
    if path is None:
        return {}
    with _stage._path_list_query(stage, pd, [path]) as query:
        columns = _stage.read_columns(stage, pd, query, _MATERIAL_ATTRIBUTES, ordinal)
    return {name.removeprefix("physics:"): _row_scalar(columns[name], 0) for name in _MATERIAL_ATTRIBUTES}


def _body_material_path(stage: Any, pd: Any, path: str, ordinal: int) -> Optional[str]:
    with _stage._path_list_query(stage, pd, [path]) as query:
        rows = _stage.read_columns(
            stage,
            pd,
            query,
            ["material:binding:physics"],
            ordinal,
            ragged=["material:binding:physics"],
        )["material:binding:physics"]
    return _path_value(pd, rows.get(0), path)


def _body_values(stage: Any, pd: Any, path: str, ordinal: int) -> Mapping[str, Any]:
    with _stage._path_list_query(stage, pd, [path]) as query:
        columns = _stage.read_columns(stage, pd, query, _BODY_ATTRIBUTES, ordinal)
    if not _row_scalar(columns["physics:collisionEnabled"], 0, 1.0):
        raise UnsupportedPhysicsError(f"deformable collision disabling is unsupported at {path}")
    return {
        "body_enabled": bool(_row_scalar(columns["physics:bodyEnabled"], 0, 1.0)),
        "kinematic_enabled": bool(_row_scalar(columns["physics:kinematicEnabled"], 0, 0.0)),
        "mass": _row_scalar(columns["physics:mass"], 0),
        "density": _row_scalar(columns["physics:density"], 0),
    }


def _family_payload(
    family: DeformableFamily,
    pd: Any,
    columns: Mapping[str, Mapping[int, np.ndarray]],
    point_count: int,
    path: str,
) -> tuple[DeformablePayload, Mapping[str, int]]:
    if family is DeformableFamily.CURVES:
        curve_counts = np.asarray(columns["curveVertexCounts"].get(0, ()), dtype=np.int32).reshape(-1)
        if not len(curve_counts) or np.any(curve_counts < 2) or int(curve_counts.sum()) != point_count:
            raise OvstageContractError(f"curveVertexCounts does not describe the deformable points at {path}")
        curve_type = _token_value(pd, columns["type"].get(0)) or ""
        wrap = _token_value(pd, columns["wrap"].get(0)) or ""
        if curve_type != "linear" or wrap != "nonperiodic":
            raise UnsupportedPhysicsError(f"only linear nonperiodic deformable curves are supported at {path}")
        normals = _vectors("normals", columns["normals"].get(0), 3, required=False, path=path)
        if len(normals) not in (0, point_count):
            raise OvstageContractError(f"deformable curve normals must have one value per point at {path}")
        payload = CurvesPayload(curve_counts, normals, curve_type, wrap)
        return payload, {
            "constant": 1,
            "curve": len(curve_counts),
            "segment": int(np.sum(curve_counts - 1)),
            "point": point_count,
        }
    if family is DeformableFamily.SURFACE:
        face_counts = np.asarray(columns["faceVertexCounts"].get(0, ()), dtype=np.int32).reshape(-1)
        face_indices = np.asarray(columns["faceVertexIndices"].get(0, ()), dtype=np.int32).reshape(-1)
        if not len(face_counts) or np.any(face_counts < 3) or int(face_counts.sum()) != len(face_indices):
            raise OvstageContractError(f"surface face topology is incomplete at {path}")
        if np.any(face_indices < 0) or np.any(face_indices >= point_count):
            raise InvalidPhysicsError(f"surface face topology contains an invalid point index at {path}")
        payload = SurfacePayload(
            face_counts,
            face_indices,
            _token_value(pd, columns["orientation"].get(0)) or "rightHanded",
        )
        return payload, {"constant": 1, "face": len(face_counts), "point": point_count}
    tet_values = np.asarray(columns["tetVertexIndices"].get(0, ()), dtype=np.int32).reshape(-1)
    if len(tet_values) % 4:
        raise OvstageContractError(f"tetVertexIndices must contain four point indices per tetrahedron at {path}")
    tetrahedra = tet_values.reshape(-1, 4)
    if not len(tetrahedra) or np.any(tetrahedra < 0) or np.any(tetrahedra >= point_count):
        raise InvalidPhysicsError(f"volume topology is empty or contains an invalid point index at {path}")
    payload = VolumePayload(
        tetrahedra,
        _token_value(pd, columns["orientation"].get(0)) or "rightHanded",
    )
    return payload, {"constant": 1, "tetrahedron": len(tetrahedra), "point": point_count}


def read_deformables(stage: Any, pd: Any, ordinal: int) -> list[DeformableInput]:
    """Read normalized canonical deformable inputs without choosing a Newton solver.

    Stage reads return owned rows. Normalized arrays retain those snapshots
    rather than copying them again; none borrows released stage storage.
    """
    hierarchy = read_hierarchy(stage, pd, ordinal)
    meters_per_unit, _ = read_stage_units(stage, pd, ordinal)
    paths = sorted(hierarchy.parents)
    schema_paths = _deformable_schema_paths(stage, pd, ordinal, paths)
    body_paths = schema_paths["PhysicsDeformableBodyAPI"]
    rigid_paths = _api_paths(stage, pd, ordinal, "PhysicsRigidBodyAPI")
    values: list[DeformableInput] = []
    for family, schema in _FAMILY_APIS:
        for path in sorted(schema_paths[schema]):
            body_path = hierarchy.nearest(path, body_paths)
            if body_path is None:
                raise OvstageContractError(f"deformable simulation geometry has no body owner: {path}")
            rigid_path = hierarchy.nearest(path, rigid_paths)
            if rigid_path is not None:
                raise UnsupportedPhysicsError(
                    f"deformable under a rigid body is unsupported at {path} (rigid prim {rigid_path})"
                )
            attrs = (*_COMMON_ATTRIBUTES, *_FAMILY_ATTRIBUTES[family])
            ragged = tuple(name for name in attrs if name in _RAGGED_ATTRIBUTES)
            with _stage._path_list_query(stage, pd, [path]) as query:
                columns = _stage.read_columns(stage, pd, query, attrs, ordinal, ragged=ragged)
            if not _row_scalar(columns["physics:collisionEnabled"], 0, 1.0):
                raise UnsupportedPhysicsError(f"deformable collision disabling is unsupported at {path}")
            for name in _REST_ATTRIBUTES:
                if 0 in columns[name] and columns[name][0].size:
                    raise UnsupportedPhysicsError(
                        f"separate deformable rest geometry is unsupported: {name} at {path}"
                    )
            points = _vectors("points", columns["points"].get(0), 3, required=True, path=path)
            velocities = _vectors("velocities", columns["velocities"].get(0), 3, required=False, path=path)
            if len(velocities) not in (0, len(points)):
                raise OvstageContractError(f"velocities must be empty or have one value per point at {path}")
            payload, counts = _family_payload(family, pd, columns, len(points), path)
            masses = _element_field(
                "masses",
                columns["physics:masses"].get(0, ()),
                _token_value(pd, columns["physics:masses:elementType"].get(0)) or "",
                counts,
                path=path,
            )
            thicknesses = None
            if family is not DeformableFamily.VOLUME:
                thicknesses = _element_field(
                    "thicknesses",
                    columns["physics:thicknesses"].get(0, ()),
                    _token_value(pd, columns["physics:thicknesses:elementType"].get(0)) or "",
                    counts,
                    path=path,
                )
            material_path = _path_value(pd, columns["material:binding:physics"].get(0), path)
            if material_path is None and body_path != path:
                material_path = _body_material_path(stage, pd, body_path, ordinal)
            world_values = np.asarray(columns[WORLD_MATRIX].get(0, np.eye(4)), dtype=np.float64)
            if world_values.size != 16:
                raise OvstageContractError(f"{WORLD_MATRIX} must contain 16 values at {path}")
            world_matrix = world_values.reshape(4, 4)
            body = _body_values(stage, pd, body_path, ordinal)
            values.append(
                DeformableInput(
                    family=family,
                    body_path=body_path,
                    geometry_path=path,
                    points=points,
                    velocities=velocities,
                    world_matrix=world_matrix,
                    material_path=material_path,
                    material=_material_values(stage, pd, material_path, ordinal),
                    masses=masses,
                    thicknesses=thicknesses,
                    payload=payload,
                    meters_per_unit=meters_per_unit,
                    **body,
                )
            )
    return values


def reject_unsupported_material(value: DeformableInput, supported: Collection[str]) -> None:
    unsupported = sorted(
        f"physics:{name}" for name, authored in value.material.items() if authored is not None and name not in supported
    )
    if unsupported:
        raise UnsupportedPhysicsError(
            f"unsupported {value.family.value} material properties {', '.join(unsupported)} at {value.geometry_path}"
        )


def constant_thickness(value: DeformableInput) -> float:
    thicknesses = value.thicknesses
    if thicknesses is None or thicknesses.element_type != "constant" or len(thicknesses.values) != 1:
        raise UnsupportedPhysicsError(f"{value.family.value} requires one constant thickness at {value.geometry_path}")
    return float(thicknesses.values[0])


def world_particle_data(value: DeformableInput) -> tuple[np.ndarray, np.ndarray]:
    """Validate transformed vectors in the storage type used by Newton."""
    linear = value.world_matrix[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.asarray(value.points @ linear + value.world_matrix[3, :3], dtype=np.float32)
        velocities = np.asarray(value.velocities @ linear, dtype=np.float32)
    for name, data in (("points", points), ("velocities", velocities)):
        if not np.all(np.isfinite(data)):
            raise InvalidPhysicsError(f"deformable world {name} exceed finite float32 storage at {value.geometry_path}")
    return points, velocities


def _validate_rigid_frame(value: DeformableInput) -> None:
    path = value.geometry_path
    matrix = value.world_matrix
    linear = matrix[:3, :3]
    if (
        not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[:, 3], [0, 0, 0, 1])
        or not np.allclose(linear @ linear.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(linear), 1, atol=1e-6)
    ):
        raise UnsupportedPhysicsError(f"deformable lowering requires a rigid world transform without scale at {path}")


def resolve_density(value: DeformableInput, default_density: float) -> float:
    """Authored densities use stage units; the builder's SI fallback needs conversion."""
    body, material = value.density, value.material.get("density")
    density = body if body is not None else material
    if density is None:
        density = default_density * value.meters_per_unit**3
    for number in (body, material, density):
        if number is not None and (not np.isfinite(number) or number <= 0):
            raise InvalidPhysicsError(f"deformable density must be finite and positive at {value.geometry_path}")
    return float(density)


def validate_particle_masses(masses: np.ndarray, path: str, *, quantity: str = "mass") -> None:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        native = np.asarray(masses, dtype=np.float32)
        inverse = np.divide(np.float32(1), native, out=np.zeros_like(native), where=native != 0)
    if not np.all(np.isfinite(native)) or not np.all(np.isfinite(inverse)) or np.any((masses > 0) & (native == 0)):
        raise InvalidPhysicsError(
            f"deformable {quantity} or inverse {quantity} exceeds finite float32 storage at {path}"
        )


def validate_common(value: DeformableInput) -> None:
    """Reject body, mass, and frame semantics unsupported by every family."""
    path = value.geometry_path
    if not value.body_enabled:
        raise UnsupportedPhysicsError(f"deformable lowering does not support disabled bodies at {path}")
    if value.kinematic_enabled:
        raise UnsupportedPhysicsError(f"deformable lowering does not support kinematic bodies at {path}")
    if value.mass is not None or len(value.masses.values):
        raise UnsupportedPhysicsError(f"deformable lowering does not support authored masses at {path}")
    _validate_rigid_frame(value)


def world_frame(value: DeformableInput) -> tuple[float, ...]:
    """Convert the USD row-vector frame to Newton's translation and quaternion."""
    rotation = wp.quat_from_matrix(wp.mat33(*value.world_matrix[:3, :3].T.reshape(-1).tolist()))
    return tuple(value.world_matrix[3, :3]) + tuple(rotation)


def model_particle_ranges(model: Any, prefix: str) -> tuple[ParticleBuildRange, ...]:
    """Read finalized surface or volume ownership once when creating a binding."""
    attributes = getattr(model, "ovnewton", None)
    paths = getattr(attributes, f"{prefix}_path", ())
    if not any(paths):
        return ()
    # Newton adjusts native ranges when builders are combined. Copy these small
    # static columns once; per-frame publication stays on the device.
    starts = getattr(model, f"{prefix}_particle_start").numpy().tolist()
    ends = getattr(model, f"{prefix}_particle_end").numpy().tolist()
    frames = getattr(attributes, f"{prefix}_world_transform").numpy().tolist()
    return tuple(ParticleBuildRange(path, starts[i], ends[i], tuple(frames[i])) for i, path in enumerate(paths) if path)
