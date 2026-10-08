# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral inputs for canonical deformable bodies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from . import _stage
from ._errors import InvalidPhysicsError, OvstageContractError, UnsupportedPhysicsError
from ._parse import _row_scalar, _token_value, read_hierarchy
from ._schema_names import USD_SCHEMAS, WORLD_MATRIX


class DeformableFamily(Enum):
    CURVES = "curves"
    SURFACE = "surface"
    VOLUME = "volume"


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


@dataclass(frozen=True)
class SurfaceBuildRange:
    geometry_path: str
    particle_start: int
    particle_end: int
    triangle_start: int
    triangle_end: int
    edge_start: int
    edge_end: int


@dataclass(frozen=True)
class DeformableBuildInfo:
    surfaces: tuple[SurfaceBuildRange, ...] = ()


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
_COMMON_ATTRIBUTES = (
    "points",
    "velocities",
    WORLD_MATRIX,
    "material:binding:physics",
    "physics:masses",
    "physics:masses:elementType",
    "physics:thicknesses",
    "physics:thicknesses:elementType",
)
_BODY_ATTRIBUTES = (
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
            "curveVertexCounts",
            "normals",
            "type",
            "wrap",
            "faceVertexCounts",
            "faceVertexIndices",
            "orientation",
            "tetVertexIndices",
        )
    )
)
_RAW_API_SCHEMAS = "usd-metadata:apiSchemas"


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
    if not paths:
        return result
    with _stage._path_list_query(stage, pd, paths) as query:
        columns = _stage.read_columns(
            stage,
            pd,
            query,
            (USD_SCHEMAS, _RAW_API_SCHEMAS),
            ordinal,
            ragged=(USD_SCHEMAS, _RAW_API_SCHEMAS),
        )
    for rows in columns.values():
        for index, row in rows.items():
            path = paths[index]
            for token in row:
                schema = pd.token_to_string(int(token))
                if schema in result:
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
    return values.reshape(-1, width).copy()


def _element_field(
    name: str,
    values: Sequence[float] | np.ndarray,
    element_type: str,
    counts: Mapping[str, int],
    *,
    path: str,
) -> ElementField:
    array = np.asarray(values, dtype=np.float32).reshape(-1).copy()
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


def _path_value(pd: Any, row: Optional[np.ndarray]) -> Optional[str]:
    if row is None or not len(row):
        return None
    value = pd.path_to_string(int(row[0]))
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
    return _path_value(pd, rows.get(0))


def _body_values(stage: Any, pd: Any, path: str, ordinal: int) -> Mapping[str, Any]:
    with _stage._path_list_query(stage, pd, [path]) as query:
        columns = _stage.read_columns(stage, pd, query, _BODY_ATTRIBUTES, ordinal)
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
        normals = _vectors("normals", columns["normals"].get(0), 3, required=True, path=path)
        if len(normals) != point_count:
            raise OvstageContractError(f"deformable curve normals must have one value per point at {path}")
        payload = CurvesPayload(curve_counts.copy(), normals, curve_type, wrap)
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
            face_counts.copy(),
            face_indices.copy(),
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
        tetrahedra.copy(),
        _token_value(pd, columns["orientation"].get(0)) or "rightHanded",
    )
    return payload, {"constant": 1, "tetrahedron": len(tetrahedra), "point": point_count}


def read_deformables(stage: Any, pd: Any, ordinal: int) -> list[DeformableInput]:
    """Read normalized canonical deformable inputs without choosing a Newton solver."""
    hierarchy = read_hierarchy(stage, pd, ordinal)
    paths = sorted(hierarchy.parents)
    schema_paths = _deformable_schema_paths(stage, pd, ordinal, paths)
    body_paths = schema_paths["PhysicsDeformableBodyAPI"]
    values: list[DeformableInput] = []
    for family, schema in _FAMILY_APIS:
        for path in sorted(schema_paths[schema]):
            body_path = hierarchy.nearest(path, body_paths)
            if body_path is None:
                raise OvstageContractError(f"deformable simulation geometry has no body owner: {path}")
            family_attrs = {
                DeformableFamily.CURVES: ("curveVertexCounts", "normals", "type", "wrap"),
                DeformableFamily.SURFACE: ("faceVertexCounts", "faceVertexIndices", "orientation"),
                DeformableFamily.VOLUME: ("tetVertexIndices", "orientation"),
            }[family]
            attrs = (*_COMMON_ATTRIBUTES, *family_attrs)
            ragged = tuple(
                name
                for name in attrs
                if name
                in {
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
                }
            )
            with _stage._path_list_query(stage, pd, [path]) as query:
                columns = _stage.read_columns(stage, pd, query, attrs, ordinal, ragged=ragged)
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
            material_path = _path_value(pd, columns["material:binding:physics"].get(0))
            if material_path is None and body_path != path:
                material_path = _body_material_path(stage, pd, body_path, ordinal)
            world_values = np.asarray(columns[WORLD_MATRIX].get(0, np.eye(4)), dtype=np.float64)
            if world_values.size != 16:
                raise OvstageContractError(f"{WORLD_MATRIX} must contain 16 values at {path}")
            world_matrix = world_values.reshape(4, 4).copy()
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
                    **body,
                )
            )
    return values


def _triangulate_surface(value: DeformableInput) -> np.ndarray:
    payload = value.payload
    if not isinstance(payload, SurfacePayload):
        raise TypeError("surface deformable has an incompatible payload")
    indices = payload.face_vertex_indices
    triangles = []
    offset = 0
    for count in payload.face_vertex_counts:
        face = indices[offset : offset + int(count)]
        triangles.extend((int(face[0]), int(face[i]), int(face[i + 1])) for i in range(1, len(face) - 1))
        offset += int(count)
    result = np.asarray(triangles, dtype=np.int32)
    reflects = np.linalg.det(value.world_matrix[:3, :3]) < 0.0
    if (payload.orientation == "leftHanded") != reflects:
        result = result[:, ::-1]
    return result


def _validate_surface_lowering_input(value: DeformableInput) -> None:
    """Reject authored semantics that the surface implementation cannot preserve."""
    path = value.geometry_path
    if value.family is not DeformableFamily.SURFACE:
        raise UnsupportedPhysicsError(f"surface lowering does not support {value.family.value} deformables at {path}")
    if not value.body_enabled:
        raise UnsupportedPhysicsError(f"surface lowering does not support disabled bodies at {path}")
    if value.kinematic_enabled:
        raise UnsupportedPhysicsError(f"surface lowering does not support kinematic bodies at {path}")
    if value.mass is not None or len(value.masses.values):
        raise UnsupportedPhysicsError(f"surface lowering does not support authored masses at {path}")
    if not np.array_equal(value.world_matrix, np.eye(4, dtype=value.world_matrix.dtype)):
        raise UnsupportedPhysicsError(f"surface lowering requires an identity world transform at {path}")
    if value.thicknesses is None or value.thicknesses.element_type != "constant" or len(value.thicknesses.values) != 1:
        raise UnsupportedPhysicsError(f"surface lowering requires one constant thickness at {path}")

    supported_material = {"density", "surfaceStretchStiffness", "surfaceBendStiffness"}
    unsupported_material = sorted(
        name for name, authored in value.material.items() if authored is not None and name not in supported_material
    )
    if unsupported_material:
        names = ", ".join(f"physics:{name}" for name in unsupported_material)
        raise UnsupportedPhysicsError(f"surface lowering does not support {names} at {path}")


def build_deformables(builder: Any, values: Sequence[DeformableInput]) -> DeformableBuildInfo:
    """Lower supported canonical surface records into a Newton builder."""
    for value in values:
        _validate_surface_lowering_input(value)

    surfaces = []
    for value in values:
        triangles = _triangulate_surface(value)
        points = np.asarray(value.points, dtype=np.float32).copy()
        edge_a = points[triangles[:, 1]] - points[triangles[:, 0]]
        edge_b = points[triangles[:, 2]] - points[triangles[:, 0]]
        if np.any(0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1) < 1.0e-12):
            raise OvstageContractError(f"surface deformable has a zero-area triangle: {value.geometry_path}")

        assert value.thicknesses is not None
        thickness = float(value.thicknesses.values[0])

        material_density = value.material.get("density")
        volumetric_density = value.density if value.density is not None else material_density
        volumetric_density = (
            float(volumetric_density) if volumetric_density is not None else float(builder.default_shape_cfg.density)
        )
        areal_density = volumetric_density * thickness
        stretch = value.material.get("surfaceStretchStiffness")
        bend = value.material.get("surfaceBendStiffness")
        tri_ke = float(stretch) * thickness if stretch is not None else None
        edge_ke = float(bend) * thickness**3 if bend is not None else None

        particle_start = builder.particle_count
        triangle_start = builder.tri_count
        edge_start = builder.edge_count
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=points.tolist(),
            indices=triangles.reshape(-1).tolist(),
            density=areal_density,
            tri_ke=tri_ke,
            tri_ka=0.0,
            tri_kd=0.1,
            edge_ke=edge_ke,
            particle_radius=0.5 * thickness,
            validate_mesh=True,
            label=value.geometry_path,
        )
        if len(value.velocities):
            builder.particle_qd[particle_start : builder.particle_count] = value.velocities.tolist()
        surfaces.append(
            SurfaceBuildRange(
                geometry_path=value.geometry_path,
                particle_start=particle_start,
                particle_end=builder.particle_count,
                triangle_start=triangle_start,
                triangle_end=builder.tri_count,
                edge_start=edge_start,
                edge_end=builder.edge_count,
            )
        )
    if surfaces:
        import newton  # noqa: PLC0415

        # Newton carries these attributes through ordinary builder.finalize().
        # Keep the mapping with the model without wrapping finalization or
        # requiring callers to copy ovnewton's private metadata themselves.
        builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="surface", namespace="ovnewton"))
        for name, dtype, references in (
            ("surface_path", str, None),
            ("surface_particle_range", wp.vec2i, "particle"),
            ("surface_triangle_range", wp.vec2i, "triangle"),
            ("surface_edge_range", wp.vec2i, "edge"),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    dtype=dtype,
                    frequency="ovnewton:surface",
                    namespace="ovnewton",
                    references=references,
                )
            )
        for surface in surfaces:
            builder.add_custom_values(
                **{
                    "ovnewton:surface_path": surface.geometry_path,
                    "ovnewton:surface_particle_range": (surface.particle_start, surface.particle_end),
                    "ovnewton:surface_triangle_range": (surface.triangle_start, surface.triangle_end),
                    "ovnewton:surface_edge_range": (surface.edge_start, surface.edge_end),
                }
            )
    return DeformableBuildInfo(surfaces=tuple(surfaces))


def model_surface_ranges(model: Any) -> tuple[SurfaceBuildRange, ...]:
    """Read finalized surface mappings once, when a binding is initialized."""
    attributes = getattr(model, "ovnewton", None)
    paths = getattr(attributes, "surface_path", ())
    if not paths:
        return ()
    # Only the small, static index ranges are copied at binding setup.
    # Publishing particle positions does not read these arrays back each frame.
    particles = attributes.surface_particle_range.numpy().tolist()
    triangles = attributes.surface_triangle_range.numpy().tolist()
    edges = attributes.surface_edge_range.numpy().tolist()
    return tuple(SurfaceBuildRange(path, *particles[i], *triangles[i], *edges[i]) for i, path in enumerate(paths))
