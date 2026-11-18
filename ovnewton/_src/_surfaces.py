# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical cloth prepared and appended through Newton's public builder."""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from ._deformables import (
    DeformableInput,
    SurfacePayload,
    constant_thickness,
    reject_unsupported_material,
    resolve_density,
    validate_common,
    world_frame,
    world_particle_data,
)
from ._errors import OvstageContractError


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
    if payload.orientation == "leftHanded":
        result = result[:, ::-1]
    return result


@dataclass(frozen=True)
class PreparedSurface:
    points: np.ndarray
    velocities: np.ndarray
    triangles: np.ndarray
    thickness: float
    areal_density: float
    tri_ke: float | None
    edge_ke: float | None


def prepare_surface(value: DeformableInput, default_density: float) -> PreparedSurface:
    """Resolve surface geometry and material before mutating a builder."""
    validate_common(value)
    thickness = constant_thickness(value)
    reject_unsupported_material(value, {"density", "surfaceStretchStiffness", "surfaceBendStiffness"})
    triangles = _triangulate_surface(value)
    points, velocities = world_particle_data(value)
    edge_a = points[triangles[:, 1]] - points[triangles[:, 0]]
    edge_b = points[triangles[:, 2]] - points[triangles[:, 0]]
    if np.any(0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1) < 1.0e-12):
        raise OvstageContractError(f"surface deformable has a zero-area triangle: {value.geometry_path}")

    volumetric_density = resolve_density(value, default_density)
    areal_density = volumetric_density * thickness
    stretch = value.material.get("surfaceStretchStiffness")
    bend = value.material.get("surfaceBendStiffness")
    tri_ke = float(stretch) if stretch is not None else None
    edge_ke = float(bend) if bend is not None else None
    return PreparedSurface(points, velocities, triangles, thickness, areal_density, tri_ke, edge_ke)


def build_surface(builder, value, prepared):
    """Append prevalidated cloth and retain its USD ownership through finalize."""
    particle_start = builder.particle_count
    surface_index = len(builder.surface_label)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=prepared.points.tolist(),
        indices=prepared.triangles.reshape(-1).tolist(),
        density=prepared.areal_density,
        tri_ke=prepared.tri_ke,
        tri_ka=0.0,
        edge_ke=prepared.edge_ke,
        particle_radius=0.5 * prepared.thickness,
        validate_mesh=True,
        label=value.geometry_path,
    )
    if len(prepared.velocities):
        builder.particle_qd[particle_start : builder.particle_count] = prepared.velocities.tolist()
    # Retain USD ownership on Newton's surface domain. Other caller-added
    # surfaces keep an empty path, even if their labels match USD prims.
    for name, dtype, data in (
        ("surface_path", str, value.geometry_path),
        ("surface_world_transform", wp.transformf, world_frame(value)),
    ):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=name, dtype=dtype, frequency=newton.Model.AttributeFrequency.SURFACE, namespace="ovnewton"
            )
        )
        builder.custom_attributes[f"ovnewton:{name}"].values[surface_index] = data
