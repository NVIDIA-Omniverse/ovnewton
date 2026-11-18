# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded canonical tetrahedral inputs lowered through Newton's public builder."""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from ._deformables import (
    VolumePayload,
    reject_unsupported_material,
    resolve_density,
    validate_common,
    validate_particle_masses,
    world_frame,
    world_particle_data,
)
from ._errors import InvalidPhysicsError, UnsupportedPhysicsError


@dataclass(frozen=True)
class PreparedVolume:
    points: np.ndarray
    velocities: np.ndarray
    density: float
    mu: float | None
    lam: float | None


def prepare_volume(value, default_density):
    """Resolve and validate volume data before mutating a builder."""
    path = value.geometry_path
    validate_common(value)
    reject_unsupported_material(value, {"density", "youngsModulus", "poissonsRatio"})
    young = value.material.get("youngsModulus")
    poisson = value.material.get("poissonsRatio")
    if (young is None) != (poisson is None):
        raise UnsupportedPhysicsError(f"volume requires both Young's modulus and Poisson's ratio at {path}")
    if young is not None and (
        not np.isfinite(young) or young <= 0 or not np.isfinite(poisson) or not 0 <= poisson < 0.5
    ):
        raise InvalidPhysicsError(f"invalid volume elastic material at {path}")
    mu, lam = _elastic_parameters(value)
    payload = value.payload
    if not isinstance(payload, VolumePayload) or payload.orientation != "rightHanded":
        raise UnsupportedPhysicsError(f"volume requires right-handed tetrahedra at {path}")
    tets = payload.tet_vertex_indices
    points, velocities = world_particle_data(value)
    points64 = points.astype(np.float64)
    edges = points64[tets[:, 1:]] - points64[tets[:, :1]]
    determinants = np.linalg.det(edges)
    if not np.all(np.isfinite(determinants)) or np.any(determinants <= 1e-12):
        raise InvalidPhysicsError(f"volume has inverted or degenerate tetrahedra at {path}")
    density = resolve_density(value, default_density)
    masses = np.zeros(len(points), dtype=np.float64)
    np.add.at(masses, tets.reshape(-1), np.repeat(density * determinants / 24, 4))
    validate_particle_masses(masses, path)
    if len(np.unique(tets)) != len(points):
        raise UnsupportedPhysicsError(f"volume contains unused points at {path}")
    if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
        raise InvalidPhysicsError(f"volume contains duplicate tetrahedra at {path}")
    faces = np.concatenate([tets[:, face] for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise UnsupportedPhysicsError(f"volume has non-manifold faces at {path}")
    return PreparedVolume(points, velocities, density, mu, lam)


def _elastic_parameters(value):
    young, poisson = value.material.get("youngsModulus"), value.material.get("poissonsRatio")
    if young is None:
        return None, None
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    if any(not np.isfinite(number) or number > float(np.finfo(np.float32).max) for number in (mu, lam)):
        raise InvalidPhysicsError(f"volume Lame parameters exceed finite float32 storage at {value.geometry_path}")
    return mu, lam


def build_volume(builder, value, prepared):
    """Append a prevalidated volume and retain its ownership through finalize."""
    frame = world_frame(value)
    start = builder.particle_count
    volume_index = len(builder.volume_label)
    builder.add_soft_mesh(
        pos=wp.vec3(0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0),
        vertices=prepared.points.tolist(),
        indices=value.payload.tet_vertex_indices.reshape(-1).tolist(),
        density=prepared.density,
        k_mu=prepared.mu,
        k_lambda=prepared.lam,
        validate_mesh=True,
        label=value.geometry_path,
    )
    if len(prepared.velocities):
        builder.particle_qd[start : builder.particle_count] = prepared.velocities.tolist()
    for name, dtype, data in (
        ("volume_path", str, value.geometry_path),
        ("volume_world_transform", wp.transformf, frame),
    ):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=name, dtype=dtype, frequency=newton.Model.AttributeFrequency.VOLUME, namespace="ovnewton"
            )
        )
        builder.custom_attributes[f"ovnewton:{name}"].values[volume_index] = data
