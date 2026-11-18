# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical open curves built as Newton rods, with persistent USD ownership."""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from ._deformables import (
    CurvesPayload,
    constant_thickness,
    reject_unsupported_material,
    resolve_density,
    validate_common,
    validate_particle_masses,
    world_frame,
    world_particle_data,
)
from ._errors import InvalidPhysicsError, UnsupportedPhysicsError


@dataclass(frozen=True)
class CurveBuildRange:
    geometry_path: str
    body_start: int
    body_end: int
    joint_start: int
    joint_end: int
    world_transform: tuple[float, ...]


@dataclass(frozen=True)
class PreparedCurve:
    rod: newton.Rod
    velocity: np.ndarray
    masses: np.ndarray
    half_lengths: np.ndarray


@dataclass(frozen=True)
class PreparedCurves:
    curves: tuple[PreparedCurve, ...]
    density: float


def prepare_curves(value, default_density):
    """Resolve geometry and structural rigidities before mutating a builder."""
    path = value.geometry_path
    validate_common(value)
    radius = constant_thickness(value) * 0.5
    if not np.isfinite(radius) or radius <= 0 or radius > float(np.finfo(np.float32).max):
        raise InvalidPhysicsError(f"cable radius must be finite and positive at {path}")
    names = ("curvesStretchStiffness", "curvesShearStiffness", "curvesBendStiffness", "curvesTwistStiffness")
    reject_unsupported_material(value, {"density", "youngsModulus", "poissonsRatio", *names})
    density = resolve_density(value, default_density)
    young = value.material.get("youngsModulus")
    poisson = value.material.get("poissonsRatio")
    young = 1e6 * value.meters_per_unit if young is None else young
    poisson = 0.3 if poisson is None else poisson
    if not np.isfinite(young) or young < 0 or not np.isfinite(poisson) or not -1 < poisson < 0.5:
        raise InvalidPhysicsError(f"invalid cable elastic material at {path}")
    shear = young / (2 * (1 + poisson))
    area, moment = np.pi * radius**2, np.pi * radius**4 / 4
    defaults = (young * area, 0.9 * shear * area, young * moment, shear * 2 * moment)
    rigidities = tuple(
        value.material.get(name) if value.material.get(name) is not None else fallback
        for name, fallback in zip(names, defaults)
    )
    if any(not np.isfinite(number) or number < 0 for number in rigidities):
        raise InvalidPhysicsError(f"cable structural stiffness must be finite and nonnegative at {path}")
    payload = value.payload
    if not isinstance(payload, CurvesPayload):
        raise TypeError("cable deformable has an incompatible payload")
    if np.any(payload.curve_vertex_counts < 3):
        raise UnsupportedPhysicsError(f"cable requires at least three points per curve at {path}")
    points, velocities = world_particle_data(value)
    normals = np.asarray(payload.normals, dtype=np.float64) @ value.world_matrix[:3, :3]
    result = []
    offset = 0
    for count in payload.curve_vertex_counts:
        count = int(count)
        curve = points[offset : offset + count]
        edges = np.diff(curve.astype(np.float64), axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 1e-8):
            raise InvalidPhysicsError(f"cable has a zero-length or unrepresentable segment at {path}")
        gains = np.asarray(rigidities)[:, None] / (0.5 * (lengths[:-1] + lengths[1:]))
        if np.any(gains > float(np.finfo(np.float32).max)):
            raise InvalidPhysicsError(f"cable stiffness exceeds finite float32 storage at {path}")
        masses = density * area * lengths
        validate_particle_masses(masses, path)
        validate_particle_masses(masses * radius**2, path, quantity="inertia term")
        frames = None
        if len(normals):
            tangent = edges / lengths[:, None]
            director = normals[offset : offset + count - 1]
            director = director - np.sum(director * tangent, axis=1)[:, None] * tangent
            norm = np.linalg.norm(director, axis=1)
            if not np.all(np.isfinite(norm)) or np.any(norm <= 1e-8):
                raise InvalidPhysicsError(f"cable normals must define a transverse material frame at {path}")
            director /= norm[:, None]
            frames = [
                tuple(wp.quat_from_matrix(wp.mat33(*np.column_stack((x, np.cross(z, x), z)).ravel())))
                for x, z in zip(director, tangent)
            ]
        velocity = np.zeros(3, dtype=np.float32)
        if len(velocities):
            samples = velocities[offset : offset + count]
            if not np.all(samples == samples[0]):
                raise UnsupportedPhysicsError(f"cable supports uniform initial velocity per curve at {path}")
            velocity = samples[0]
        rod = newton.Rod(
            points=curve,
            quaternions=frames,
            radius=radius,
            stretch_rigidity=rigidities[0],
            shear_rigidity=rigidities[1],
            bend_rigidity=rigidities[2],
            twist_rigidity=rigidities[3],
        )
        result.append(PreparedCurve(rod, velocity, masses, lengths * 0.5))
        offset += count
    return PreparedCurves(tuple(result), density)


def build_curves(builder, value, prepared):
    cfg = builder.default_shape_cfg.copy()
    cfg.density = prepared.density
    frame = world_frame(value)
    for name, dtype, frequency in (
        ("curve_path", str, newton.Model.AttributeFrequency.CURVE),
        ("curve_world_transform", wp.transformf, newton.Model.AttributeFrequency.CURVE),
        ("curve_half_length", wp.float32, newton.Model.AttributeFrequency.BODY),
    ):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(name=name, dtype=dtype, frequency=frequency, namespace="ovnewton")
        )
    for index, curve in enumerate(prepared.curves):
        curve_index = len(builder.curve_label)
        bodies, _ = builder.add_rod(rod=curve.rod, cfg=cfg, label=f"{value.geometry_path}/curve_{index}")
        builder.custom_attributes["ovnewton:curve_path"].values[curve_index] = value.geometry_path
        builder.custom_attributes["ovnewton:curve_world_transform"].values[curve_index] = frame
        for body, length, mass in zip(bodies, curve.half_lengths, curve.masses):
            # Capsule endcaps are collision geometry, not extra cable mass.
            # Preserve Newton's inertia shape while applying the canonical line mass.
            ratio = float(mass) / builder.body_mass[body]
            builder.body_mass[body] = float(mass)
            builder.body_inv_mass[body] = 1.0 / float(mass)
            builder.body_inertia[body] *= ratio
            builder.body_inv_inertia[body] *= 1.0 / ratio
            builder.custom_attributes["ovnewton:curve_half_length"].values[body] = float(length)
            builder.body_qd[body] = wp.spatial_vector(*curve.velocity, 0, 0, 0)


def model_curve_ranges(model):
    attributes = getattr(model, "ovnewton", None)
    paths = getattr(attributes, "curve_path", ())
    if not any(paths):
        return ()
    columns = [
        getattr(model, name).numpy().tolist()
        for name in ("curve_body_start", "curve_body_end", "curve_joint_start", "curve_joint_end")
    ]
    frames = attributes.curve_world_transform.numpy().tolist()
    return tuple(
        CurveBuildRange(path, *(column[i] for column in columns), tuple(frames[i]))
        for i, path in enumerate(paths)
        if path
    )
