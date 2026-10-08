# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Populate a Newton ``ModelBuilder`` from a populated ovstage.

:func:`add_ovstage` reconstructs bodies + colliders + mass + joints from the
readable surface (discovery and values via :mod:`._parse`), decodes
``omni:fabric:worldMatrix`` natively (:func:`_decode_pose`), and constructs the
builder with the helpers in this module (geometry dispatch, mass preparation,
the joint pipeline ``_build_joints``). :func:`build_model` remains the
compatibility path that also finalizes the populated builder.

Prim identity and hierarchy come from ovstage's readable built-in metadata, so
body/collider/joint paths remain available without loading USD.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations, combinations_with_replacement, product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import newton
import numpy as np
import warp as wp

from . import _parse, _stage
from ._errors import InvalidPhysicsError, OvstageContractError, UnsupportedPhysicsError, log_diagnostic
from ._parse import _DofDesc, _JointDesc
from ._schema_names import (
    BODY_ANGULAR_VELOCITY,
    BODY_VELOCITY,
    NEWTON_INERTIA,
    RIGID_BODY_ENABLED,
    WORLD_MATRIX,
)

# USD gprim/joint axis token -> Newton axis. Shared by collider shape placement
# (capsule/cylinder/cone) and joint axis selection.
_AXIS = {"X": newton.Axis.X, "Y": newton.Axis.Y, "Z": newton.Axis.Z}

# USD authors angular joint drives in degrees; Newton's revolute DOF is radians.
_DEG_PER_RAD = 180.0 / np.pi  # degrees per radian
_RAD_PER_DEG = np.pi / 180.0  # radians per degree
# add_usd lowers the +inf schema sentinel to this finite stiffness.
_HARD_LIMIT_KE = 1.0e8

_MESH_APPROXIMATION_METHOD = {
    "convexDecomposition": "coacd",
    "convexHull": "convex_hull",
    "boundingSphere": "bounding_sphere",
    "boundingCube": "bounding_box",
    "meshSimplification": "quadratic",
}

_SDF_CONFIG_FIELDS = {
    "sdf_max_resolution",
    "sdf_narrow_band_range",
    "sdf_padding",
    "sdf_target_voxel_size",
    "sdf_texture_format",
}
_SDF_TEXTURE_FORMATS = {"float32", "uint16", "uint8"}


def _shape_cfg(
    builder: Any,
    density: Optional[float],
    material: Optional[Dict[str, Optional[float]]] = None,
    *,
    collision_enabled: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Overlay resolved mass/contact material values on the builder defaults."""
    if (
        (density is None or density <= 0.0)
        and material is None
        and collision_enabled
        and not overrides
    ):
        return None
    cfg = builder.default_shape_cfg.copy()
    cfg.has_shape_collision = collision_enabled
    if density is not None and density > 0.0:
        cfg.density = float(density)
    if overrides:
        cfg = replace(cfg, **overrides)
    if material is not None:
        for field in ("mu", "restitution", "mu_torsional", "mu_rolling", "ke", "kd", "kf", "ka"):
            value = material.get(field)
            if value is not None:
                setattr(cfg, field, float(value))
    return cfg


def _shape_options(builder: Any, collider: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[float]]:
    """Resolve schema-owned ``ShapeConfig`` fields and the post-inertia margin."""
    options: Dict[str, Any] = {}
    mass_model = collider.get("mass_model")
    if mass_model is not None:
        options["is_solid"] = mass_model != "shell"
    contact_margin = collider.get("contact_margin")
    if contact_margin is not None:
        options["margin"] = contact_margin
    contact_gap = collider.get("contact_gap")
    if contact_gap is not None and not np.isneginf(contact_gap):
        if np.isfinite(contact_gap) and contact_gap >= 0.0:
            options["gap"] = contact_gap
        else:
            log_diagnostic(
                "contact-gap-ignored",
                "newton:contactGap must be finite and non-negative; the builder default was used",
                path=collider["path"],
            )
    shell_thickness = collider.get("shell_thickness")
    restore_margin = None
    if shell_thickness is not None and np.isfinite(shell_thickness):
        restore_margin = contact_margin if contact_margin is not None else builder.default_shape_cfg.margin
        if shell_thickness >= 0.0:
            options["margin"] = shell_thickness
        else:
            log_diagnostic(
                "negative-shell-thickness-ignored",
                "negative newton:shellThickness was ignored; the contact margin was used",
                path=collider["path"],
            )

    if collider.get("sdf_api"):
        options.update(_sdf_shape_options(builder, collider))
    return options, restore_margin


def _sdf_shape_options(builder: Any, collider: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve applied ``NewtonSDFCollisionAPI`` values for ``ShapeConfig``."""
    options: Dict[str, Any] = {}
    path = collider["path"]
    if collider.get("newton_mesh_api"):
        log_diagnostic(
            "sdf-mesh-collision-api-ignored",
            "NewtonSDFCollisionAPI and NewtonMeshCollisionAPI overlap; SDF configuration was used",
            path=path,
        )
    target_voxel_size = collider.get("sdf_target_voxel_size")
    if target_voxel_size is not None and np.isneginf(target_voxel_size):
        target_voxel_size = None
    elif target_voxel_size is not None and (not np.isfinite(target_voxel_size) or target_voxel_size <= 0.0):
        log_diagnostic(
            "sdf-target-voxel-size-ignored",
            "newton:sdfTargetVoxelSize must be finite and positive; the default resolution was used",
            path=path,
        )
        target_voxel_size = builder.default_shape_cfg.sdf_target_voxel_size

    max_resolution = collider.get("sdf_max_resolution")
    if max_resolution is not None and (max_resolution <= 0 or max_resolution % 8):
        log_diagnostic(
            "sdf-max-resolution-ignored",
            "newton:sdfMaxResolution must be positive and divisible by 8; the schema default was used",
            path=path,
        )
        max_resolution = None
    if target_voxel_size is not None:
        if max_resolution not in (None, 64):
            log_diagnostic(
                "sdf-resolution-conflict",
                "newton:sdfTargetVoxelSize takes precedence over newton:sdfMaxResolution",
                path=path,
            )
        max_resolution = None
        options["sdf_target_voxel_size"] = target_voxel_size
    else:
        options["sdf_max_resolution"] = max_resolution if max_resolution is not None else 64

    inner = collider.get("sdf_narrow_band_inner")
    outer = collider.get("sdf_narrow_band_outer")
    default_inner, default_outer = builder.default_shape_cfg.sdf_narrow_band_range
    if inner is None or np.isneginf(inner):
        inner = default_inner
    elif not np.isfinite(inner):
        log_diagnostic(
            "sdf-narrow-band-ignored",
            "newton:sdfNarrowBandInner must be finite; the builder default was used",
            path=path,
        )
        inner = default_inner
    if outer is None or np.isneginf(outer):
        outer = default_outer
    elif not np.isfinite(outer):
        log_diagnostic(
            "sdf-narrow-band-ignored",
            "newton:sdfNarrowBandOuter must be finite; the builder default was used",
            path=path,
        )
        outer = default_outer
    options["sdf_narrow_band_range"] = (inner, outer)

    texture_format = collider.get("sdf_texture_format")
    if texture_format is not None and texture_format not in _SDF_TEXTURE_FORMATS:
        log_diagnostic(
            "sdf-texture-format-ignored",
            f"unsupported newton:sdfTextureFormat={texture_format!r}; the builder default was used",
            path=path,
        )
        texture_format = None
    options["sdf_texture_format"] = texture_format or builder.default_shape_cfg.sdf_texture_format

    padding = collider.get("sdf_padding")
    if padding is not None and np.isneginf(padding):
        padding = None
    elif padding is not None and (not np.isfinite(padding) or padding < 0.0):
        log_diagnostic(
            "sdf-padding-ignored",
            "newton:sdfPadding must be finite and non-negative; the builder default was used",
            path=path,
        )
        padding = None
    if padding is not None:
        options["sdf_padding"] = padding

    stiffness = collider.get("hydroelastic_stiffness")
    if stiffness is not None and (not np.isfinite(stiffness) or stiffness <= 0.0):
        if not np.isneginf(stiffness):
            log_diagnostic(
                "hydroelastic-stiffness-ignored",
                "newton:hydroelasticStiffness must be finite and positive; the builder default was used",
                path=path,
            )
        stiffness = None
    options["is_hydroelastic"] = bool(collider.get("hydroelastic_enabled"))
    options["kh"] = stiffness if stiffness is not None else builder.default_shape_cfg.kh
    return options


def _mesh_shape_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Remove deferred mesh-SDF fields that ``add_shape_mesh`` rejects."""
    return {key: value for key, value in options.items() if key not in _SDF_CONFIG_FIELDS | {"is_hydroelastic"}}


def _apply_deferred_mesh_sdf(builder: Any, shape: int, options: Dict[str, Any]) -> None:
    """Store mesh SDF intent for Newton's finalize-time construction."""
    for field in _SDF_CONFIG_FIELDS:
        if field in options:
            getattr(builder, f"shape_{field}")[shape] = options[field]
    if options.get("is_hydroelastic"):
        builder.shape_flags[shape] |= newton.ShapeFlags.HYDROELASTIC


def _wp_xform(t: np.ndarray, q: np.ndarray) -> "wp.transformf":
    """Build a Warp ``transformf`` from a translation row (3,) + quat row (4,).

    The quaternion is already in ``(x,y,z,w)`` order: ovstage stores USD
    ``quatf`` columns (including ``physics:localRot0/1``) with the real part in
    the last slot, matching ``wp.quat`` and newton's ``value_to_warp``."""
    return wp.transform(
        wp.vec3(float(t[0]), float(t[1]), float(t[2])), wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    )


def _relative_xform(
    body_pose: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    collider_pose: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
) -> Optional["wp.transformf"]:
    """Pose of a collider expressed in its body's frame: ``inv(T_body) ∘
    T_collider``, from the two rigid (scale-free) world poses ``(trans, quat)``.

    A collider attached to a rigid-body ancestor (e.g. a cartpole rail's child
    ``Cube``) must be placed at its offset *within* that body, not at its world
    pose. Returns ``None`` (→ identity, collider at the body origin) when either
    pose is missing; for a collider on the body prim itself the two poses are
    equal and this is identity."""
    bt, bq = body_pose
    ct, cq = collider_pose
    if bt is None or ct is None:
        return None
    return wp.transform_multiply(wp.transform_inverse(_wp_xform(bt, bq)), _wp_xform(ct, cq))


# ─── Analytic-shape dispatch (prim_type → Newton shape) ──────────────


def _resolve_geometry(
    prim_type: str,
    *,
    half: Optional[Tuple[float, float, float]],
    radius: Optional[float],
    size: Optional[float],
    height: Optional[float],
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    axis: str = "Z",
) -> Optional[Tuple[str, Dict[str, float]]]:
    """Resolve a gprim's geometry to ``(kind, dims)`` from already-read
    dimensions — the single source of truth for shape creation
    (:func:`_add_analytic_shape`). ``kind`` ∈ {'sphere', 'box', 'capsule',
    'cylinder', 'cone'}; ``None`` when nothing can be sized. Typed gprims use their
    authored size. Unsupported types return ``None``.

    ``scale`` is the gprim's world scale (column norms of its worldMatrix);
    it is folded into the returned dimensions so a unit gprim under a scaled
    xform — e.g. a cartpole rail's ``Cube`` at scale ``(0.03, 8, 0.03)`` — gets
    its true size, matching how UsdPhysics / ``add_usd`` bake xform scale into
    collision geometry. A non-uniformly scaled sphere uses the largest scale
    component. For axial shapes, height uses the authored-axis component and
    radius uses the largest perpendicular component. The same axis-aware rule
    applies to collision shapes and sites."""
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    if prim_type == "Sphere":
        r = radius if radius is not None else (max(half) if half is not None else 0.5)
        return "sphere", {"radius": float(r * max(sx, sy, sz))}
    if prim_type == "Cube":
        if size is not None:
            hx = hy = hz = size / 2.0
        elif half is not None:
            hx, hy, hz = half
        else:
            hx = hy = hz = 0.5
        return "box", {"hx": float(hx * sx), "hy": float(hy * sy), "hz": float(hz * sz)}
    if prim_type in ("Capsule", "Cylinder", "Cone"):
        r = radius if radius is not None else (max(half[0], half[1]) if half is not None else 0.5)
        hh = (height / 2.0) if height is not None else (half[2] if half is not None else 0.5)
        axial = {"X": sx, "Y": sy, "Z": sz}.get(axis, sz)
        radial = {"X": max(sy, sz), "Y": max(sx, sz), "Z": max(sx, sy)}.get(axis, max(sx, sy))
        kind = prim_type.lower()
        return kind, {"radius": float(r * radial), "half_height": float(hh * axial)}
    return None


def _analytic_geometry_kwargs(shape: Dict[str, Any]) -> Dict[str, Any]:
    extent = shape["extent"]
    half = (
        tuple(float((extent[3 + axis] - extent[axis]) * 0.5) for axis in range(3))
        if extent is not None and len(extent) >= 6
        else None
    )
    return {
        "half": half,
        "radius": float(shape["radius"][0]) if shape["radius"] is not None else None,
        "size": float(shape["size"][0]) if shape["size"] is not None else None,
        "height": float(shape["height"][0]) if shape["height"] is not None else None,
    }


def _add_analytic_shape(
    builder: Any,
    prim_type: str,
    *,
    body: int,
    xform: Optional[Any],
    half: Optional[Tuple[float, float, float]],
    radius: Optional[float] = None,
    size: Optional[float] = None,
    height: Optional[float] = None,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    density: Optional[float] = None,
    material: Optional[Dict[str, Optional[float]]] = None,
    collision_enabled: bool = True,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    axis: Optional[str] = None,
    label: Optional[str] = None,
    as_site: bool = False,
) -> int:
    """Add the Newton analytic shape matching ``prim_type`` to ``body``, placed
    at ``xform`` in the body's frame and sized via :func:`_resolve_geometry`.

    ``as_site`` delegates the massless, visible, non-colliding invariants to
    ``ModelBuilder`` and ignores collision configuration.

    When ``density`` is given (the body's authored ``physics:density``), the shape
    carries it so Newton accumulates the body's mass / inertia from it; otherwise
    the builder's default shape density is used. ``physics:mass`` then overrides
    the result in :func:`_finalize_body_mass`, exactly as ``add_usd`` does.

    ``axis`` is the gprim's local long axis for capsule/cylinder/cone. Newton's
    axial analytic shapes are canonical-Z, so a non-Z axis rotates the shape
    frame by the rotation from Z to ``axis``, matching ``add_usd``, which
    reorients the (anisotropic)
    inertia as well. The authored radius/half-height are axis-independent;
    non-uniform xform scale is resolved against the authored axis before this
    canonical-Z rotation.
    """
    geo = _resolve_geometry(
        prim_type,
        half=half,
        radius=radius,
        size=size,
        height=height,
        scale=scale,
        axis=axis or "Z",
    )
    if geo is None:
        raise InvalidPhysicsError(f"could not construct {prim_type} shape at {label or '<unknown>'}")
    kind, d = geo
    ax = _AXIS.get(axis or "Z", newton.Axis.Z)
    if kind in ("capsule", "cylinder", "cone") and ax != newton.Axis.Z:
        rot = wp.quat_between_vectors(newton.Axis.Z.to_vec3(), ax.to_vec3())
        xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), rot) if xform is None else wp.transform(xform.p, xform.q * rot)
    cfg = None
    if not as_site:
        cfg = _shape_cfg(
            builder,
            density,
            material,
            collision_enabled=collision_enabled,
            overrides=cfg_overrides,
        )
    if kind == "sphere":
        return builder.add_shape_sphere(
            body=body, xform=xform, radius=d["radius"], cfg=cfg, as_site=as_site, label=label
        )
    if kind == "box":
        hx, hy, hz = d["hx"], d["hy"], d["hz"]
        return builder.add_shape_box(
            body=body, xform=xform, hx=hx, hy=hy, hz=hz, cfg=cfg, as_site=as_site, label=label
        )
    if kind == "capsule":
        return builder.add_shape_capsule(
            body=body,
            xform=xform,
            radius=d["radius"],
            half_height=d["half_height"],
            cfg=cfg,
            as_site=as_site,
            label=label,
        )
    add_shape = builder.add_shape_cone if kind == "cone" else builder.add_shape_cylinder
    return add_shape(
        body=body,
        xform=xform,
        radius=d["radius"],
        half_height=d["half_height"],
        cfg=cfg,
        as_site=as_site,
        label=label,
    )


def _fan_triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Convert USD polygon faces to a flat triangle index array."""
    counts = np.asarray(counts, dtype=np.int32).reshape(-1)
    indices = np.asarray(indices, dtype=np.int32).reshape(-1)
    if not counts.size or np.any(counts < 3):
        raise InvalidPhysicsError("mesh faces must contain at least three vertices")
    if int(np.sum(counts)) != indices.size:
        raise InvalidPhysicsError(f"mesh face counts total {int(np.sum(counts))}, but there are {indices.size} indices")
    if counts.size and np.all(counts == 3):
        return indices
    tris: List[int] = []
    offset = 0
    for count in counts.tolist():
        for k in range(1, count - 1):
            tris.extend((int(indices[offset]), int(indices[offset + k]), int(indices[offset + k + 1])))
        offset += count
    return np.asarray(tris, dtype=np.int32)


def _mesh_geometry(collider: Dict[str, Any]) -> newton.Mesh:
    """Construct Newton mesh geometry from a collider's ovstage arrays."""
    points = collider.get("points")
    counts = collider.get("face_counts")
    indices = collider.get("face_indices")
    path = collider.get("path", "<unknown collider>")
    if points is None or counts is None or indices is None:
        raise InvalidPhysicsError(f"mesh collider {path} is missing points or face topology")
    points = np.asarray(points, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int32).reshape(-1)
    if points.size < 9 or points.size % 3:
        raise InvalidPhysicsError(f"mesh collider {path} has an invalid point array")
    try:
        triangles = _fan_triangulate(counts, indices)
    except InvalidPhysicsError as exc:
        raise InvalidPhysicsError(f"invalid mesh topology at {path}: {exc}") from exc
    point_count = points.size // 3
    if np.any(triangles < 0) or np.any(triangles >= point_count):
        raise InvalidPhysicsError(f"mesh collider {path} contains an out-of-range vertex index")
    if collider.get("orientation") == "leftHanded":
        triangles = triangles.reshape(-1, 3)[:, ::-1].reshape(-1)
    return newton.Mesh(
        points.reshape(-1, 3),
        triangles,
        maxhullvert=collider.get("max_hull_vertices"),
    )


# ─── Body mass / inertia / CoM consumption ───────────────────────────


def _inv_inertia(I: "wp.mat33") -> "wp.mat33":
    """Inverse of a nonsingular inertia tensor, else a zero matrix."""
    if float(np.linalg.det(np.asarray(I, dtype=float).reshape(3, 3))) > 0.0:
        return wp.inverse(I)
    return wp.mat33(0.0)


def _newton_inertia_tensor(values: Optional[Any], path: str) -> Optional[np.ndarray]:
    """Validate and expand Newton's compact symmetric inertia tensor."""
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reason = None
    if len(values) != 6:
        reason = f"has {len(values)} elements, expected 6"
    elif not np.all(np.isfinite(values)):
        reason = "contains non-finite values"
    elif np.any(values[:3] < 0.0):
        reason = "has negative diagonal elements"
    else:
        ixx, iyy, izz, ixy, ixz, iyz = values
        inertia = np.array(((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)), dtype=np.float64)
        if np.any(np.linalg.eigvalsh(inertia) < 0.0):
            reason = "is not positive semidefinite"
        else:
            return inertia
    log_diagnostic(
        "newton-inertia-ignored",
        f"newton:inertia {reason}; standard or collider-derived inertia was retained",
        path=path,
    )
    return None


# ── UsdPhysics value sentinels (design §6) ──────────────────────────
# ovstage carries no per-prim *authored* bit and bakes schema defaults, so
# "is this value authored?" is decided by the value itself. These predicates are
# the single source of that convention; every place that consumes a PhysicsMassAPI
# value asks one of them, so the rule can't drift as more attributes are added.
#
# Compatibility note: reconstructing the authored-vs-fallback cascade from value
# sentinels is ovpopulation's job — it has the authored bit, the consumer doesn't.
# Retire these predicates once ovpopulation provides resolved mass/inertia/CoM.


def _authored_scalar(v: Optional[float]) -> bool:
    """An authored mass/density is > 0 (the unauthored default is 0)."""
    return v is not None and v > 0.0


def _authored_inertia(diag: Optional[Tuple[float, float, float]]) -> bool:
    """An authored diagonalInertia has norm > 0 (the default is (0, 0, 0))."""
    return diag is not None and float(np.linalg.norm(diag)) > 0.0


def _authored_point(p: Optional[Any]) -> bool:
    """A finite 3-vector — an authored centerOfMass, as opposed to UsdPhysics's
    (-inf, -inf, -inf) "derive from colliders" default (also used to detect a
    shapeless body's uninitialised, non-finite accumulated CoM)."""
    return p is not None and bool(np.all(np.isfinite(np.asarray(p, dtype=float))))


def _finalize_body_mass(
    builder: Any,
    body: int,
    *,
    body_scale: Sequence[float],
    mass: Optional[float],
    inertia_tensor: Optional[np.ndarray],
    inertia_diag: Optional[Tuple[float, float, float]],
    principal_axes: Optional[Tuple[float, float, float, float]],
    com: Optional[Tuple[float, float, float]],
) -> None:
    """Apply authored body mass properties over collider-derived values.

    ``newton:inertia`` takes precedence over standard diagonal inertia. An
    authored mass without inertia rescales the collider-derived inertia.

    A body with neither colliders nor an authored CoM would carry Newton's
    uninitialised sentinel into the solver (a non-finite CoM → NaN ``QACC``); we
    default such a CoM to the body origin, matching ``add_usd``.
    """
    if body < 0:
        return

    has_inertia = inertia_tensor is not None or _authored_inertia(inertia_diag)
    if has_inertia:
        inertia = inertia_tensor
        if inertia is None:
            inertia = np.diag(np.asarray(inertia_diag, dtype=np.float32))
            q = np.asarray(principal_axes, dtype=np.float32) if principal_axes is not None else np.zeros(4)
            if float(np.linalg.norm(q)) > 0.0:
                q /= np.linalg.norm(q)
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*q)), dtype=np.float32).reshape(3, 3)
                inertia = rotation @ inertia @ rotation.T
        I = wp.mat33(inertia)
        builder.body_inertia[body] = I
        builder.body_inv_inertia[body] = _inv_inertia(I)

    if _authored_scalar(mass):
        accumulated = builder.body_mass[body]
        # Authored mass, no authored inertia → scale the shape-accumulated
        # inertia so it stays consistent with the new mass (matches add_usd).
        if not has_inertia and accumulated > 0.0:
            mass_scale = mass / accumulated
            scaled = wp.mat33(np.array(builder.body_inertia[body]) * mass_scale)
            builder.body_inertia[body] = scaled
            builder.body_inv_inertia[body] = _inv_inertia(scaled)
        builder.body_mass[body] = mass
        builder.body_inv_mass[body] = 1.0 / mass

    if _authored_point(com):
        scaled_com = np.asarray(com) * np.asarray(body_scale)
        builder.body_com[body] = wp.vec3(*scaled_com)
    elif not _authored_point(builder.body_com[body]):
        builder.body_com[body] = wp.vec3(0.0, 0.0, 0.0)


# ─── Public API ──────────────────────────────────────────────────────


@dataclass
class _Bodies:
    """Resolved rigid-body state shared across importer passes."""

    paths: List[str]  # body index -> prim path
    index: Dict[str, int]  # prim path -> body index
    world: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]  # rigid world pose
    authored: Dict[str, Dict[str, Any]]  # applied mass-schema values
    scale: Dict[str, np.ndarray]  # world scale, for baking joint anchors
    kinematic: Dict[str, bool]  # PhysicsRigidBodyAPI mode


@dataclass(frozen=True)
class _ResolvedJoint:
    path: str
    desc: _JointDesc
    parent: int
    child: int
    parent_xform: Any
    child_xform: Any


@dataclass(frozen=True)
class _JointBuildRecord:
    sources: Tuple[_ResolvedJoint, ...]
    parent: int
    child: int

    def __post_init__(self) -> None:
        if not self.sources:
            if self.parent != -1:
                raise ValueError("synthetic free joint must be world-rooted")
            return
        if any((source.parent, source.child) != (self.parent, self.child) for source in self.sources):
            raise ValueError("merged joint sources must share one body pair")


def _prescan_articulation_bodies(joint_descs: List[Tuple[str, "_JointDesc"]]) -> set:
    """Find bodies whose explicit joints must remain their sole connection."""
    paths: set = set()
    for _jpath, d in joint_descs:
        if d.body0:
            paths.add(d.body0)
        if d.body1:
            paths.add(d.body1)
    return paths


def _finalize_masses(builder: Any, bodies: _Bodies) -> None:
    """Apply each body's authored mass properties over shape-derived values."""
    for path, b in bodies.index.items():
        a = bodies.authored[path]
        _finalize_body_mass(
            builder,
            b,
            body_scale=bodies.scale[path],
            mass=a["mass"],
            inertia_tensor=a["inertia_tensor"],
            inertia_diag=a["inertia"],
            principal_axes=a["principal_axes"],
            com=a["com"],
        )


def _joint_records(joint_descs: List[Tuple[str, "_JointDesc"]], bodies: _Bodies) -> List[_JointBuildRecord]:
    """Resolve scaled anchors; rebase world joints to preserve the body pose."""
    ones = np.ones(3)
    records = []
    for jpath, d in joint_descs:
        for relation, body_path in (("body0", d.body0), ("body1", d.body1)):
            if body_path is not None and body_path not in bodies.index:
                raise InvalidPhysicsError(f"joint {jpath} physics:{relation} targets non-rigid body {body_path}")
        if d.body0 is None and d.body1 is None:
            raise InvalidPhysicsError(f"joint {jpath} does not target a rigid body")
        parent = bodies.index.get(d.body0, -1)
        child = bodies.index.get(d.body1, -1)
        pos0 = np.array(d.local_pos0) * bodies.scale.get(d.body0, ones)
        pos1 = np.array(d.local_pos1) * bodies.scale.get(d.body1, ones)
        rot0 = np.asarray(d.local_rot0, dtype=float)
        rot1 = np.asarray(d.local_rot1, dtype=float)
        if child == -1 and parent != -1:
            parent, child = -1, parent
            pos0, pos1 = pos1, pos0
            rot0, rot1 = rot1, rot0
        pxf = _wp_xform(pos0, rot0)
        cxf = _wp_xform(pos1, rot1)
        if parent == -1:
            body_path = bodies.paths[child]
            body_t, body_q = bodies.world[body_path]
            if body_t is None or body_q is None:
                raise OvstageContractError(f"jointed rigid body has no world transform: {body_path}")
            pxf = wp.transform_multiply(_wp_xform(body_t, body_q), cxf)
        joint = _ResolvedJoint(jpath, d, parent, child, pxf, cxf)
        records.append(_JointBuildRecord((joint,), parent, child))
    return records


def _joint_components(records: List[_JointBuildRecord]) -> List[List[_JointBuildRecord]]:
    """Partition body edges without connecting components through the world."""
    uf: Dict[int, int] = {}

    def find(x: int) -> int:
        uf.setdefault(x, x)
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    for record in records:
        p, c = record.parent, record.child
        if p != -1 and c != -1:
            uf[find(p)] = find(c)

    groups: Dict[int, List[_JointBuildRecord]] = {}
    order: List[int] = []
    for record in records:
        key = find(record.child if record.child != -1 else record.parent)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    return [groups[k] for k in order]


def _with_free_base(records: List[_JointBuildRecord]) -> List[_JointBuildRecord]:
    """Root each never-a-child body with a synthetic FREE joint."""
    children = {record.child for record in records}
    members = {record.parent for record in records if record.parent != -1} | children
    roots = sorted(b for b in members if b != -1 and b not in children)
    base = [_JointBuildRecord((), -1, body) for body in roots]
    return base + records


def _order_joints_root_first(records: List[_JointBuildRecord]) -> List[_JointBuildRecord]:
    """Order joints so Newton sees each parent before its child."""
    ordered: List[_JointBuildRecord] = []
    produced: set = set()
    remaining = list(records)
    while remaining:
        ready = [record for record in remaining if record.parent == -1 or record.parent in produced]
        if not ready:
            paths = [path for record in remaining for path in _record_paths(record)]
            raise InvalidPhysicsError("joint graph is cyclic or cannot be rooted at the world: " + ", ".join(paths))
        ready_ids = {id(r) for r in ready}
        for record in ready:
            ordered.append(record)
            produced.add(record.child)
        remaining = [record for record in remaining if id(record) not in ready_ids]
    return ordered


def _dof_kw(d: Any, rotational: bool, lo: Optional[float], hi: Optional[float]) -> Dict[str, Any]:
    """Translate one normalized USD degree of freedom into ModelBuilder kwargs."""
    angle = _RAD_PER_DEG if rotational else 1.0
    angular_gain = _DEG_PER_RAD if rotational else 1.0
    kw: Dict[str, Any] = {}
    if lo is not None:
        kw["limit_lower"] = lo * angle
    if hi is not None:
        kw["limit_upper"] = hi * angle
    limit_ke = None
    limit_kd = None
    broadcast_ke = d.broadcast_limit_stiffness
    broadcast_kd = d.broadcast_limit_damping
    if broadcast_ke == float("inf"):
        limit_ke = _HARD_LIMIT_KE
    elif broadcast_ke is not None and broadcast_ke != float("-inf"):
        limit_ke = broadcast_ke
    if broadcast_ke == float("inf") or broadcast_kd == float("inf"):
        limit_kd = 0.0
    elif broadcast_kd is not None and broadcast_kd != float("-inf"):
        limit_kd = broadcast_kd
    for name, value in (
        ("limit_ke", limit_ke),
        ("limit_kd", limit_kd),
        ("target_ke", d.drive_stiffness),
        ("target_kd", d.drive_damping),
    ):
        if value is not None:
            kw[name] = value * angular_gain
    for name, value in (
        ("target_pos", d.drive_target),
        ("target_vel", d.drive_velocity),
        ("velocity_limit", d.velocity_limit),
    ):
        if value is not None:
            kw[name] = value * angle
    for name, value in (
        ("damping", d.damping * angular_gain if d.damping is not None else None),
        ("armature", d.armature),
        ("friction", d.friction),
        ("effort_limit", d.drive_force_limit),
    ):
        if value is not None:
            kw[name] = value
    if d.drive_enabled:
        kw["actuator_mode"] = newton.JointTargetMode.from_gains(
            d.drive_stiffness or 0.0, d.drive_damping or 0.0, has_drive=True
        )
    return kw


def _joint_dof_config(builder: Any, axis: Any, overrides: Dict[str, Any]) -> Any:
    """Overlay resolved USD values on the caller's default joint configuration."""
    defaults = builder.default_joint_cfg
    values = {
        name: getattr(defaults, name)
        for name in (
            "limit_lower",
            "limit_upper",
            "limit_ke",
            "limit_kd",
            "target_pos",
            "target_vel",
            "target_ke",
            "target_kd",
            "damping",
            "armature",
            "effort_limit",
            "velocity_limit",
            "friction",
            "actuator_mode",
        )
    }
    newton_defaults = newton.ModelBuilder.JointDofConfig()
    if "effort_limit" not in overrides and defaults.effort_limit == newton_defaults.effort_limit:
        # Match add_usd's unbounded D6 fallback unless the caller changed the builder default.
        values["effort_limit"] = float("inf")
    values.update(overrides)
    return newton.ModelBuilder.JointDofConfig(axis=axis, **values)


def _apply_initial_joint_state(builder: Any, joint: int, d: _DofDesc, dof: int = 0) -> None:
    """Convert authored USD angular positions and velocities to Newton radians."""
    if d.initial_position is not None:
        position = d.initial_position
        if d.rotational:
            position = math.radians(position)
        builder.joint_q[builder.joint_q_start[joint] + dof] = position
    if d.initial_velocity is not None:
        velocity = d.initial_velocity
        if d.rotational:
            velocity = math.radians(velocity)
        builder.joint_qd[builder.joint_qd_start[joint] + dof] = velocity


def _add_merged_d6(builder: Any, record: _JointBuildRecord) -> int:
    """Merge same-body-pair axes into the first joint's D6 frame."""
    group = record.sources
    rep = group[0]
    rep_q = rep.parent_xform.q
    rep_qa = np.array([rep_q[0], rep_q[1], rep_q[2], rep_q[3]], dtype=float)
    linear: List[Any] = []
    angular: List[Any] = []
    linear_records: List[_ResolvedJoint] = []
    angular_records: List[_ResolvedJoint] = []
    for r in group:
        d = r.desc
        dof = d.dofs[0]
        unit = list(dof.axis)
        jp_q = r.parent_xform.q
        if abs(float(np.dot(rep_qa, np.array([jp_q[0], jp_q[1], jp_q[2], jp_q[3]])))) > 1.0 - 1e-6:
            axis = (unit[0], unit[1], unit[2])  # same rotation — axis as-is
        else:
            rel = wp.mul(wp.quat_inverse(rep_q), jp_q)
            v = wp.quat_rotate(rel, wp.vec3(unit[0], unit[1], unit[2]))
            axis = (float(v[0]), float(v[1]), float(v[2]))
        lo = None if dof.limit_lower == float("-inf") else dof.limit_lower
        hi = None if dof.limit_upper == float("inf") else dof.limit_upper
        if d.prim_type == "PhysicsRevoluteJoint":
            kw: Dict[str, Any] = _dof_kw(dof, True, lo, hi)
            angular.append(_joint_dof_config(builder, axis, kw))
            angular_records.append(r)
        else:  # PhysicsPrismaticJoint — linear DOF, no unit conversion
            kw = _dof_kw(dof, False, lo, hi)
            linear.append(_joint_dof_config(builder, axis, kw))
            linear_records.append(r)
    joint = builder.add_joint_d6(
        parent=record.parent,
        child=record.child,
        parent_xform=rep.parent_xform,
        child_xform=rep.child_xform,
        linear_axes=linear or None,
        angular_axes=angular or None,
        # One disabling source joint wins for the merged body pair.
        collision_filter_parent=any(not r.desc.collision_enabled for r in group),
    )
    # Newton D6 coordinate/DOF order is linear first, then angular.
    for dof, r in enumerate(linear_records + angular_records):
        _apply_initial_joint_state(builder, joint, r.desc.dofs[0], dof)
    return joint


def _add_joint(builder: Any, record: _JointBuildRecord) -> int:
    """Build a synthetic, authored, or merged joint from its source count."""
    if not record.sources:
        return builder.add_joint_free(child=record.child)
    if len(record.sources) > 1:
        return _add_merged_d6(builder, record)
    source = record.sources[0]
    d = source.desc
    collision_filter_parent = not d.collision_enabled
    if d.prim_type == "PhysicsRevoluteJoint":
        dof = d.dofs[0]
        lo = None if dof.limit_lower == float("-inf") else dof.limit_lower
        hi = None if dof.limit_upper == float("inf") else dof.limit_upper
        rkw = _dof_kw(dof, True, lo, hi)
        joint = builder.add_joint_revolute(
            parent=record.parent,
            child=record.child,
            parent_xform=source.parent_xform,
            child_xform=source.child_xform,
            axis=dof.axis,
            collision_filter_parent=collision_filter_parent,
            **rkw,
        )
    elif d.prim_type == "PhysicsPrismaticJoint":
        dof = d.dofs[0]
        lo = None if dof.limit_lower == float("-inf") else dof.limit_lower
        hi = None if dof.limit_upper == float("inf") else dof.limit_upper
        pkw = _dof_kw(dof, False, lo, hi)
        joint = builder.add_joint_prismatic(
            parent=record.parent,
            child=record.child,
            parent_xform=source.parent_xform,
            child_xform=source.child_xform,
            axis=dof.axis,
            collision_filter_parent=collision_filter_parent,
            **pkw,
        )
    elif d.prim_type == "PhysicsFixedJoint":
        return builder.add_joint_fixed(
            parent=record.parent,
            child=record.child,
            parent_xform=source.parent_xform,
            child_xform=source.child_xform,
            collision_filter_parent=collision_filter_parent,
        )
    elif d.prim_type == "PhysicsSphericalJoint":
        return builder.add_joint_ball(
            parent=record.parent,
            child=record.child,
            parent_xform=source.parent_xform,
            child_xform=source.child_xform,
            collision_filter_parent=collision_filter_parent,
        )
    elif d.prim_type == "PhysicsDistanceJoint":
        return builder.add_joint_distance(
            parent=record.parent,
            child=record.child,
            parent_xform=source.parent_xform,
            child_xform=source.child_xform,
            min_distance=d.min_distance if d.min_distance >= 0.0 else -1.0,
            max_distance=d.max_distance if d.max_distance >= 0.0 else -1.0,
            collision_filter_parent=collision_filter_parent,
        )
    elif d.prim_type == "PhysicsJoint":
        linear = []
        angular = []
        for dof in d.dofs:
            kw = _dof_kw(dof, dof.rotational, dof.limit_lower, dof.limit_upper)
            config = _joint_dof_config(builder, dof.axis, kw)
            (angular if dof.rotational else linear).append((config, dof))
        joint = builder.add_joint_d6(
            parent=record.parent,
            child=record.child,
            parent_xform=source.parent_xform,
            child_xform=source.child_xform,
            linear_axes=[config for config, _ in linear] or None,
            angular_axes=[config for config, _ in angular] or None,
            collision_filter_parent=collision_filter_parent,
        )
        for offset, (_, dof) in enumerate(linear + angular):
            _apply_initial_joint_state(builder, joint, dof, offset)
        return joint
    else:
        raise UnsupportedPhysicsError(f"unsupported joint type {d.prim_type} at {source.path}")
    _apply_initial_joint_state(builder, joint, d.dofs[0])
    return joint


def _merge_same_body_pairs(records: List[_JointBuildRecord]) -> List[_JointBuildRecord]:
    """Collapse same-body-pair revolute/prismatic joints into one D6 record."""
    groups: Dict[Tuple[int, int], List[_JointBuildRecord]] = {}
    order: List[Tuple[int, int]] = []
    for record in records:
        key = (record.parent, record.child)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    out: List[_JointBuildRecord] = []
    for key in order:
        # Newton's USD importer uses deterministic joint-path order when several
        # MuJoCo axis prims share a body pair. The first path supplies the D6
        # frame, so preserve that choice as well as the resulting DOF order.
        group = sorted(groups[key], key=lambda record: record.sources[0].path)
        single_axis = all(
            record.sources[0].desc.prim_type in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
            for record in group
        )
        if len(group) > 1 and single_axis:
            sources = tuple(record.sources[0] for record in group)
            out.append(_JointBuildRecord(sources, *key))
        else:
            out.extend(group)
    return out


def _record_paths(record: _JointBuildRecord) -> List[str]:
    """Return every USD joint path represented by one Newton joint."""
    return [source.path for source in record.sources]


def _component_bodies(component: List[_JointBuildRecord]) -> List[int]:
    return sorted({body for record in component for body in (record.parent, record.child) if body != -1})


def _component_paths(component: List[_JointBuildRecord], bodies: _Bodies) -> set[str]:
    paths = {bodies.paths[body] for body in _component_bodies(component)}
    paths.update(record.sources[0].path for record in component)
    return paths


def _resolve_articulation_policies(
    components: List[List[_JointBuildRecord]],
    bodies: _Bodies,
    authored: Dict[str, bool],
    hierarchy: "_parse._Hierarchy",
) -> Tuple[Dict[int, bool], Dict[int, str]]:
    component_paths = [_component_paths(component, bodies) for component in components]
    resolved: Dict[int, bool] = {}
    owners: Dict[int, str] = {}
    for root, enabled in authored.items():
        exact = [i for i, paths in enumerate(component_paths) if root in paths]
        candidates = exact or [
            i for i, paths in enumerate(component_paths) if any(hierarchy.contains(root, path) for path in paths)
        ]
        if len(candidates) > 1:
            raise InvalidPhysicsError(f"articulation root {root} contains multiple disconnected joint components")
        if not candidates:
            continue
        component = candidates[0]
        if component in owners:
            raise InvalidPhysicsError(
                f"articulation roots {owners[component]} and {root} resolve to the same joint component"
            )
        owners[component] = root
        resolved[component] = enabled
    return resolved, owners


def _filter_articulation_self_collisions(builder: Any, body_indices: List[int]) -> None:
    for offset, body0 in enumerate(body_indices):
        for body1 in body_indices[offset + 1 :]:
            for shape0 in builder.body_shapes[body0]:
                for shape1 in builder.body_shapes[body1]:
                    builder.add_shape_collision_filter_pair(shape0, shape1)


def _build_joints(
    builder: Any,
    joint_descs: List[Tuple[str, "_JointDesc"]],
    bodies: _Bodies,
    hierarchy: "_parse._Hierarchy",
    articulation_self_collisions: Optional[Dict[str, bool]] = None,
) -> Tuple[bool, Dict[str, int]]:
    """Build rooted articulations and add all remaining joints as orphans."""
    records = _joint_records(joint_descs, bodies)
    included = [record for record in records if not record.sources[0].desc.excluded_from_articulation]
    excluded = [record for record in records if record.sources[0].desc.excluded_from_articulation]
    components = _joint_components(included)
    policies, articulation_roots = _resolve_articulation_policies(
        components,
        bodies,
        articulation_self_collisions or {},
        hierarchy,
    )
    # add_body already created single-body articulations. Preserve an authored
    # root only when it identifies one of them, not several independent bodies
    # or a root already assigned to an explicit joint component.
    standalone_roots: Dict[str, List[int]] = {}
    for articulation, start in enumerate(builder.articulation_start):
        body_path = builder.body_label[builder.joint_child[start]]
        root = hierarchy.nearest(body_path, articulation_self_collisions or {})
        if root is not None:
            standalone_roots.setdefault(root, []).append(articulation)
    for root, articulations in standalone_roots.items():
        if len(articulations) == 1 and root not in articulation_roots.values():
            builder.articulation_label[articulations[0]] = root
    has_orphans = bool(excluded)
    joint_by_path: Dict[str, int] = {}
    for component_index, component in enumerate(components):
        merged = _merge_same_body_pairs(component)
        is_articulation = component_index in policies
        joint_ids: List[int] = []
        build_records = _order_joints_root_first(_with_free_base(merged)) if is_articulation else merged
        for r in build_records:
            jid = _add_joint(builder, r)
            joint_ids.append(jid)
            jpaths = _record_paths(r)
            if jpaths and jpaths[0].startswith("/"):
                builder.joint_label[jid] = jpaths[0]
            joint_by_path.update((path, jid) for path in jpaths)
        if is_articulation and joint_ids:
            builder.add_articulation(joints=joint_ids, label=articulation_roots[component_index])
        else:
            has_orphans = has_orphans or bool(joint_ids)
        if policies.get(component_index) is False:
            _filter_articulation_self_collisions(builder, _component_bodies(component))
    for r in _merge_same_body_pairs(excluded):
        jid = _add_joint(builder, r)
        jpaths = _record_paths(r)
        if jpaths and jpaths[0].startswith("/"):
            builder.joint_label[jid] = jpaths[0]
        joint_by_path.update((path, jid) for path in jpaths)
    return has_orphans, joint_by_path


def _build_mimics(builder: Any, mimics: List[Dict[str, Any]], joint_by_path: Dict[str, int]) -> None:
    source_counts = Counter(joint_by_path.values())
    for mimic in mimics:
        follower = joint_by_path.get(mimic["path"])
        leader = joint_by_path.get(mimic["leader"])
        if follower is None:
            raise InvalidPhysicsError(f"NewtonMimicAPI at {mimic['path']} is not applied to an imported joint")
        if leader is None:
            raise InvalidPhysicsError(
                f"NewtonMimicAPI at {mimic['path']} references unknown joint {mimic['leader']}"
            )
        # Newton mimics address whole joints, not individual source axes of a
        # merged D6. Applying one here would constrain unrelated coordinates.
        for role, path, joint in (("follower", mimic["path"], follower), ("leader", mimic["leader"], leader)):
            if source_counts[joint] > 1:
                raise UnsupportedPhysicsError(
                    f"NewtonMimicAPI at {mimic['path']}: {role} joint {path} is merged into a D6 joint; "
                    "Newton's mimic APIs cannot target individual source axes after merging"
                )
        coef0 = mimic["coef0"]
        if mimic["rotational"]:
            coef0 = math.radians(coef0)
        set_joint_mimic = getattr(builder, "set_joint_mimic", None)
        if set_joint_mimic is not None:
            # Newer solvers consume per-joint mimics, not legacy constraints.
            try:
                set_joint_mimic(follower, leader, coeffs=(coef0, mimic["coef1"]))
            except ValueError as exc:
                raise InvalidPhysicsError(f"NewtonMimicAPI at {mimic['path']}: {exc}") from exc
        else:
            builder.add_constraint_mimic(
                joint0=follower,
                joint1=leader,
                coef0=coef0,
                coef1=mimic["coef1"],
                label=mimic["path"],
            )


_R3 = [0, 1, 2, 4, 5, 6, 8, 9, 10]  # flat row-major indices of the upper-left 3x3


def _matrix_scale_rotation(linear: np.ndarray, path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(linear)):
        raise InvalidPhysicsError(f"non-finite world transform at {path}")
    scale = np.linalg.norm(linear, axis=1)
    if np.any(scale <= 1.0e-12):
        return scale, np.eye(3)
    rotation = linear / scale[:, None]
    if np.linalg.det(rotation) < 0.0:
        scale *= -1.0
        rotation *= -1.0
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6, rtol=1.0e-6):
        raise UnsupportedPhysicsError(f"sheared transform hierarchy is unsupported at {path}")
    return scale, rotation


def _validate_transform_hierarchies(
    stage,
    pd,
    ordinal: int,
    hierarchy: "_parse._Hierarchy",
    world_matrices: Dict[str, np.ndarray],
) -> None:
    """Reject scale/rotation hierarchies whose authored decomposition is lost."""
    targets = tuple(world_matrices)
    ancestor_paths = set()
    for target in targets:
        ancestor_paths.update(path for path in hierarchy.ancestors(target) if path)
    missing = sorted(ancestor_paths.difference(world_matrices))
    if missing:
        with _stage._path_list_query(stage, pd, missing) as query:
            rows = _stage.read_columns(stage, pd, query, [WORLD_MATRIX], ordinal)[WORLD_MATRIX]
        world_matrices = {
            **world_matrices,
            **{path: rows[i] for i, path in enumerate(missing) if i in rows},
        }

    for target in targets:
        chain = [node for node in reversed(hierarchy.ancestors(target, include_self=True)) if node in world_matrices]

        parent_linear = np.eye(3)
        parent_scale = np.ones(3)
        for node in chain:
            world = np.asarray(world_matrices[node], dtype=np.float64).reshape(16)[_R3].reshape(3, 3)
            try:
                local = world @ np.linalg.inv(parent_linear)
            except np.linalg.LinAlgError:
                break
            local_scale, local_rotation = _matrix_scale_rotation(local, target)
            scale_matrix = np.diag(parent_scale)
            if not np.allclose(
                local_rotation @ scale_matrix,
                scale_matrix @ local_rotation,
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise UnsupportedPhysicsError(
                    f"non-uniform scale across a rotated hierarchy is unsupported at {target}"
                )
            parent_scale *= local_scale
            parent_linear = world


def _decode_pose(mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode ovstage ``omni:fabric:worldMatrix`` rows (flat ``(N,16)``) into
    ``(translation (N,3), quaternion xyzw (N,4), scale (N,3))``.

    The matrix is a row-major, row-vector ``GfMatrix4d`` whose linear part is
    ``S·R`` (3x3 row ``i`` is ``scale_i · R_row_i``): translation is row 3, scale
    is the 3x3 **row** norms, and ``R`` is the row-normalized 3x3. Reflections
    use three negative scale components and the corresponding proper rotation,
    matching USD Physics' decomposition. The
    column-vector rotation Newton wants is ``Rᵀ``; the quaternion comes from it via
    a branch-selected Shepperd's method (vectorized, one branch per row)."""
    m = np.asarray(mats, dtype=np.float64)
    trans = np.ascontiguousarray(m[:, 12:15])
    r = m[:, _R3].reshape(-1, 3, 3)  # row-vector S·R
    scale = np.linalg.norm(r, axis=2)
    rot = r / np.where(scale > 1e-12, scale, 1.0)[:, :, None]  # row-vector R
    reflected = np.linalg.det(rot) < 0.0
    scale[reflected] *= -1.0
    rot[reflected] *= -1.0
    # Column-vector rotation Rᵀ: m_ij = rot[:, j, i].
    m00, m10, m20 = rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2]
    m01, m11, m21 = rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2]
    m02, m12, m22 = rot[:, 2, 0], rot[:, 2, 1], rot[:, 2, 2]
    tr = m00 + m11 + m22
    c0 = tr > 0.0
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    c2 = (~c0) & (~c1) & (m11 > m22)
    c3 = ~(c0 | c1 | c2)

    def _s(expr):
        return np.sqrt(np.maximum(expr, 1e-12)) * 2.0

    s0, s1, s2, s3 = _s(tr + 1.0), _s(1.0 + m00 - m11 - m22), _s(1.0 + m11 - m00 - m22), _s(1.0 + m22 - m00 - m11)
    conds = [c0, c1, c2, c3]
    qx = np.select(conds, [(m21 - m12) / s0, 0.25 * s1, (m01 + m10) / s2, (m02 + m20) / s3])
    qy = np.select(conds, [(m02 - m20) / s0, (m01 + m10) / s1, 0.25 * s2, (m12 + m21) / s3])
    qz = np.select(conds, [(m10 - m01) / s0, (m02 + m20) / s1, (m12 + m21) / s2, 0.25 * s3])
    qw = np.select(conds, [0.25 * s0, (m21 - m12) / s1, (m02 - m20) / s2, (m10 - m01) / s3])
    quat = np.stack([qx, qy, qz, qw], axis=1)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return trans, quat / norm, scale


def _scalar(rows: Dict[int, np.ndarray], i: int) -> Optional[float]:
    v = rows.get(i)
    return float(v[0]) if v is not None and len(v) else None


def _triple(rows: Dict[int, np.ndarray], i: int) -> Optional[Tuple[float, float, float]]:
    v = rows.get(i)
    return tuple(float(x) for x in v[:3]) if v is not None and len(v) >= 3 else None


def _quadruple(rows: Dict[int, np.ndarray], i: int) -> Optional[Tuple[float, float, float, float]]:
    v = rows.get(i)
    return tuple(float(x) for x in v[:4]) if v is not None and len(v) >= 4 else None


def _apply_initial_body_velocities(
    builder: Any,
    quats: np.ndarray,
    linear_rows: Dict[int, np.ndarray],
    angular_rows: Dict[int, np.ndarray],
) -> None:
    """Apply authored local body velocities as Newton world-frame CoM twists."""
    for body in range(len(builder.body_qd)):
        linear = np.asarray(_triple(linear_rows, body) or (0.0, 0.0, 0.0), dtype=np.float64)
        angular_degrees = np.asarray(_triple(angular_rows, body) or (0.0, 0.0, 0.0), dtype=np.float64)
        q = wp.quat(*(float(value) for value in quats[body]))
        linear_world = wp.quat_rotate(q, wp.vec3(*(float(value) for value in linear)))
        omega_world = wp.quat_rotate(
            q,
            wp.vec3(*(float(value * _RAD_PER_DEG) for value in angular_degrees)),
        )
        builder.body_qd[body] = wp.spatial_vector(
            float(linear_world[0]),
            float(linear_world[1]),
            float(linear_world[2]),
            float(omega_world[0]),
            float(omega_world[1]),
            float(omega_world[2]),
        )


def _initialize_free_joint_velocities(builder: Any) -> None:
    """Keep world-rooted FREE coordinates consistent with imported body twists."""
    for joint, joint_type in enumerate(builder.joint_type):
        if joint_type != newton.JointType.FREE:
            continue
        child = builder.joint_child[joint]
        qd = builder.body_qd[child]
        start = builder.joint_qd_start[joint]
        builder.joint_qd[start : start + 6] = [float(qd[i]) for i in range(6)]


def _require_empty_builder(builder: Any) -> None:
    """Reject model data that the current zero-based ovstage importer cannot append to."""
    count_names = (
        "world_count",
        "body_count",
        "shape_count",
        "joint_count",
        "articulation_count",
        "particle_count",
        "tri_count",
        "tet_count",
        "edge_count",
        "spring_count",
        "muscle_count",
    )
    populated = {
        name: int(getattr(builder, name, 0))
        for name in count_names
        if int(getattr(builder, name, 0)) != 0
    }
    collision_filter_pair_count = len(builder.shape_collision_filter_pairs)
    if collision_filter_pair_count:
        populated["shape_collision_filter_pairs"] = collision_filter_pair_count
    actuator_entry_count = len(builder.actuator_entries)
    if actuator_entry_count:
        populated["actuator_entries"] = actuator_entry_count
    for frequency, count in builder._custom_frequency_counts.items():
        if count:
            populated[f"custom_frequency:{frequency}"] = count
    for name, attribute in builder.custom_attributes.items():
        value_count = len(attribute.values) if attribute.values is not None else 0
        if value_count:
            populated[f"custom_attribute:{name}"] = value_count
    if populated:
        details = ", ".join(f"{name}={count}" for name, count in populated.items())
        raise ValueError(
            f"builder must not contain model entities, relations, or custom values before add_ovstage(): {details}"
        )


def _apply_collision_groups(
    builder: Any,
    groups: List[Dict[str, Any]],
    shape_by_path: Dict[str, int],
    hierarchy: _parse._Hierarchy,
) -> None:
    """Lower USD group rules to pairs without changing Newton's numeric groups.

    USD groups collide by default, unlike distinct positive Newton group IDs.
    Named merges share rules; any membership may veto a collision. Collection
    membership still supports only direct and subtree includes/excludes (OV-7).
    """
    if not groups:
        return
    merged_ids: Dict[Tuple[str, str], int] = {}
    group_ids: Dict[str, int] = {}
    for group in groups:
        key = ("merge", group["merge_group"]) if group["merge_group"] else ("path", group["path"])
        group_ids[group["path"]] = merged_ids.setdefault(key, len(merged_ids))

    # -1 denotes colliders outside all USD groups, not a Newton collision ID.
    blocked = {index: set() for index in [-1, *merged_ids.values()]}
    for group in groups:
        source = group_ids[group["path"]]
        targets = set()
        for path in group["filtered_groups"]:
            if path not in group_ids:
                raise InvalidPhysicsError(f"collision group {group['path']} filters unknown group {path}")
            targets.add(group_ids[path])
        if group["inverted"]:
            targets = blocked.keys() - targets
        for target in targets:
            blocked[source].add(target)
            blocked[target].add(source)
    if not any(blocked.values()):
        return

    def in_collection(path: str, roots: List[str]) -> bool:
        return any(hierarchy.contains(root, path) for root in roots)

    # Resolve rules per membership combination, expanding to shape pairs only
    # when blocked. This avoids a shape-by-shape pass over allowed collisions.
    classes: Dict[Tuple[int, ...], List[int]] = {}
    for path, shape in shape_by_path.items():
        if not builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES:
            continue
        membership = tuple(sorted({
            group_ids[group["path"]]
            for group in groups
            if in_collection(path, group["includes"]) and not in_collection(path, group["excludes"])
        })) or (-1,)
        classes.setdefault(membership, []).append(shape)

    for (members_a, shapes_a), (members_b, shapes_b) in combinations_with_replacement(classes.items(), 2):
        if not any(b in blocked[a] for a in members_a for b in members_b):
            continue
        pairs = combinations(shapes_a, 2) if members_a == members_b else product(shapes_a, shapes_b)
        for shape_a, shape_b in pairs:
            builder.add_shape_collision_filter_pair(shape_a, shape_b)


def add_ovstage(builder: Any, stage: Any, pd: Any, ordinal: int = 1) -> bool:
    """Populate ``builder`` from ovstage and report whether it has orphan joints."""
    _require_empty_builder(builder)
    meters_per_unit, up_axis = _parse.read_stage_units(stage, pd, ordinal)
    hierarchy = _parse.read_hierarchy(stage, pd, ordinal)
    _parse.validate_unsupported_point_instancers(stage, pd, ordinal, hierarchy=hierarchy)
    gravity = _parse.scene_gravity_vector(
        stage,
        pd,
        ordinal,
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
    )
    builder.up_axis = getattr(newton.Axis, up_axis)
    if gravity is None:
        gravity = tuple(-9.81 * component for component in builder.up_axis.to_vector())
    builder.gravity = wp.vec3(*gravity)
    _, body_paths = _parse.read_bodies(stage, pd, ordinal, hierarchy=hierarchy)
    materials = _parse.read_physics_materials(stage, pd, ordinal)
    n = len(body_paths)

    # Body world poses + PhysicsMassAPI values, read in known order via a path list.
    wm: Dict[int, np.ndarray] = {}
    mass: Dict[int, np.ndarray] = {}
    diag: Dict[int, np.ndarray] = {}
    newton_inertia: Dict[int, np.ndarray] = {}
    com: Dict[int, np.ndarray] = {}
    density: Dict[int, np.ndarray] = {}
    principal: Dict[int, np.ndarray] = {}
    kinematic: Dict[int, np.ndarray] = {}
    enabled: Dict[int, np.ndarray] = {}
    initial_velocity: Dict[int, np.ndarray] = {}
    initial_angular_velocity: Dict[int, np.ndarray] = {}
    starts_asleep: Dict[int, np.ndarray] = {}
    if n:
        with _stage._path_list_query(stage, pd, body_paths) as qb:
            columns = _stage.read_columns(
                stage,
                pd,
                qb,
                (
                    WORLD_MATRIX,
                    "physics:mass",
                    "physics:diagonalInertia",
                    "physics:centerOfMass",
                    "physics:density",
                    "physics:principalAxes",
                    "physics:kinematicEnabled",
                    RIGID_BODY_ENABLED,
                    BODY_VELOCITY,
                    BODY_ANGULAR_VELOCITY,
                    "physics:startsAsleep",
                    NEWTON_INERTIA,
                ),
                ordinal,
                ragged=(NEWTON_INERTIA,),
            )
        wm = columns[WORLD_MATRIX]
        mass = columns["physics:mass"]
        diag = columns["physics:diagonalInertia"]
        com = columns["physics:centerOfMass"]
        density = columns["physics:density"]
        principal = columns["physics:principalAxes"]
        kinematic = columns["physics:kinematicEnabled"]
        enabled = columns[RIGID_BODY_ENABLED]
        initial_velocity = columns[BODY_VELOCITY]
        initial_angular_velocity = columns[BODY_ANGULAR_VELOCITY]
        starts_asleep = columns["physics:startsAsleep"]
        newton_inertia = columns[NEWTON_INERTIA]

        missing_transforms = [body_paths[i] for i in range(n) if i not in wm]
        if missing_transforms:
            raise OvstageContractError(f"rigid body has no world transform: {missing_transforms[0]}")
        for i, path in enumerate(body_paths):
            e = enabled.get(i)
            if e is not None and len(e) and not bool(e[0]):
                log_diagnostic(
                    "disabled-rigid-body-imported",
                    "physics:rigidBodyEnabled=false is not represented by Newton; the body was imported, "
                    "matching add_usd's default enabled-body policy",
                    path=path,
                )
            asleep = starts_asleep.get(i)
            if asleep is not None and len(asleep) and bool(asleep[0]):
                log_diagnostic(
                    "starts-asleep-ignored",
                    "physics:startsAsleep=true has no portable Newton model representation; the body starts awake",
                    path=path,
                )

    mats = np.stack([np.asarray(wm[i], dtype=np.float64).reshape(16) for i in range(n)]) if n else np.empty((0, 16))
    _validate_transform_hierarchies(
        stage,
        pd,
        ordinal,
        hierarchy,
        {path: mats[i] for i, path in enumerate(body_paths)},
    )
    trans, quats, scales = _decode_pose(mats)

    # Mass properties are active only when their owning API is applied.
    mass_api = _parse._mass_api_paths(stage, pd, ordinal)
    newton_mass_api = _parse._newton_mass_api_paths(stage, pd, ordinal)

    # Assemble the _Bodies struct the joint pipeline + mass finalize consume.
    world = {body_paths[i]: (trans[i], quats[i]) for i in range(n)}
    bodies = _Bodies(
        paths=list(body_paths),
        index={p: i for i, p in enumerate(body_paths)},
        world=world,
        authored={
            body_paths[i]: (
                {
                    "density": _scalar(density, i),
                    "mass": _scalar(mass, i),
                    "inertia_tensor": (
                        _newton_inertia_tensor(newton_inertia.get(i), body_paths[i])
                        if body_paths[i] in newton_mass_api
                        else None
                    ),
                    "inertia": _triple(diag, i),
                    "principal_axes": _quadruple(principal, i),
                    "com": _triple(com, i),
                }
                if body_paths[i] in mass_api or body_paths[i] in newton_mass_api
                else {
                    "density": None,
                    "mass": None,
                    "inertia_tensor": None,
                    "inertia": None,
                    "principal_axes": None,
                    "com": None,
                }
            )
            for i in range(n)
        },
        scale={body_paths[i]: scales[i] for i in range(n)},
        kinematic={
            body_paths[i]: bool(kinematic[i][0]) if i in kinematic and len(kinematic[i]) else False for i in range(n)
        },
    )

    joint_descs = _parse.read_joints(stage, pd, ordinal, hierarchy=hierarchy)

    # Bodies a joint references use add_link (the joint is their sole connection);
    # the rest are free bodies (implicit free joint).
    art_paths = _prescan_articulation_bodies(joint_descs)
    for i, path in enumerate(body_paths):
        xform = _wp_xform(trans[i], quats[i])
        # A standalone kinematic body has maximal state but no implicit FREE
        # joint/articulation in add_usd; explicit joints still own articulated bodies.
        if path in art_paths or bodies.kinematic[path]:
            builder.add_link(xform=xform, label=path, is_kinematic=bodies.kinematic[path])
        else:
            builder.add_body(xform=xform, label=path, is_kinematic=bodies.kinematic[path])

    # Colliders, assigned by nearest rigid-body ancestor. Orphans are static
    # world geometry; in particular, a USD Plane becomes Newton's ground plane.
    def _nearest_body(shape_path: Optional[str]) -> Optional[str]:
        return hierarchy.nearest(shape_path, bodies.index) if shape_path is not None else None

    sites = _parse.read_sites(stage, pd, ordinal)
    _validate_transform_hierarchies(
        stage,
        pd,
        ordinal,
        hierarchy,
        {
            site["path"]: np.asarray(site["world_matrix"], dtype=np.float64).reshape(16)
            for site in sites
            if site["world_matrix"] is not None
        },
    )
    for site in sites:
        path = site["path"]
        body_path = _nearest_body(path)
        if body_path is None:
            raise UnsupportedPhysicsError(f"NewtonSiteAPI at {path} is not beneath a rigid body")
        world_matrix = site["world_matrix"]
        if world_matrix is None:
            raise OvstageContractError(f"site has no world transform: {path}")
        site_matrix = np.asarray(world_matrix, dtype=np.float64).reshape(1, 16)
        translation, rotation, scale_array = _decode_pose(site_matrix)
        scale = tuple(float(value) for value in scale_array[0])
        xform = _relative_xform(world[body_path], (translation[0], rotation[0]))
        _add_analytic_shape(
            builder,
            site["type"],
            body=bodies.index[body_path],
            xform=xform,
            scale=scale,
            axis=site.get("axis"),
            label=path,
            as_site=True,
            **_analytic_geometry_kwargs(site),
        )

    remeshing: Dict[str, List[int]] = {}
    shape_by_path: Dict[str, int] = {}
    disabled_shapes: List[int] = []
    colliders = _parse.read_colliders(
        stage,
        pd,
        ordinal,
        newton_mass_api=newton_mass_api,
    )
    _validate_transform_hierarchies(
        stage,
        pd,
        ordinal,
        hierarchy,
        {
            collider["path"]: np.asarray(collider["world_matrix"], dtype=np.float64).reshape(16)
            for collider in colliders
            if collider["world_matrix"] is not None
        },
    )
    for c in colliders:
        path = c["path"]
        shape_options, restore_margin = _shape_options(builder, c)
        cm = c["world_matrix"]
        if cm is None:
            raise OvstageContractError(f"collider has no world transform: {path}")
        collider_matrix = np.asarray(cm, dtype=np.float64).reshape(1, 16)
        ct, cq, cscale_arr = _decode_pose(collider_matrix)
        cscale = tuple(float(x) for x in cscale_arr[0])
        body_path = _nearest_body(path)
        static = body_path is None
        material_path = c.get("material")
        material = materials.get(material_path) if material_path else None
        if material_path and material is None:
            raise InvalidPhysicsError(f"collider {path} binds unknown physics material {material_path}")
        if c["type"] == "Plane":
            if not static:
                raise UnsupportedPhysicsError(f"body-attached Plane collider is unsupported at {path}")
            axis = c.get("axis") or "Z"
            quat = np.asarray(cq[0], dtype=float)
            if axis != "Z" or not np.allclose(np.abs(quat), (0.0, 0.0, 0.0, 1.0), atol=1e-6):
                raise UnsupportedPhysicsError(f"only an unrotated Z-axis Plane is supported at {path}")
            shape = builder.add_ground_plane(
                height=float(ct[0][2]),
                cfg=_shape_cfg(
                    builder,
                    None,
                    material,
                    collision_enabled=c["collision_enabled"],
                    overrides=shape_options,
                ),
                label=path,
            )
            if restore_margin is not None:
                builder.shape_margin[shape] = restore_margin
            shape_by_path[path] = shape
            if not c["collision_enabled"]:
                disabled_shapes.append(shape)
            continue
        body = -1 if static else bodies.index[body_path]
        xform = _wp_xform(ct[0], cq[0]) if static else _relative_xform(world[body_path], (ct[0], cq[0]))
        # Shape density: bound material (priority, matches add_usd), else the
        # body's authored physics:density, else the builder default (None).
        body_density = None if static else bodies.authored[body_path]["density"]
        # A bound material owns the shape density; non-positive material density
        # falls back to the ModelBuilder default, matching add_usd.
        material_density = material.get("density") if material is not None else None
        shape_density = (
            (
                material_density
                if material_density is not None and material_density > 0.0
                else builder.default_shape_cfg.density
            )
            if material is not None
            else body_density
        )
        if c["type"] == "Mesh":
            mesh = _mesh_geometry(c)
            shape = builder.add_shape_mesh(
                body=body,
                xform=xform,
                mesh=mesh,
                scale=wp.vec3(*cscale),
                cfg=_shape_cfg(
                    builder,
                    shape_density,
                    material,
                    collision_enabled=c["collision_enabled"],
                    overrides=_mesh_shape_options(shape_options),
                ),
                label=path,
            )
            if c["sdf_api"]:
                _apply_deferred_mesh_sdf(builder, shape, shape_options)
            method = _MESH_APPROXIMATION_METHOD.get(c.get("approximation"))
            if method is not None:
                if c["sdf_api"]:
                    log_diagnostic(
                        "sdf-mesh-approximation-ignored",
                        f"physics:approximation={c['approximation']!r} is ignored by NewtonSDFCollisionAPI",
                        path=path,
                    )
                else:
                    remeshing.setdefault(method, []).append(shape)
        else:
            shape = _add_analytic_shape(
                builder,
                c["type"],
                body=body,
                xform=xform,
                scale=cscale,
                density=shape_density,
                material=material,
                collision_enabled=c["collision_enabled"],
                cfg_overrides=shape_options,
                axis=c.get("axis"),
                label=path,
                **_analytic_geometry_kwargs(c),
            )
        if restore_margin is not None:
            builder.shape_margin[shape] = restore_margin
        shape_by_path[path] = shape
        if not c["collision_enabled"]:
            disabled_shapes.append(shape)

    # Newton propagates existing filters to additional convex parts during remeshing.
    _apply_collision_groups(builder, _parse.read_collision_groups(stage, pd, ordinal), shape_by_path, hierarchy)

    for method, shapes in remeshing.items():
        builder.approximate_meshes(method=method, shape_indices=shapes)

    def _shapes_at(path: str) -> List[int]:
        exact = shape_by_path.get(path)
        if exact is not None:
            return [exact]
        return [shape for collider_path, shape in shape_by_path.items() if hierarchy.contains(path, collider_path)]

    for source, target in _parse.read_filtered_pairs(stage, pd, ordinal):
        source_shapes = _shapes_at(source)
        target_shapes = _shapes_at(target)
        if not source_shapes or not target_shapes:
            raise InvalidPhysicsError(f"filtered pair {source} -> {target} does not resolve to collision shapes")
        for shape0 in source_shapes:
            for shape1 in target_shapes:
                if shape0 != shape1:
                    builder.add_shape_collision_filter_pair(shape0, shape1)

    for shape0 in disabled_shapes:
        for shape1 in range(builder.shape_count):
            if shape0 != shape1:
                builder.add_shape_collision_filter_pair(shape0, shape1)

    _finalize_masses(builder, bodies)
    _apply_initial_body_velocities(builder, quats, initial_velocity, initial_angular_velocity)
    has_orphan_joints, joint_by_path = _build_joints(
        builder,
        joint_descs,
        bodies,
        hierarchy,
        articulation_self_collisions=_parse.read_articulation_self_collisions(stage, pd, ordinal),
    )
    _build_mimics(builder, _parse.read_mimics(stage, pd, ordinal), joint_by_path)
    _initialize_free_joint_velocities(builder)
    from . import _deformables  # noqa: PLC0415

    deformable_values = _deformables.read_deformables(stage, pd, ordinal)
    _deformables.build_deformables(builder, deformable_values)
    return has_orphan_joints


def build_model(stage: Any, pd: Any, ordinal: int = 1):
    """Build and finalize a ``newton.Model`` from a populated ovstage."""
    builder = newton.ModelBuilder()
    has_orphan_joints = add_ovstage(builder, stage, pd, ordinal=ordinal)
    if builder.particle_count:
        builder.color(include_bending=True)
    return builder.finalize(skip_validation_joints=has_orphan_joints)
