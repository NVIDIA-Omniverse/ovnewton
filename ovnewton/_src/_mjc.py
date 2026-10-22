# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and lower MuJoCo joints and equalities without importing USD or MuJoCo."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from . import _parse, _stage
from ._errors import InvalidPhysicsError, OvstageContractError, UnsupportedPhysicsError, log_diagnostic
from ._schema_names import USD_API_SCHEMAS, USD_PRIM_TYPE
from ._schemas import _MJC_EQUALITY_PROPERTIES, _MJC_JOINT_CUSTOM_ATTRIBUTES

_CONNECT = "MjcEqualityConnectAPI"
_WELD = "MjcEqualityWeldAPI"
_JOINT = "MjcEqualityJointAPI"


def joint_custom_attributes(builder: Any, properties: dict[str, Any], path: str) -> dict[str, Any]:
    """Preserve MuJoCo joint values using the caller's registered attributes."""
    attributes = {}
    for property_name, (name, size) in _MJC_JOINT_CUSTOM_ATTRIBUTES.items():
        if property_name not in properties:
            continue
        key = f"mujoco:{name}"
        definition = builder.custom_attributes.get(key)
        if definition is None:
            raise UnsupportedPhysicsError(
                f"{property_name} at {path} requires "
                "SolverMuJoCo.register_custom_attributes(builder) before add_ovstage"
            )
        value = properties[property_name]
        attributes[key] = definition.dtype(*value) if size > 1 else value
    return attributes


@dataclass(frozen=True)
class _BodyEquality:
    path: str
    joint: _parse._JointDesc
    weld: bool
    torque_scale: float
    solref: tuple[float, ...]
    solimp: tuple[float, ...]


def read_equalities(
    stage: Any, pd: Any, ordinal: int, hierarchy: _parse._Hierarchy
) -> tuple[dict[str, _BodyEquality], dict[str, _JointEquality]]:
    """Classify composed equality schemas once, then read their distinct payloads."""
    body_candidates = {}
    joint_paths = []
    for path, schemas in _parse.read_applied_schemas(stage, pd, ordinal, tuple(hierarchy.parents)).items():
        if {_CONNECT, _WELD} <= schemas:
            raise InvalidPhysicsError(f"conflicting Connect and Weld equality schemas at {path}")
        if _JOINT in schemas:
            if schemas.intersection((_CONNECT, _WELD)):
                raise InvalidPhysicsError(f"conflicting body and joint equality schemas at {path}")
            joint_paths.append(path)
        elif schemas.intersection((_CONNECT, _WELD)):
            body_candidates[path] = _WELD in schemas
    return (
        _read_body_equalities(stage, pd, ordinal, body_candidates),
        _read_joint_equalities(stage, pd, ordinal, joint_paths),
    )


def _read_body_equalities(stage: Any, pd: Any, ordinal: int, candidates: dict[str, bool]) -> dict[str, _BodyEquality]:
    """Read body equalities separately from articulation-tree joints."""
    paths = list(candidates)
    if not paths:
        return {}
    attributes = (USD_API_SCHEMAS, "mjc:solref", "mjc:solimp", "mjc:torqueScale")
    with _stage._path_list_query(stage, pd, paths) as query:
        values = _stage.read_columns(stage, pd, query, attributes, ordinal, ragged=attributes)
    for i, path in enumerate(paths):
        if i not in values[USD_API_SCHEMAS]:
            raise OvstageContractError(f"equality at {path} requires population with ovnewton.population_desc()")
    types = _parse._paths_by_prim_type(stage, pd, paths, ordinal)
    joints = []
    for jtype, typed_paths in types.items():
        for path in typed_paths:
            expected = "PhysicsFixedJoint" if candidates[path] else "PhysicsSphericalJoint"
            if jtype != expected:
                raise UnsupportedPhysicsError(f"equality at {path} requires {expected}, got {jtype}")
        joints.extend(_parse._read_joints_of_type(stage, pd, jtype, typed_paths, ordinal, ()))
    indices = {path: i for i, path in enumerate(paths)}
    equalities = {}
    for path, joint in joints:
        for endpoint, (position, rotation) in enumerate(
            ((joint.local_pos0, joint.local_rot0), (joint.local_pos1, joint.local_rot1))
        ):
            if not np.all(np.isfinite(position)):
                raise InvalidPhysicsError(f"invalid physics:localPos{endpoint} at {path}; expected finite values")
            if not np.all(np.isfinite(rotation)) or not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-6):
                raise InvalidPhysicsError(f"invalid physics:localRot{endpoint} at {path}; expected a unit quaternion")
        i = indices[path]
        solref = _value(values["mjc:solref"].get(i), (0.02, 1.0), path, "mjc:solref")
        solimp = _value(values["mjc:solimp"].get(i), (0.9, 0.95, 0.001, 0.5, 2.0), path, "mjc:solimp")
        torque_scale = _value(values["mjc:torqueScale"].get(i), (1.0,), path, "mjc:torqueScale")[0]
        if torque_scale < 0:
            raise InvalidPhysicsError(f"negative mjc:torqueScale at {path}")
        equalities[path] = _BodyEquality(path, joint, candidates[path], torque_scale, solref, solimp)
    return equalities


def _value(row: Any, default: tuple[float, ...], path: str, name: str) -> tuple[float, ...]:
    if row is None:
        return default
    if len(row) != len(default) or not np.all(np.isfinite(row)):
        raise InvalidPhysicsError(f"invalid {name} at {path}; expected {len(default)} finite values")
    return tuple(float(v) for v in row)


@dataclass(frozen=True)
class _JointEquality:
    path: str
    leader: str
    polycoef: tuple[float, ...]
    enabled: bool
    solref: tuple[float, ...]
    solimp: tuple[float, ...]


def _read_joint_equalities(stage: Any, pd: Any, ordinal: int, paths: list[str]) -> dict[str, _JointEquality]:
    """Read scalar joint couplings, leaving their joints in the articulation tree."""
    if not paths:
        return {}
    attributes = (USD_API_SCHEMAS, USD_PRIM_TYPE, "physics:jointEnabled", *_MJC_EQUALITY_PROPERTIES)
    with _stage._path_list_query(stage, pd, paths) as query:
        values = _stage.read_columns(
            stage,
            pd,
            query,
            attributes,
            ordinal,
            ragged=(USD_API_SCHEMAS, "mjc:solref", "mjc:solimp", "newton:mimicJoint", "mjc:target"),
        )
    equalities = {}
    for i, path in enumerate(paths):

        def value(name, default):
            return _value(values[name].get(i), default, path, name)

        def coefficient(index, default):
            modern, legacy = f"newton:mimicCoef{index}", f"mjc:coef{index}"
            if i not in values[modern]:
                return value(legacy, (default,))[0], False
            resolved = value(modern, (default,))[0]
            legacy_row = values[legacy].get(i)
            # Population does not distinguish an authored default from a schema
            # fallback. Do not guess which coefficient Newton would consume.
            if resolved == default and legacy_row is not None and not np.array_equal(legacy_row, (default,)):
                raise UnsupportedPhysicsError(
                    f"ambiguous {modern} and {legacy} at {path}; "
                    "migrate the legacy coefficient to the newton:mimicCoef property and remove the legacy alias"
                )
            return resolved, True

        if not value("physics:jointEnabled", (1.0,))[0]:
            continue
        if i not in values[USD_API_SCHEMAS]:
            raise OvstageContractError(f"equality at {path} requires population with ovnewton.population_desc()")
        prim_type = _parse._token_value(pd, values[USD_PRIM_TYPE].get(i))
        if prim_type is None:
            raise OvstageContractError(f"prim has no readable {USD_PRIM_TYPE!r} value: {path}")
        if prim_type not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            raise UnsupportedPhysicsError(f"joint equality at {path} requires a revolute or prismatic joint")
        targets = values["newton:mimicJoint"].get(i)
        if targets is None or not len(targets):
            targets = values["mjc:target"].get(i)
        if targets is None or not len(targets):
            raise InvalidPhysicsError(f"joint equality at {path} has no newton:mimicJoint target")
        if len(targets) != 1:
            raise InvalidPhysicsError(f"joint equality at {path} has multiple newton:mimicJoint targets")
        leader = pd.path_to_string(int(targets[0]))
        if leader == path:
            raise InvalidPhysicsError(f"joint equality at {path} cannot target itself")
        coef0, modern_offset = coefficient(0, 0.0)
        if modern_offset and prim_type == "PhysicsRevoluteJoint":
            coef0 = math.radians(coef0)
        coef1, _ = coefficient(1, 1.0)
        polycoef = (coef0, coef1, *(value(f"mjc:coef{n}", (0.0,))[0] for n in (2, 3, 4)))
        equalities[path] = _JointEquality(
            path,
            leader,
            polycoef,
            bool(value("newton:mimicEnabled", (1.0,))[0]),
            value("mjc:solref", (0.02, 1.0)),
            value("mjc:solimp", (0.9, 0.95, 0.001, 0.5, 2.0)),
        )
    return equalities


def build_joint_equalities(builder: Any, equalities: dict[str, _JointEquality], joints: dict[str, int]) -> None:
    """Project the linear mimic while preserving complete polynomial MuJoCo rows."""
    source_counts = Counter(joints.values())
    for equality in equalities.values():
        indices = []
        for path in (equality.path, equality.leader):
            joint = joints.get(path)
            if joint is None:
                raise InvalidPhysicsError(f"joint equality at {equality.path} references unknown joint {path}")
            if source_counts[joint] > 1:
                raise UnsupportedPhysicsError(
                    f"joint equality at {equality.path} targets {path} merged into a D6 joint"
                )
            if builder.joint_type[joint] not in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                raise UnsupportedPhysicsError(f"joint equality at {equality.path} requires scalar joint target {path}")
            indices.append(joint)
        if any(equality.polycoef[2:]):
            log_diagnostic(
                "joint-equality-linear-projection",
                "Higher-order coefficients are preserved for SolverMuJoCo; "
                "the generic Newton mimic uses only coef0/coef1.",
                path=equality.path,
            )
        # Newton 1.7's equality exporter still uses a legacy mimic index to suppress its
        # linear projection in favor of the authoritative polynomial row.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"ModelBuilder\.add_constraint_mimic\(\) is deprecated", category=DeprecationWarning
            )
            target = builder.add_constraint_mimic(
                joint0=indices[0],
                joint1=indices[1],
                coef0=equality.polycoef[0],
                coef1=equality.polycoef[1],
                enabled=equality.enabled,
                label=equality.path,
            )
        builder.add_custom_values(
            **{
                "mujoco:equality_constraint_type": 2,  # JOINT
                "mujoco:equality_constraint_objtype": 3,  # MuJoCo joint object
                "mujoco:equality_constraint_joint1": indices[0],
                "mujoco:equality_constraint_joint2": indices[1],
                "mujoco:equality_constraint_polycoef": equality.polycoef,
                "mujoco:equality_constraint_label": equality.path,
                "mujoco:equality_constraint_enabled": equality.enabled,
                "mujoco:equality_constraint_world": builder.current_world,
                "mujoco:equality_constraint_target_kind": 2,  # projected Newton mimic
                "mujoco:equality_constraint_target": target,
                "mujoco:eq_solref": equality.solref,
                "mujoco:eq_solimp": equality.solimp,
            }
        )


def build_equalities(
    builder: Any, equalities: dict[str, _BodyEquality], bodies: dict[str, int], scales: dict[str, np.ndarray]
) -> bool:
    """Add projected loop joints and the authoritative MuJoCo equality rows."""
    for equality in equalities.values():
        joint, path = equality.joint, equality.path
        for target in (joint.body0, joint.body1):
            if target is not None and target not in bodies:
                raise UnsupportedPhysicsError(f"equality {path} must target a rigid body or world, not {target}")
        body0, body1 = bodies.get(joint.body0, -1), bodies.get(joint.body1, -1)
        if body0 == body1:
            raise InvalidPhysicsError(f"equality {path} must connect distinct bodies, with at least one rigid body")
        pos0 = wp.vec3(*(np.asarray(joint.local_pos0) * scales.get(joint.body0, np.ones(3))))
        pos1 = wp.vec3(*(np.asarray(joint.local_pos1) * scales.get(joint.body1, np.ones(3))))
        anchor = pos1 if equality.weld else pos0
        rotation = wp.quat(*joint.local_rot0) * wp.quat_inverse(wp.quat(*joint.local_rot1))
        relative_pose = wp.transform(pos0 - wp.quat_rotate(rotation, pos1), rotation)
        parent, child = (body0, body1) if body1 >= 0 else (-1, body0)
        world_anchor = wp.transform_point(builder.body_q[body0], anchor) if body0 >= 0 else anchor
        parent_anchor = (
            wp.transform_point(wp.transform_inverse(builder.body_q[parent]), world_anchor)
            if parent >= 0
            else world_anchor
        )
        child_anchor = wp.transform_point(wp.transform_inverse(builder.body_q[child]), world_anchor)
        add_joint = builder.add_joint_fixed if equality.weld else builder.add_joint_ball
        target = add_joint(
            parent=parent,
            child=child,
            parent_xform=wp.transform(parent_anchor, wp.quat_identity()),
            child_xform=wp.transform(child_anchor, wp.quat_identity()),
            label=path,
        )
        # The public custom-value contract links this row to its projected joint,
        # so SolverMuJoCo exports the equality once, using its original parameters.
        builder.add_custom_values(
            **{
                "mujoco:equality_constraint_type": 1 if equality.weld else 0,  # WELD / CONNECT
                "mujoco:equality_constraint_objtype": 1,  # MuJoCo body object
                "mujoco:equality_constraint_body1": body0,
                "mujoco:equality_constraint_body2": body1,
                "mujoco:equality_constraint_anchor": anchor,
                "mujoco:equality_constraint_relpose": relative_pose if equality.weld else wp.transform_identity(),
                "mujoco:equality_constraint_torquescale": equality.torque_scale if equality.weld else 0.0,
                "mujoco:equality_constraint_label": path,
                "mujoco:equality_constraint_enabled": True,
                "mujoco:equality_constraint_world": builder.current_world,
                "mujoco:equality_constraint_target_kind": 1,  # projected Newton joint
                "mujoco:equality_constraint_target": target,
                "mujoco:eq_solref": equality.solref,
                "mujoco:eq_solimp": equality.solimp,
            }
        )
    return bool(equalities)
