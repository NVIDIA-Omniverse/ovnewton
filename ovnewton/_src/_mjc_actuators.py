# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import MuJoCo scalar position/velocity actuators using Newton joint targets."""

from collections import Counter

import newton
import numpy as np
import warp as wp

from . import _mjc, _parse, _stage
from ._errors import InvalidPhysicsError, UnsupportedPhysicsError
from ._schemas import _MJC_ACTUATOR_PROPERTIES, _MJC_ACTUATOR_UNSUPPORTED_PROPERTIES


def build_actuators(builder, stage, pd, ordinal, joint_by_path, hierarchy):
    paths = _parse._type_paths(stage, pd, "MjcActuator", ordinal)
    if not paths:
        return
    instanced = {path for path in paths if _parse._is_generated_prototype_path(path)}
    instanced.update(_parse._native_instance_paths(stage, hierarchy, [path for path in paths if path not in instanced]))
    if instanced:
        raise UnsupportedPhysicsError(
            f"native USD actuator instancing is unsupported at {min(instanced)}; "
            "flatten physics instances before attachment"
        )
    if "mujoco:actuator_target_label" not in builder.custom_attributes:
        raise UnsupportedPhysicsError(
            "MjcActuator requires SolverMuJoCo.register_custom_attributes(builder) before add_ovstage"
        )
    with _stage._path_list_query(stage, pd, paths) as query:
        columns = _stage.read_columns(
            stage, pd, query, _MJC_ACTUATOR_PROPERTIES, ordinal, ragged=_MJC_ACTUATOR_PROPERTIES
        )
    ctrl_type = newton.solvers.SolverMuJoCo.CtrlType
    ctrl_source = newton.solvers.SolverMuJoCo.CtrlSource
    source_counts = Counter(joint_by_path.values())
    prepared = []
    targets_seen = set()
    for row, path in enumerate(paths):

        def vector(name, default):
            return _mjc._value(columns[name].get(row), default, path, name)

        def number(name, default):
            return vector(name, (default,))[0]

        def token(name, default):
            values = columns[name].get(row)
            if values is None:
                return default
            if len(values) != 1:
                raise InvalidPhysicsError(f"invalid {name} at {path}; expected one token")
            return _stage._token_to_string(pd, int(values[0]))

        targets = columns["mjc:target"].get(row, ())
        if len(targets) != 1:
            raise InvalidPhysicsError(f"MjcActuator requires exactly one mjc:target at {path}")
        target = _stage._path_to_string(pd, int(targets[0]))
        if target not in hierarchy.parents:
            raise InvalidPhysicsError(f"MjcActuator at {path} has missing target {target}")
        joint = joint_by_path.get(target)
        if (
            joint is None
            or source_counts[joint] != 1
            or builder.joint_type[joint] not in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)
        ):
            raise UnsupportedPhysicsError(f"MjcActuator at {path} requires an unmerged scalar joint: {target}")
        dof = builder.joint_qd_start[joint]
        gear = vector("mjc:gear", (1, 0, 0, 0, 0, 0))
        if (
            token("mjc:dynType", "none") != "none"
            or token("mjc:gainType", "fixed") != "fixed"
            or token("mjc:biasType", "none") != "affine"
            or not np.allclose(gear, (1, 0, 0, 0, 0, 0))
        ):
            raise UnsupportedPhysicsError(
                f"MjcActuator at {path} requires fixed gain, affine bias and unit gear without dynamics"
            )
        gain = vector("mjc:gainPrm", (1,) + (0,) * 9)
        bias = vector("mjc:biasPrm", (0,) * 10)
        kp = float(gain[0])
        position = kp > 0 and np.isclose(bias[0], 0) and np.isclose(bias[1], -kp) and bias[2] <= 0
        velocity = kp > 0 and np.isclose(bias[0], 0) and np.isclose(bias[1], 0) and np.isclose(bias[2], -kp)
        dampratio = kp > 0 and np.isclose(bias[0], 0) and np.isclose(bias[1], -kp) and bias[2] > 0
        if not (position or velocity or dampratio):
            raise UnsupportedPhysicsError(f"MjcActuator at {path} is not a supported position or velocity actuator")
        for name in _MJC_ACTUATOR_UNSUPPORTED_PROPERTIES:
            if row in columns[name]:
                raise UnsupportedPhysicsError(f"unsupported {name} on MjcActuator at {path}")

        kind = ctrl_type.POSITION if position else ctrl_type.VELOCITY if velocity else ctrl_type.GENERAL
        key = (dof, kind)
        if key in targets_seen:
            raise UnsupportedPhysicsError(f"multiple MuJoCo actuators of the same kind target {target}")
        targets_seen.add(key)

        values = {
            "actuator_label": path,
            "actuator_target_label": target,
            "actuator_trnid": wp.vec2i(-1, -1) if dampratio else wp.vec2i(dof, 0),
            "ctrl_source": int(ctrl_source.CTRL_DIRECT if dampratio else ctrl_source.JOINT_TARGET),
            "ctrl_type": int(ctrl_type.GENERAL if dampratio else kind),
            "actuator_biastype": 1,
        }
        for name, value in (("gainprm", gain), ("biasprm", bias), ("gear", gear)):
            values[f"actuator_{name}"] = builder.custom_attributes[f"mujoco:actuator_{name}"].dtype(*value)
        for prefix in ("ctrl", "force"):
            low, high = f"mjc:{prefix}Range:min", f"mjc:{prefix}Range:max"
            bounds = (number(low, 0.0), number(high, 0.0)) if row in columns[low] or row in columns[high] else None
            inherited = False
            if prefix == "ctrl":
                scale = number("mjc:inheritRange", 0.0)
                ref = builder.custom_attributes["mujoco:dof_ref"]
                offset = ref.values.get(dof, ref.default)
                lower, upper = builder.joint_limit_lower[dof] + offset, builder.joint_limit_upper[dof] + offset
                inherited = scale > 0 and lower < upper
                if bounds is None and inherited:
                    center, radius = (lower + upper) / 2, (upper - lower) * scale / 2
                    bounds = (center - radius, center + radius)
            limited = token(f"mjc:{prefix}Limited", "true" if inherited else "auto")
            if limited not in ("false", "true", "auto"):
                raise InvalidPhysicsError(f"invalid mjc:{prefix}Limited at {path}")
            values[f"actuator_{prefix}limited"] = {"false": 0, "true": 1, "auto": 2}[limited]
            values[f"actuator_has_{prefix}range"] = int(bounds is not None)
            if bounds is not None:
                values[f"actuator_{prefix}range"] = wp.vec2(*bounds)
        prepared.append((dof, kp, float(-bias[2]), values))

    # Match add_usd's joint-target classification and gain precedence. Keep a
    # custom actuator row so SolverMuJoCo retains each actuator's control limits.
    for dof, gain, damping, values in prepared:
        builder.custom_attributes["mujoco:joint_dof_label"].values[dof] = values["actuator_target_label"]
        kind = values["ctrl_type"]
        if kind == ctrl_type.GENERAL:
            # add_usd leaves positive biasPrm[2] for MuJoCo to resolve via mj_setConst.
            builder.add_custom_values(**{f"mujoco:{name}": value for name, value in values.items()})
            continue
        current = builder.joint_target_mode[dof]
        mode = newton.JointTargetMode
        if kind == ctrl_type.POSITION:
            builder.joint_target_ke[dof] = gain
            if current == mode.VELOCITY:
                builder.joint_target_mode[dof] = mode.POSITION_VELOCITY
            elif current == mode.NONE:
                builder.joint_target_mode[dof] = mode.POSITION
                builder.joint_target_kd[dof] = damping
        else:
            builder.joint_target_kd[dof] = gain
            if current == mode.POSITION:
                builder.joint_target_mode[dof] = mode.POSITION_VELOCITY
            elif current == mode.NONE:
                builder.joint_target_mode[dof] = mode.VELOCITY
        builder.add_custom_values(**{f"mujoco:{name}": value for name, value in values.items()})
