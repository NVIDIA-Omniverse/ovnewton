# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pxr-free names shared by ovnewton's ovstage transport paths."""

USD_PATH = "usd-path"
USD_PARENT = "usd-parent"
USD_PRIM_TYPE = "usd-prim-type"
USD_SCHEMAS = "usd-schemas"

WORLD_MATRIX = "omni:fabric:worldMatrix"
BODY_VELOCITY = "physics:velocity"
BODY_ANGULAR_VELOCITY = "physics:angularVelocity"
RIGID_BODY_ENABLED = "physics:rigidBodyEnabled"
COLLISION_ENABLED = "physics:collisionEnabled"
JOINT_LOCAL_POS_0 = "physics:localPos0"
NEWTON_CONTACT_ADHESION = "newton:contactAdhesion"
NEWTON_CONTACT_DAMPING = "newton:contactDamping"
NEWTON_CONTACT_FRICTION_GAIN = "newton:contactFrictionGain"
NEWTON_CONTACT_MARGIN = "newton:contactMargin"
NEWTON_CONTACT_GAP = "newton:contactGap"
NEWTON_CONTACT_STIFFNESS = "newton:contactStiffness"
NEWTON_GRAVITY_ENABLED = "newton:gravityEnabled"
NEWTON_HYDROELASTIC_ENABLED = "newton:hydroelasticEnabled"
NEWTON_HYDROELASTIC_STIFFNESS = "newton:hydroelasticStiffness"
NEWTON_INERTIA = "newton:inertia"
NEWTON_JOINT_ARMATURE = "newton:armature"
NEWTON_JOINT_DAMPING = "newton:damping"
NEWTON_JOINT_FRICTION = "newton:friction"
NEWTON_JOINT_LIMIT_DAMPING = "newton:limitDamping"
NEWTON_JOINT_LIMIT_STIFFNESS = "newton:limitStiffness"
NEWTON_JOINT_VELOCITY_LIMIT = "newton:velocityLimit"
NEWTON_MASS_MODEL = "newton:massModel"
NEWTON_MAX_HULL_VERTICES = "newton:maxHullVertices"
NEWTON_SDF_MAX_RESOLUTION = "newton:sdfMaxResolution"
NEWTON_SDF_NARROW_BAND_INNER = "newton:sdfNarrowBandInner"
NEWTON_SDF_NARROW_BAND_OUTER = "newton:sdfNarrowBandOuter"
NEWTON_SDF_PADDING = "newton:sdfPadding"
NEWTON_SDF_TARGET_VOXEL_SIZE = "newton:sdfTargetVoxelSize"
NEWTON_SDF_TEXTURE_FORMAT = "newton:sdfTextureFormat"
NEWTON_ROLLING_FRICTION = "newton:rollingFriction"
NEWTON_SHELL_THICKNESS = "newton:shellThickness"
NEWTON_TORSIONAL_FRICTION = "newton:torsionalFriction"


def joint_state(instance: str, name: str) -> str:
    return f"state:{instance}:physics:{name}"


def joint_drive(instance: str, name: str) -> str:
    return f"drive:{instance}:physics:{name}"


def joint_limit(instance: str, name: str) -> str:
    return f"limit:{instance}:physics:{name}"
