# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata types for ``StageBinding.read()`` fields.

The values live in the generated ``_output_descriptors`` module. Edit
``contracts/read_fields.json`` and run ``tools/generate_read_fields.py`` to
change them. Keep this module free of runtime dependencies, so tools and
documentation can load it without the simulation stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

ReadFieldObject = Literal[
    "rigid_body",
    "articulation_link",
    "articulation_root",
    "physics_scene",
    "articulation_joint",
    "joint",
]
ReadFieldRequirement = Literal["state", "control", "model"]
ReadFieldFrame = Literal["world", "body", "com", "joint"]
ReadFieldStorage = Literal["borrowed", "borrowed_or_gathered", "gathered", "query_owned"]


@dataclass(frozen=True)
class _OutputDescriptor:
    """How the runtime produces one field. Private."""

    domain: str
    owner: str
    source: str
    width: int
    conversion: str | None = None


@dataclass(frozen=True)
class JointLayoutPart:
    """One run of values in a joint coordinate row."""

    #: The ``UnitTerm`` scope that gives the units of this run.
    scope: str
    #: The number of values, or ``"configured"`` for the scalar axes that a D6 joint enables.
    count: int | str


@dataclass(frozen=True)
class JointLayout:
    """How one joint representation packs its values into a row.

    A field's ``unit_terms`` give units for every kind of value the field can
    hold. A layout says which of them one joint representation uses, in row
    order. Picking the layout still depends on the live joint's type.
    """

    #: ``prismatic``, ``revolute``, ``ball``, ``free``, ``distance`` or ``d6``.
    representation: str
    #: The runs of values, in row order.
    parts: tuple[JointLayoutPart, ...]


@dataclass(frozen=True)
class UnitTerm:
    """The unit of some values in a field, as powers of base dimensions.

    ``dimensions`` is a tuple of ``(name, exponent)`` pairs. The names are
    ``mass``, ``length``, ``time`` and ``angle``. For example, a velocity in
    length units per second is ``(("length", 1), ("time", -1))``. Length and
    mass use the model's units. Time is in seconds.
    """

    #: The values this term covers. ``"all"`` covers every value in a row.
    #: ``"linear_axes"`` and ``"angular_axes"`` cover the scalar joint axes of that
    #: kind. ``"translation"`` covers the three XYZ coordinates of a free or distance
    #: joint. ``"quaternion"`` covers the four XYZW components of a ball, free or
    #: distance joint. It has no dimensions, so its components are not angles.
    #: ``"linear_components"`` and ``"angular_components"`` cover the linear and
    #: angular parts of a spatial vector. A tuple names components.
    scope: str | tuple[str, ...]
    dimensions: tuple[tuple[str, int], ...]
    #: ``"radian"`` or ``"degree"`` when ``dimensions`` include ``angle``, else ``None``.
    angle_unit: Literal["radian", "degree"] | None = None


@dataclass(frozen=True)
class ReadField:
    """What one ``StageBinding.read()`` field returns.

    Every value comes from ``contracts/read_fields.json``. Tests check each
    declaration against actual reads.
    """

    #: The field name to pass in ``attributes``. A snake_case name is a Newton
    #: array name and returns Newton's values. A camelCase name is a standard
    #: field; its per-axis joint values use USD joint units on angular axes.
    name: str
    #: One-sentence description.
    summary: str
    #: Object kinds that can produce rows: ``rigid_body``, ``articulation_link``,
    #: ``articulation_root``, ``physics_scene``, ``articulation_joint`` or
    #: ``joint`` (any bound joint, including articulation joints).
    objects: tuple[ReadFieldObject, ...]
    #: Which prim a row belongs to.
    identity: str
    #: The Newton arrays the values come from, such as ``"State.body_q"``.
    sources: tuple[str, ...]
    #: For a standard field, the Newton fields that read the same Newton
    #: arrays, such as ``("joint_qd",)`` for ``jointVelocity``. Empty for a
    #: Newton field, and for a standard field whose arrays have no Newton field.
    newton_fields: tuple[str, ...]
    #: The ``read()`` input that must be supplied: ``state``, ``control``, or
    #: ``model`` when the connected model alone is enough.
    requires: ReadFieldRequirement
    #: Data type of each value: ``float32`` or ``int32``.
    dtype: str
    #: Values per row, or ``None`` when the count depends on the selection.
    lanes: int | None
    #: How the value count is chosen when ``lanes`` is ``None``.
    lanes_rule: str | None
    #: Name of each value in a row, when ``lanes`` is fixed.
    components: tuple[str, ...] | None
    #: How values are ordered in a row.
    layout: str
    #: Units of the values. For joint fields, this gives linear and angular axes separately.
    units: str
    #: The same units as structured terms. Empty when the values have no units.
    unit_terms: tuple[UnitTerm, ...]
    #: Reference frame of the values, or ``None`` when no frame applies.
    frame: str | None
    #: A short tag for ``frame``. ``world``. ``body`` (the body's origin frame).
    #: ``com`` (the body frame at the center of mass). ``joint`` (the parent joint
    #: frame, which also defines the joint axes). ``None`` when no frame applies.
    #: Exceptions for free and distance joints are in ``frame``.
    frame_tag: ReadFieldFrame | None
    #: The conversion applied to Newton's values, or ``None`` for native values.
    conversion: str | None
    #: How the column relates to Newton memory: ``borrowed`` (a live view of the
    #: Newton array), ``gathered`` (a buffer written by this read),
    #: ``borrowed_or_gathered`` (borrowed when the selection allows it), or
    #: ``query_owned`` (computed once when the query was created).
    storage: ReadFieldStorage
    #: Where the payload lives. ``model`` means the connected model's device.
    device: str
    #: For a joint field whose row layout depends on the joint type, one layout
    #: for each joint representation. Empty when every joint packs the row the same way.
    joint_layouts: tuple[JointLayout, ...] = ()


def read_fields() -> Mapping[str, ReadField]:
    """Return the declaration of every field that ``StageBinding.read()`` accepts.

    The mapping is read-only and keyed by field name. No simulation code is
    loaded.
    """
    from ._output_descriptors import _READ_FIELDS  # noqa: PLC0415

    return _READ_FIELDS
