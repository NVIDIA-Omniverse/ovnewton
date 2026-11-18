# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Look up the attributes that ``StageBinding.read()`` can return.

Use it like this::

    from ovnewton import attributes

    field = attributes.get_attribute("ARTICULATION_JOINT", "joint_q")
    fields = attributes.list_attributes("ARTICULATION_JOINT", readable=True)

An object type is a ``SimObjectType`` member or its name, such as
``"RIGID_BODY"``. Names are not case sensitive. The lookup returns
``ReadField`` objects, which are the same declarations as ``read_fields()``.
Each one has structured units in ``unit_terms``.

There are two families of names. A snake_case name, such as ``body_q`` or
``joint_q``, is a Newton array. It returns Newton's values, with Newton's
units and frames. A camelCase name, such as ``position`` or ``jointPosition``,
is a standard attribute. Its names follow the ovphysx attribute names. For
example, ``body_q`` holds position and orientation in one row of 7 values. The
standard attributes ``position`` and ``orientation`` hold the same data as
separate attributes. ``ReadField.newton_fields`` links a standard attribute
to the Newton fields that read the same arrays. Importing this module does
not load the simulation stack.
"""

from __future__ import annotations

from enum import Enum

from ._src._read_fields import JointLayout as JointLayout
from ._src._read_fields import JointLayoutPart as JointLayoutPart
from ._src._read_fields import ReadField as ReadField
from ._src._read_fields import UnitTerm as UnitTerm
from ._src._read_fields import read_fields

__all__ = ["JointLayout", "JointLayoutPart", "ReadField", "UnitTerm", "get_attribute", "list_attributes"]

# Declaration objects that each object type reads. A "joint" field reads any
# bound joint, so it is also readable for ARTICULATION_JOINT.
_OBJECT_TYPES = {
    "RIGID_BODY": ("rigid_body",),
    "ARTICULATION_LINK": ("articulation_link",),
    "ARTICULATION": ("articulation_root",),
    "PHYSICS_SCENE": ("physics_scene",),
    "ARTICULATION_JOINT": ("articulation_joint", "joint"),
    "JOINT": ("joint",),
}


def _objects(object_type: str | Enum) -> tuple[str, ...]:
    if isinstance(object_type, Enum):
        name = object_type.name
    elif isinstance(object_type, str):
        name = object_type
    else:
        raise TypeError(f"object_type must be a string or SimObjectType, not {type(object_type).__name__}")
    name = name.upper()
    if name not in _OBJECT_TYPES:
        raise ValueError(f"unknown object type {name!r}; expected one of {sorted(_OBJECT_TYPES)}")
    return _OBJECT_TYPES[name]


def list_attributes(object_type: str | Enum, *, readable: bool | None = None) -> list[ReadField]:
    """Return the attributes of one object type, in declaration order.

    Args:
        object_type: A ``SimObjectType`` member or its name.
        readable: ``True`` keeps attributes that ``read()`` returns. ``False`` keeps
            the others. ``None`` keeps all. Every declared attribute is readable
            today, so ``False`` returns an empty list.

    Raises:
        ValueError: The object type is unknown.
    """
    objects = _objects(object_type)
    result = []
    if readable is False:
        return result
    for field in read_fields().values():
        for kind in objects:
            if kind in field.objects:
                result.append(field)
                break
    return result


def get_attribute(object_type: str | Enum, name: str) -> ReadField:
    """Return one attribute of an object type.

    Raises:
        ValueError: The object type is unknown.
        KeyError: The object type has no attribute with that name.
    """
    for field in list_attributes(object_type):
        if field.name == name:
            return field
    label = object_type.name if isinstance(object_type, Enum) else object_type
    raise KeyError(f"{label} has no attribute {name!r}")
