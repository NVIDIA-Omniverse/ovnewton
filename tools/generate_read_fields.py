# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate read-field metadata from ``contracts/read_fields.json``.

The JSON file is the single source of truth for every ``StageBinding.read()``
field. This script validates it and writes two committed artifacts:

- ``ovnewton/_src/_output_descriptors.py``: name constants, runtime
  descriptors and public ``ReadField`` metadata.
- ``docs/site/read-fields.md``: the public field reference.

Usage::

    python tools/generate_read_fields.py --write   # rewrite both artifacts
    python tools/generate_read_fields.py --check   # fail when either one drifts

The script uses only the standard library and does not import ovnewton.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT = Path("contracts/read_fields.json")
PYTHON_OUTPUT = Path("ovnewton/_src/_output_descriptors.py")
MARKDOWN_OUTPUT = Path("docs/site/read-fields.md")

_LINE_LENGTH = 120
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SOURCE_PATTERN = re.compile(r"^(State|Model|Control)\.[a-z_]+$")
# Newlines and other control characters would break a Markdown table row, and lone
# surrogates cannot be written as UTF-8.
_CONTROL_CHARACTERS = re.compile("[\x00-\x1f\x7f-\x9f\ud800-\udfff]")
_OBJECTS = {
    "rigid_body": "Rigid body (`SimObjectType.RIGID_BODY`).",
    "articulation_link": "Articulation link (`SimObjectType.ARTICULATION_LINK`).",
    "articulation_root": "Prim with an authored articulation-root API (`SimObjectType.ARTICULATION`).",
    "physics_scene": "The effective PhysicsScene prim (`SimObjectType.PHYSICS_SCENE`).",
    "articulation_joint": (
        "Articulation joint with coordinates or degrees of freedom (`SimObjectType.ARTICULATION_JOINT`)."
    ),
    "joint": (
        "Any bound joint with coordinates or degrees of freedom. Articulation joints can be selected with "
        "`SimObjectType.ARTICULATION_JOINT`. Other joints are selected only by queries without an object type."
    ),
}
_STORAGE = {
    "borrowed": (
        "A read-only view of the Newton array, with no copy. It can include unselected rows; use the "
        "column's row map. It shows later writes to that array. Do not change the array until every "
        "reader has finished."
    ),
    "gathered": (
        "A new buffer that this read fills on the model device. It holds a copy and does not change "
        "afterwards. It stays valid while the result, a group, a column or one of its DLPack objects is "
        "alive."
    ),
    "borrowed_or_gathered": (
        "Borrowed when the selected joints share one width and their source rows line up. Otherwise "
        "gathered."
    ),
    "query_owned": "Computed once when the query was created. It does not change between reads.",
}
_REQUIRES = {"state", "control", "model"}
_DTYPES = {"float32", "int32"}
_DEVICES = {"model": "The connected model's device."}
_DOMAINS = {
    "body",
    "scene",
    "shape",
    "articulation",
    "joint",
    "joint_native_dof",
    "joint_position",
    "joint_velocity",
    "joint_target_position",
    "joint_limit",
}
_OWNERS = {"state", "model", "control", "query"}
_OPERATIONS = {
    None,
    "transform_position",
    "transform_orientation",
    "spatial_linear",
    "spatial_angular",
    "angular_degrees",
    "angular_per_degree",
}
_UNIT_OPERATIONS = {"angular_degrees", "angular_per_degree"}
_UNIT_DIMENSIONS = ("mass", "length", "time", "angle")
_UNIT_SCOPES = {
    "all",
    "linear_axes",
    "angular_axes",
    "translation",
    "quaternion",
    "linear_components",
    "angular_components",
}
_JOINT_REPRESENTATIONS = ("prismatic", "revolute", "ball", "free", "distance", "d6")
_ANGLE_UNITS = {"radian", "degree"}
_FRAME_TAGS = {"world", "body", "com", "joint"}
_FIELD_KEYS = {
    "name",
    "summary",
    "objects",
    "identity",
    "sources",
    "newton_fields",
    "requires",
    "dtype",
    "lanes",
    "components",
    "layout",
    "units",
    "unit_terms",
    "frame",
    "frame_tag",
    "conversion",
    "storage",
    "device",
    "runtime",
}
_OPTIONAL_FIELD_KEYS = {"lanes_rule", "joint_layouts"}
_RUNTIME_KEYS = {"domain", "owner", "array", "operation"}


class ContractError(ValueError):
    """The read-field contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _text(field: str, key: str, value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    _require(isinstance(value, str) and value.strip() == value and value, f"{field}: {key} must be non-empty text")
    _require(
        _CONTROL_CHARACTERS.search(value) is None,
        f"{field}: {key} must be one line of text without control characters or lone surrogates",
    )


def _string_list(field: str, key: str, value: Any, allowed: Mapping[str, str] | None = None) -> None:
    _require(isinstance(value, list) and value, f"{field}: {key} must be a non-empty list")
    _require(len(set(value)) == len(value), f"{field}: {key} has duplicates")
    for item in value:
        _text(field, key, item)
        if allowed is not None:
            _require(item in allowed, f"{field}: {key} has unknown value {item!r}")


def _validate_unit_terms(name: str, terms: Any, components: Sequence[str] | None) -> None:
    """Check the structured units: a list of terms, empty when the values have no units."""
    _require(isinstance(terms, list), f"{name}: unit_terms must be a list")
    for term in terms:
        _require(isinstance(term, dict), f"{name}: each unit term must be an object")
        for key in ("scope", "dimensions"):
            _require(key in term, f"{name}: each unit term needs {key}")
        for key in term:
            _require(key in ("scope", "dimensions", "angle_unit"), f"{name}: unknown unit term key {key!r}")

        scope = term["scope"]
        if isinstance(scope, str):
            _require(scope in _UNIT_SCOPES, f"{name}: unknown unit scope {scope!r}")
        else:
            _string_list(name, "unit scope", scope)
            _require(components is not None, f"{name}: a component scope needs fixed lanes")
            for component in scope:
                _require(component in components, f"{name}: unit scope names unknown component {component!r}")

        dimensions = term["dimensions"]
        _require(isinstance(dimensions, dict), f"{name}: dimensions must be an object")
        if scope == "quaternion":
            _require(len(dimensions) == 0, f"{name}: a quaternion term has no dimensions")
        else:
            _require(len(dimensions) > 0, f"{name}: dimensions must be a non-empty object")
        has_angle = False
        for dimension, exponent in dimensions.items():
            _require(dimension in _UNIT_DIMENSIONS, f"{name}: unknown dimension {dimension!r}")
            _require(
                isinstance(exponent, int) and not isinstance(exponent, bool) and exponent != 0,
                f"{name}: dimension exponents must be non-zero integers",
            )
            if dimension == "angle":
                has_angle = True

        angle_unit = term.get("angle_unit")
        if has_angle:
            _require(angle_unit in _ANGLE_UNITS, f"{name}: angle_unit must be one of {sorted(_ANGLE_UNITS)}")
        else:
            _require(angle_unit is None, f"{name}: angle_unit is only for dimensions that include angle")


def _validate_joint_layouts(
    name: str, layouts: Any, objects: Sequence[str], terms: Sequence[Mapping[str, Any]]
) -> None:
    """Check the per-representation row layouts of a joint field."""
    is_joint = "joint" in objects or "articulation_joint" in objects
    _require(is_joint, f"{name}: joint_layouts are only for joint fields")
    _require(isinstance(layouts, dict) and len(layouts) > 0, f"{name}: joint_layouts must be a non-empty object")
    term_scopes = []
    for term in terms:
        if isinstance(term["scope"], str):
            term_scopes.append(term["scope"])
    for representation, parts in layouts.items():
        _require(representation in _JOINT_REPRESENTATIONS, f"{name}: unknown joint representation {representation!r}")
        _require(isinstance(parts, list), f"{name}: layout {representation} must be a list")
        _require(len(parts) > 0, f"{name}: layout {representation} must be a non-empty list")
        for part in parts:
            _require(isinstance(part, dict), f"{name}: each layout part must be an object")
            for key in ("scope", "count"):
                _require(key in part, f"{name}: each layout part needs {key}")
            for key in part:
                _require(key in ("scope", "count"), f"{name}: unknown layout part key {key!r}")
            scope = part["scope"]
            count = part["count"]
            _require(scope in term_scopes, f"{name}: layout {representation} uses scope {scope!r} with no unit term")
            if count == "configured":
                _require(
                    representation == "d6" and scope in ("linear_axes", "angular_axes"),
                    f"{name}: only d6 scalar axes can have the count 'configured'",
                )
                continue
            _require(
                isinstance(count, int) and not isinstance(count, bool) and count > 0,
                f"{name}: layout counts must be positive integers or 'configured'",
            )
            if scope == "translation":
                _require(count == 3, f"{name}: a translation has 3 values")
            if scope == "quaternion":
                _require(count == 4, f"{name}: a quaternion has 4 values")


def _validate_field(entry: Mapping[str, Any]) -> None:
    name = entry["name"]
    keys = set(entry)
    _require(_FIELD_KEYS <= keys, f"{name}: missing keys {sorted(_FIELD_KEYS - keys)}")
    _require(
        keys <= _FIELD_KEYS | _OPTIONAL_FIELD_KEYS,
        f"{name}: unknown keys {sorted(keys - _FIELD_KEYS - _OPTIONAL_FIELD_KEYS)}",
    )
    for key in ("summary", "identity", "layout", "units"):
        _text(name, key, entry[key])
    _text(name, "frame", entry["frame"], optional=True)
    if entry["frame"] is None:
        _require(entry["frame_tag"] is None, f"{name}: frame_tag must be null when there is no frame")
    else:
        _require(entry["frame_tag"] in _FRAME_TAGS, f"{name}: frame_tag must be one of {sorted(_FRAME_TAGS)}")
    _text(name, "conversion", entry["conversion"], optional=True)
    _string_list(name, "objects", entry["objects"], _OBJECTS)
    _string_list(name, "sources", entry["sources"])
    for source in entry["sources"]:
        _require(_SOURCE_PATTERN.match(source) is not None, f"{name}: source {source!r} is not Owner.array")
    _require(entry["requires"] in _REQUIRES, f"{name}: requires must be one of {sorted(_REQUIRES)}")
    owners = {source.split(".", 1)[0] for source in entry["sources"]}
    expected_requirement = "state" if "State" in owners else "control" if "Control" in owners else "model"
    _require(
        entry["requires"] == expected_requirement,
        f"{name}: requires must be {expected_requirement!r} for sources {entry['sources']}",
    )
    _require(entry["dtype"] in _DTYPES, f"{name}: dtype must be one of {sorted(_DTYPES)}")
    _require(entry["storage"] in _STORAGE, f"{name}: storage must be one of {sorted(_STORAGE)}")
    _require(entry["device"] in _DEVICES, f"{name}: device must be one of {sorted(_DEVICES)}")

    lanes = entry["lanes"]
    components = entry["components"]
    if lanes is None:
        _require(components is None, f"{name}: components need fixed lanes")
        _text(name, "lanes_rule", entry.get("lanes_rule"))
    else:
        _require(
            isinstance(lanes, int) and not isinstance(lanes, bool) and lanes > 0,
            f"{name}: lanes must be a positive integer or null",
        )
        _require("lanes_rule" not in entry, f"{name}: lanes_rule is only for variable lanes")
        _string_list(name, "components", components)
        _require(len(components) == lanes, f"{name}: components must name each of the {lanes} lanes")

    _validate_unit_terms(name, entry["unit_terms"], entry["components"])
    if "joint_layouts" in entry:
        _validate_joint_layouts(name, entry["joint_layouts"], entry["objects"], entry["unit_terms"])

    runtime = entry["runtime"]
    _require(isinstance(runtime, dict) and set(runtime) == _RUNTIME_KEYS, f"{name}: runtime needs {_RUNTIME_KEYS}")
    _require(runtime["domain"] in _DOMAINS, f"{name}: unknown runtime domain {runtime['domain']!r}")
    _require(runtime["owner"] in _OWNERS, f"{name}: unknown runtime owner {runtime['owner']!r}")
    _text(name, "runtime.array", runtime["array"])
    _require(runtime["operation"] in _OPERATIONS, f"{name}: unknown runtime operation {runtime['operation']!r}")
    if runtime["owner"] != "query":
        primary = f"{runtime['owner'].capitalize()}.{runtime['array']}"
        _require(entry["sources"][0] == primary, f"{name}: first source must be the runtime array {primary}")

    converted = runtime["operation"] in _UNIT_OPERATIONS
    _require(
        converted == (entry["conversion"] is not None),
        f"{name}: conversion text must be present exactly when the runtime converts units",
    )
    # snake_case names are Newton's own array names. They promise Newton's
    # values, so they must name their array and never convert.
    if "_" in name:
        _require(
            name == runtime["array"] and runtime["operation"] is None,
            f"{name}: a snake_case name is a Newton field; it must equal runtime.array and not convert",
        )


def load_contract(path: Path) -> list[dict[str, Any]]:
    """Validate the contract and return every field."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"{path}: {error}") from error
    _require(isinstance(document, dict), "the contract must be a JSON object")
    _require(
        set(document) == {"comment", "format_version", "fields"},
        "top level needs comment, format_version, fields",
    )
    _require(document["format_version"] == 1, "format_version must be 1")
    entries = document["fields"]
    _require(isinstance(entries, list) and entries, "fields must be a non-empty list")

    names = []
    for entry in entries:
        _require(isinstance(entry, dict) and isinstance(entry.get("name"), str), "each field needs a name")
        _require(_NAME_PATTERN.match(entry["name"]) is not None, f"{entry['name']!r} is not a valid field name")
        names.append(entry["name"])
    duplicates = sorted({name for name in names if names.count(name) > 1})
    _require(not duplicates, f"duplicate field names: {duplicates}")
    constants = [_constant(name) for name in names]
    _require(len(set(constants)) == len(constants), "two field names map to the same Python constant")

    for entry in entries:
        _validate_field(entry)
    _validate_newton_links(entries)
    return [dict(entry) for entry in entries]


def _validate_newton_links(entries: Sequence[Mapping[str, Any]]) -> None:
    """Require each standard field to list exactly the Newton fields that read its source arrays."""
    newton_by_source = {entry["sources"][0]: entry["name"] for entry in entries if "_" in entry["name"]}
    for entry in entries:
        name = entry["name"]
        linked = entry["newton_fields"]
        _require(
            isinstance(linked, list) and all(isinstance(item, str) for item in linked),
            f"{name}: newton_fields must be a list of field names",
        )
        if "_" in name:
            _require(linked == [], f"{name}: a Newton field has no newton_fields")
            continue
        expected = [newton_by_source[source] for source in entry["sources"] if source in newton_by_source]
        _require(
            linked == expected,
            f"{name}: newton_fields must be {expected}, the Newton fields that read its sources",
        )


def _constant(name: str) -> str:
    # Newton fields get their own prefix, so joint_armature never collides with jointArmature.
    if "_" in name:
        return "_NEWTON_" + name.upper()
    return "_" + re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).upper()


def _quote(value: str) -> str:
    """Return a double-quoted Python string literal for ``value``.

    JSON string syntax is a subset of Python's, and ``json.dumps`` escapes
    quotes, backslashes and every control character, including newlines.
    """
    return json.dumps(value, ensure_ascii=False)


def _wrap(value: str, width: int) -> list[str]:
    """Split text into quoted chunks no wider than ``width``, breaking after spaces."""
    chunks = []
    rest = value
    while len(_quote(rest)) > width:
        cut = rest.rfind(" ", 0, width - 2)
        if cut <= 0:
            cut = width - 3
        chunks.append(rest[: cut + 1])
        rest = rest[cut + 1 :]
    chunks.append(rest)
    return [_quote(chunk) for chunk in chunks]


def _python_value(key: str, value: Any, indent: int) -> list[str]:
    prefix = " " * indent + f"{key}="
    if value is None or isinstance(value, int):
        return [f"{prefix}{value!r},"]
    if isinstance(value, str):
        single = f"{prefix}{_quote(value)},"
        if len(single) <= _LINE_LENGTH:
            return [single]
        inner = " " * (indent + 4)
        chunks = _wrap(value, _LINE_LENGTH - len(inner))
        return [f"{prefix}(", *(inner + chunk for chunk in chunks), " " * indent + "),"]
    if isinstance(value, (list, tuple)):
        items = ", ".join(_quote(item) for item in value)
        text = f"{prefix}({items}{',' if len(value) == 1 else ''}),"
        _require(len(text) <= _LINE_LENGTH, f"{key} tuple is too long to render on one line")
        return [text]
    raise ContractError(f"cannot render {key}={value!r}")


def _unit_terms_lines(terms: Sequence[Mapping[str, Any]], indent: int) -> list[str]:
    pad = " " * indent
    if len(terms) == 0:
        return [f"{pad}unit_terms=(),"]
    lines = [f"{pad}unit_terms=("]
    for term in terms:
        scope = term["scope"]
        if isinstance(scope, str):
            scope_text = _quote(scope)
        else:
            scope_text = "(" + ", ".join(_quote(item) for item in scope) + ",)"
        dims = ""
        for dimension, exponent in term["dimensions"].items():
            dims += f"({_quote(dimension)}, {exponent}),"
        angle = term.get("angle_unit")
        angle_text = "None"
        if angle is not None:
            angle_text = _quote(angle)
        text = f"UnitTerm({scope_text}, ({dims}), {angle_text}),"
        _require(len(pad) + 4 + len(text) <= _LINE_LENGTH, "unit term is too long to render on one line")
        lines.append(f"{pad}    {text}")
    lines.append(f"{pad}),")
    return lines


def _joint_layouts_lines(layouts: Mapping[str, Any], indent: int) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}joint_layouts=("]
    for representation, parts in layouts.items():
        lines.append(f"{pad}    JointLayout(")
        lines.append(f"{pad}        {_quote(representation)},")
        lines.append(f"{pad}        (")
        for part in parts:
            count = part["count"]
            if isinstance(count, str):
                count_text = _quote(count)
            else:
                count_text = str(count)
            lines.append(f"{pad}            JointLayoutPart({_quote(part['scope'])}, {count_text}),")
        lines.append(f"{pad}        ),")
        lines.append(f"{pad}    ),")
    lines.append(f"{pad}),")
    return lines


def render_python(fields: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        f"# Generated by tools/generate_read_fields.py from {CONTRACT.as_posix()}. Do not edit.",
        "# Edit the JSON contract, then run: python tools/generate_read_fields.py --write",
        "",
        '"""Read-field declarations shared by the runtime, public metadata and documentation.',
        "",
        "Keep this module free of runtime dependencies so documentation and tools can",
        "load it without the simulation stack.",
        '"""',
        "",
        "from types import MappingProxyType",
        "",
        "from ._read_fields import JointLayout, JointLayoutPart, ReadField, UnitTerm, _OutputDescriptor",
        "",
    ]
    for field in fields:
        lines.append(f"{_constant(field['name'])} = {_quote(field['name'])}")
    lines += ["", "_OUTPUT_DESCRIPTORS = MappingProxyType(", "    {"]
    for field in fields:
        runtime = field["runtime"]
        arguments = [
            _quote(runtime["domain"]),
            _quote(runtime["owner"]),
            _quote(runtime["array"]),
            str(field["lanes"] or 0),
        ]
        if runtime["operation"] is not None:
            arguments.append(_quote(runtime["operation"]))
        single = f"        {_constant(field['name'])}: _OutputDescriptor({', '.join(arguments)}),"
        if len(single) <= _LINE_LENGTH:
            lines.append(single)
        else:
            lines.append(f"        {_constant(field['name'])}: _OutputDescriptor(")
            lines.append(f"            {', '.join(arguments)}")
            lines.append("        ),")
    lines += ["    }", ")", "_OUTPUT_ATTRIBUTES = tuple(_OUTPUT_DESCRIPTORS)", ""]
    lines += ["_READ_FIELDS = MappingProxyType(", "    {"]
    for field in fields:
        constant = _constant(field["name"])
        lines.append(f"        {constant}: ReadField(")
        lines.append(f"            name={constant},")
        for key in ("summary", "objects", "identity", "sources", "newton_fields", "requires", "dtype", "lanes"):
            lines += _python_value(key, field[key], 12)
        lines += _python_value("lanes_rule", field.get("lanes_rule"), 12)
        for key in ("components", "layout", "units"):
            lines += _python_value(key, field[key], 12)
        lines += _unit_terms_lines(field["unit_terms"], 12)
        for key in ("frame", "frame_tag", "conversion", "storage", "device"):
            lines += _python_value(key, field[key], 12)
        if "joint_layouts" in field:
            lines += _joint_layouts_lines(field["joint_layouts"], 12)
        lines.append("        ),")
    lines += ["    }", ")", ""]
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _unit_terms_text(field: Mapping[str, Any]) -> str:
    """Describe the structured units on one line, such as ``linear_axes: length time^-1``."""
    if len(field["unit_terms"]) == 0:
        return "None."
    parts = []
    for term in field["unit_terms"]:
        scope = term["scope"]
        if not isinstance(scope, str):
            scope = ", ".join(scope)
        dims = []
        for dimension, exponent in term["dimensions"].items():
            if exponent == 1:
                dims.append(dimension)
            else:
                dims.append(f"{dimension}^{exponent}")
        text = f"`{scope}`: `{' '.join(dims)}`"
        if term.get("angle_unit") is not None:
            text += f", angle in {term['angle_unit']}s"
        parts.append(text)
    return "; ".join(parts) + "."


def _joint_layouts_text(field: Mapping[str, Any]) -> str:
    """Describe the row layout of each joint representation on one line."""
    entries = []
    for representation, parts in field["joint_layouts"].items():
        texts = []
        for part in parts:
            texts.append(f"`{part['scope']}` × {part['count']}")
        entries.append(f"`{representation}`: " + ", then ".join(texts))
    return "; ".join(entries) + "."


def _values_text(field: Mapping[str, Any]) -> str:
    if field["lanes"] is None:
        return f"Variable × `{field['dtype']}`. {field['lanes_rule']}"
    components = ", ".join(f"`{component}`" for component in field["components"])
    return f"{field['lanes']} × `{field['dtype']}`: {components}"


def _anchor(name: str) -> str:
    return "field-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def render_markdown(fields: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "",
        f"Generated by tools/generate_read_fields.py from {CONTRACT.as_posix()}. Do not edit.",
        "-->",
        "",
        "# Read field reference",
        "",
        "This page lists every field that `StageBinding.read()` accepts. It is generated from the",
        "same declarations as the runtime and `ovnewton.read_fields()`. Tests check each declaration",
        "against actual reads.",
        "",
        "## Conventions",
        "",
        "- **Newton fields.** A snake_case name, such as `joint_q` or `joint_target_ke`, is the name of",
        "  a Newton array. It returns Newton's values unchanged: Newton's units, frames, body order and",
        "  component order. The application owns any conversion.",
        "- **Standard fields.** A camelCase name, such as `position` or `jointPosition`, is a standard",
        "  field. Per-axis joint fields use the units of the matching USD joint attributes: degrees on",
        "  angular axes, and the authored body order. All other values keep Newton's units. The table",
        "  below lists the angular unit of each field. Each field's *Units* and *Conversion* rows give",
        "  the details. Its *Newton fields* row names the Newton fields that read the same Newton arrays.",
        "- **Model units.** Lengths use the model's length unit. For a model that ovnewton builds, this is",
        "  the stage's length unit. Mass uses the model's mass unit. Time is in seconds.",
        "- **Rows.** Each group lists its prims in `prim_list`. For prim `i`, read row",
        "  `column.data_row_index(i)` of the column's tensor.",
        "- **Readiness.** On CUDA, wait for `column.cuda_sync.wait_event` before reading on another",
        "  stream. On CPU, data is ready when `read()` returns.",
        "- **Discovery.** `ovnewton.read_fields()` returns these declarations as `ReadField` objects.",
        "  `ovnewton.attributes.list_attributes()` and `get_attribute()` look them up by object type.",
        "- **Unit terms.** Each field's *Unit terms* row gives its units as powers of `mass`, `length`,",
        "  `time` and `angle`. For example, `length time^-1` is a velocity. A term covers `all` values, the",
        "  `linear_axes` or `angular_axes` of a joint, or named components. Joint fields give a linear term",
        "  and an angular term. Angular terms name their angle unit, radians or degrees. An empty row",
        "  means the values have no units. A `quaternion` term has no dimensions: its four XYZW components",
        "  are not angles. `translation` is the three XYZ coordinates of a free or distance joint.",
        "- **Joint layouts.** Newton's `joint_q` packs a different row for each joint type. Its *Joint layouts*",
        "  row lists the parts of the row, in order, for each joint representation. Each part names the unit",
        "  term that applies to it. `configured` means the number of scalar axes that a D6 joint enables.",
        "  The live joint's type chooses the layout.",
        "- **Frame tags.** A field's *Frame tag* is a short form of its *Frame* row: `world`; `body` (the",
        "  body's origin frame); `com` (the body frame at the center of mass); or `joint` (the parent joint",
        "  frame, which also defines the joint axes). Exceptions are in the *Frame* row.",
        "",
        "### Angular units",
        "",
        "The angular unit depends on the quantity. The name does not state the unit.",
        "",
        "| Quantity | Fields | Angular unit |",
        "|---|---|---|",
        "| Body angular velocity and acceleration | `angularVelocity`, `angularAcceleration`, "
        "`rootAngularVelocity`, `body_qd`, `body_qdd` | Radians per second, or per second squared. Not "
        "converted. |",
        "| Joint positions, velocities and targets | `jointPosition`, `jointVelocity`, "
        "`jointPositionTarget`, `jointVelocityTarget` | Degrees, or degrees per second, as in USD. |",
        "| Joint limits | `jointLimit`, `jointMaxVelocity` | Degrees, or degrees per second, as in USD. |",
        "| Joint drive gains | `jointStiffness`, `jointDamping` | Per degree, or per degree per second, "
        "as in USD. |",
        "| Joint efforts and armature | `jointMaxForce`, `jointFriction`, `jointArmature`, `joint_f`, "
        "`joint_effort_limit`, `joint_friction`, `joint_armature` | Torque, or moment of inertia. Not "
        "converted. |",
        "| Newton joint fields |`joint_q`, `joint_qd`, `joint_target_*`, `joint_limit_*`, "
        "`joint_velocity_limit` | Radians, radians per second, or per radian, as in Newton. |",
        "",
        "Joint fields that use degrees also follow the authored USD body order (`physics:body0` to",
        "`physics:body1`) where the quantity has a sign. Linear axes are never converted.",
        "",
        "### Objects",
        "",
        "| Object | Meaning |",
        "|---|---|",
    ]
    for key, text in _OBJECTS.items():
        lines.append(f"| `{key}` | {_markdown_cell(text)} |")
    lines += ["", "### Storage and lifetime", "", "| Storage | Meaning |", "|---|---|"]
    for key, text in _STORAGE.items():
        lines.append(f"| `{key}` | {_markdown_cell(text)} |")
    lines += ["", "### Device", "", "| Device | Meaning |", "|---|---|"]
    for key, text in _DEVICES.items():
        lines.append(f"| `{key}` | {_markdown_cell(text)} |")
    lines += [
        "",
        "## Summary",
        "",
        "| Field | Objects | Source | Values | Storage |",
        "|---|---|---|---|---|",
    ]
    for field in fields:
        objects = ", ".join(f"`{item}`" for item in field["objects"])
        sources = ", ".join(f"`{item}`" for item in field["sources"])
        values = f"{field['lanes'] or 'variable'} × `{field['dtype']}`"
        lines.append(
            f"| [`{field['name']}`](#{_anchor(field['name'])}) | {objects} | {sources} | {values} "
            f"| `{field['storage']}` |"
        )
    lines += ["", "## Fields", ""]
    standard_by_newton: dict[str, list[str]] = {}
    for field in fields:
        for linked in field["newton_fields"]:
            standard_by_newton.setdefault(linked, []).append(field["name"])

    def links(names: Sequence[str]) -> str:
        return ", ".join(f"[`{name}`](#{_anchor(name)})" for name in names) or "None."

    for field in fields:
        if "_" in field["name"]:
            related = ("Standard fields", links(standard_by_newton.get(field["name"], [])))
        else:
            related = ("Newton fields", links(field["newton_fields"]))
        layout_rows = []
        if "joint_layouts" in field:
            layout_rows.append(("Joint layouts", _joint_layouts_text(field)))
        rows = [
            ("Objects", ", ".join(f"`{item}`" for item in field["objects"])),
            ("Identity", field["identity"]),
            ("Source", ", ".join(f"`{item}`" for item in field["sources"])),
            related,
            ("Requires", f"`{field['requires']}`"),
            ("Values", _values_text(field)),
            ("Layout", field["layout"]),
            ("Units", field["units"]),
            ("Unit terms", _unit_terms_text(field)),
            *layout_rows,
            ("Frame", field["frame"] or "None."),
            ("Frame tag", f"`{field['frame_tag']}`" if field["frame_tag"] else "None."),
            ("Conversion", field["conversion"] or "None. Values are Newton's."),
            ("Storage", f"`{field['storage']}`"),
            ("Device", f"`{field['device']}`"),
        ]
        lines += [f"({_anchor(field['name'])})=", f"### `{field['name']}`", "", field["summary"], ""]
        lines += ["| Property | Value |", "|---|---|"]
        lines += [f"| {key} | {_markdown_cell(value)} |" for key, value in rows]
        lines.append("")
    return "\n".join(lines)


def render(repository: Path) -> dict[Path, str]:
    """Return every generated artifact, keyed by repository-relative path."""
    fields = load_contract(repository / CONTRACT)
    return {PYTHON_OUTPUT: render_python(fields), MARKDOWN_OUTPUT: render_markdown(fields)}


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def stale_outputs(repository: Path) -> list[Path]:
    """Return generated artifacts whose committed text differs from the contract."""
    stale = []
    for path, text in render(repository).items():
        target = repository / path
        if not target.is_file() or _normalize_newlines(target.read_text(encoding="utf-8")) != text:
            stale.append(path)
    return stale


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the generated artifacts")
    mode.add_argument("--check", action="store_true", help="fail when an artifact is out of date (default)")
    parser.add_argument("--repository", type=Path, default=REPOSITORY, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    try:
        if arguments.write:
            for path, text in render(arguments.repository).items():
                (arguments.repository / path).write_text(text, encoding="utf-8", newline="\n")
                print(f"wrote {path.as_posix()}")
            return 0
        stale = stale_outputs(arguments.repository)
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if stale:
        for path in stale:
            print(f"error: {path.as_posix()} is out of date; run python tools/generate_read_fields.py --write",
                  file=sys.stderr)
        return 1
    print("read-field artifacts are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
