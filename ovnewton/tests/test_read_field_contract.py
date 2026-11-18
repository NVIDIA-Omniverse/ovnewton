# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the read-field contract is valid and its generated artifacts are current."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
_GENERATOR_PATH = REPOSITORY / "tools" / "generate_read_fields.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_read_fields", _GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


def _contract():
    return json.loads((REPOSITORY / generator.CONTRACT).read_text(encoding="utf-8"))


def _field(document, name):
    return next(entry for entry in document["fields"] if entry["name"] == name)


def test_generated_artifacts_are_current():
    assert generator.stale_outputs(REPOSITORY) == [], "run: python tools/generate_read_fields.py --write"


def test_generated_module_compiles():
    fields = generator.load_contract(REPOSITORY / generator.CONTRACT)
    compile(generator.render_python(fields), str(generator.PYTHON_OUTPUT), "exec")


@pytest.mark.parametrize(
    "text",
    [
        'quote " and backslash \\',
        "newline\nand carriage return\r",
        "tab\tand null\x00",
        "line separatorand paragraph separator",
        "non-ASCII × · °",
    ],
)
def test_quote_produces_an_equal_python_literal(text):
    literal = generator._quote(text)
    assert "\n" not in literal and "\r" not in literal
    assert eval(compile(literal, "<literal>", "eval")) == text  # noqa: S307 - generator output only


def test_check_mode_reports_drift(tmp_path):
    for path in (generator.CONTRACT, generator.PYTHON_OUTPUT, generator.MARKDOWN_OUTPUT):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes((REPOSITORY / path).read_bytes())
    assert generator.main(["--check", "--repository", str(tmp_path)]) == 0

    document = _contract()
    _field(document, "mass")["summary"] = "Changed."
    (tmp_path / generator.CONTRACT).write_text(json.dumps(document), encoding="utf-8")
    assert generator.stale_outputs(tmp_path) == [generator.PYTHON_OUTPUT, generator.MARKDOWN_OUTPUT]
    assert generator.main(["--check", "--repository", str(tmp_path)]) == 1

    assert generator.main(["--write", "--repository", str(tmp_path)]) == 0
    assert generator.stale_outputs(tmp_path) == []


def _mutate_native_name_converts(document):
    _field(document, "joint_q")["runtime"]["operation"] = "angular_degrees"
    _field(document, "joint_q")["conversion"] = "Radians to degrees."


def _mutate_native_name_for_other_array(document):
    _field(document, "jointStiffness")["name"] = "joint_stiffness"


def _mutate_unit_suffix(document):
    _field(document, "jointStiffness")["name"] = "jointStiffness.degrees"


def _mutate_conversion_without_text(document):
    _field(document, "jointStiffness")["conversion"] = None


def _mutate_wrong_newton_link(document):
    _field(document, "jointVelocity")["newton_fields"] = ["joint_q"]


def _mutate_missing_newton_link(document):
    _field(document, "jointLimit")["newton_fields"] = ["joint_limit_lower"]


def _mutate_newton_field_with_link(document):
    _field(document, "joint_qd")["newton_fields"] = ["joint_q"]


def _mutate_multiline_summary(document):
    _field(document, "body_q")["summary"] = "Line one.\nLine two."


def _mutate_tab_in_units(document):
    _field(document, "mass")["units"] = "Model\tmass units."


def _mutate_newline_in_component(document):
    _field(document, "body_q")["components"][0] = "p\nx"


def _mutate_lone_surrogate(document):
    _field(document, "mass")["layout"] = "One value per body \ud800."


def _mutate_requirement(document):
    _field(document, "body_q")["requires"] = "model"


def _mutate_components(document):
    _field(document, "body_q")["components"] = ["px", "py", "pz"]


def _mutate_unknown_key(document):
    _field(document, "mass")["unit"] = "kg"


def _mutate_duplicate(document):
    document["fields"].append(copy.deepcopy(_field(document, "mass")))


def _mutate_runtime_source(document):
    _field(document, "mass")["runtime"]["array"] = "body_inv_mass"


def _mutate_unknown_object(document):
    _field(document, "mass")["objects"] = ["particle"]


def _mutate_variable_lanes_without_rule(document):
    del _field(document, "joint_q")["lanes_rule"]


def _mutate_missing_frame_tag(document):
    del _field(document, "position")["frame_tag"]


def _mutate_unknown_frame_tag(document):
    _field(document, "position")["frame_tag"] = "local"


def _mutate_frame_tag_without_frame(document):
    _field(document, "mass")["frame_tag"] = "world"


def _mutate_frame_without_frame_tag(document):
    _field(document, "position")["frame_tag"] = None


def _mutate_missing_unit_terms(document):
    del _field(document, "mass")["unit_terms"]


def _mutate_unit_terms_not_a_list(document):
    _field(document, "mass")["unit_terms"] = {"scope": "all"}


def _mutate_unit_term_not_an_object(document):
    _field(document, "mass")["unit_terms"] = ["kg"]


def _mutate_unit_term_without_scope(document):
    del _field(document, "mass")["unit_terms"][0]["scope"]


def _mutate_unit_term_without_dimensions(document):
    del _field(document, "mass")["unit_terms"][0]["dimensions"]


def _mutate_unit_term_unknown_key(document):
    _field(document, "mass")["unit_terms"][0]["scale"] = 1000


def _mutate_unit_scope_unknown(document):
    _field(document, "mass")["unit_terms"][0]["scope"] = "everything"


def _mutate_unit_component_scope_without_fixed_lanes(document):
    _field(document, "joint_q")["unit_terms"][0]["scope"] = ["q0"]


def _mutate_unit_scope_unknown_component(document):
    _field(document, "body_q")["unit_terms"][0]["scope"] = ["px", "pw"]


def _mutate_unit_dimensions_empty(document):
    _field(document, "mass")["unit_terms"][0]["dimensions"] = {}


def _mutate_unit_dimension_unknown(document):
    _field(document, "mass")["unit_terms"][0]["dimensions"] = {"temperature": 1}


def _mutate_unit_exponent_zero(document):
    _field(document, "mass")["unit_terms"][0]["dimensions"] = {"mass": 0}


def _mutate_unit_exponent_not_an_integer(document):
    _field(document, "mass")["unit_terms"][0]["dimensions"] = {"mass": 1.5}


def _mutate_unit_exponent_is_a_bool(document):
    _field(document, "mass")["unit_terms"][0]["dimensions"] = {"mass": True}


def _mutate_angle_without_angle_unit(document):
    del _field(document, "angularVelocity")["unit_terms"][0]["angle_unit"]


def _mutate_unknown_angle_unit(document):
    _field(document, "angularVelocity")["unit_terms"][0]["angle_unit"] = "turn"


def _mutate_angle_unit_without_angle(document):
    _field(document, "mass")["unit_terms"][0]["angle_unit"] = "radian"


def _mutate_quaternion_term_with_dimensions(document):
    _field(document, "joint_q")["unit_terms"][3]["dimensions"] = {"angle": 1}


def _mutate_layouts_on_non_joint_field(document):
    _field(document, "mass")["joint_layouts"] = {"ball": [{"scope": "all", "count": 1}]}


def _mutate_layouts_empty(document):
    _field(document, "joint_q")["joint_layouts"] = {}


def _mutate_layouts_not_an_object(document):
    _field(document, "joint_q")["joint_layouts"] = [["ball", 4]]


def _mutate_layout_unknown_representation(document):
    _field(document, "joint_q")["joint_layouts"]["fixed"] = [{"scope": "quaternion", "count": 4}]


def _mutate_layout_without_parts(document):
    _field(document, "joint_q")["joint_layouts"]["ball"] = []


def _mutate_layout_part_not_an_object(document):
    _field(document, "joint_q")["joint_layouts"]["ball"] = ["quaternion"]


def _mutate_layout_part_without_scope(document):
    del _field(document, "joint_q")["joint_layouts"]["ball"][0]["scope"]


def _mutate_layout_part_without_count(document):
    del _field(document, "joint_q")["joint_layouts"]["ball"][0]["count"]


def _mutate_layout_part_unknown_key(document):
    _field(document, "joint_q")["joint_layouts"]["ball"][0]["offset"] = 0


def _mutate_layout_scope_without_unit_term(document):
    _field(document, "joint_q")["joint_layouts"]["ball"][0]["scope"] = "all"


def _mutate_layout_configured_outside_d6(document):
    _field(document, "joint_q")["joint_layouts"]["prismatic"][0]["count"] = "configured"


def _mutate_layout_configured_quaternion(document):
    _field(document, "joint_q")["joint_layouts"]["d6"][0] = {"scope": "quaternion", "count": "configured"}


def _mutate_layout_count_zero(document):
    _field(document, "joint_q")["joint_layouts"]["prismatic"][0]["count"] = 0


def _mutate_layout_count_not_an_integer(document):
    _field(document, "joint_q")["joint_layouts"]["prismatic"][0]["count"] = 1.5


def _mutate_layout_count_is_a_bool(document):
    _field(document, "joint_q")["joint_layouts"]["prismatic"][0]["count"] = True


def _mutate_layout_translation_count(document):
    _field(document, "joint_q")["joint_layouts"]["free"][0]["count"] = 4


def _mutate_layout_quaternion_count(document):
    _field(document, "joint_q")["joint_layouts"]["ball"][0]["count"] = 3


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_mutate_native_name_converts, "snake_case name is a Newton field"),
        (_mutate_native_name_for_other_array, "snake_case name is a Newton field"),
        (_mutate_unit_suffix, "not a valid field name"),
        (_mutate_conversion_without_text, "conversion text must be present"),
        (_mutate_wrong_newton_link, r"newton_fields must be \['joint_qd'\]"),
        (_mutate_missing_newton_link, "newton_fields must be"),
        (_mutate_newton_field_with_link, "a Newton field has no newton_fields"),
        (_mutate_multiline_summary, "summary must be one line of text"),
        (_mutate_tab_in_units, "units must be one line of text"),
        (_mutate_newline_in_component, "components must be one line of text"),
        (_mutate_lone_surrogate, "layout must be one line of text"),
        (_mutate_requirement, "requires must be 'state'"),
        (_mutate_components, "components must name each"),
        (_mutate_unknown_key, "unknown keys"),
        (_mutate_duplicate, "duplicate field names"),
        (_mutate_runtime_source, "first source must be the runtime array"),
        (_mutate_unknown_object, "unknown value 'particle'"),
        (_mutate_variable_lanes_without_rule, "lanes_rule"),
        (_mutate_missing_frame_tag, "missing keys"),
        (_mutate_unknown_frame_tag, "frame_tag must be one of"),
        (_mutate_frame_tag_without_frame, "frame_tag must be null when there is no frame"),
        (_mutate_frame_without_frame_tag, "frame_tag must be one of"),
        (_mutate_missing_unit_terms, "missing keys"),
        (_mutate_unit_terms_not_a_list, "unit_terms must be a list"),
        (_mutate_unit_term_not_an_object, "each unit term must be an object"),
        (_mutate_unit_term_without_scope, "each unit term needs scope"),
        (_mutate_unit_term_without_dimensions, "each unit term needs dimensions"),
        (_mutate_unit_term_unknown_key, "unknown unit term key"),
        (_mutate_unit_scope_unknown, "unknown unit scope"),
        (_mutate_unit_component_scope_without_fixed_lanes, "a component scope needs fixed lanes"),
        (_mutate_unit_scope_unknown_component, "unit scope names unknown component"),
        (_mutate_unit_dimensions_empty, "dimensions must be a non-empty object"),
        (_mutate_unit_dimension_unknown, "unknown dimension"),
        (_mutate_unit_exponent_zero, "exponents must be non-zero integers"),
        (_mutate_unit_exponent_not_an_integer, "exponents must be non-zero integers"),
        (_mutate_unit_exponent_is_a_bool, "exponents must be non-zero integers"),
        (_mutate_angle_without_angle_unit, "angle_unit must be one of"),
        (_mutate_unknown_angle_unit, "angle_unit must be one of"),
        (_mutate_angle_unit_without_angle, "angle_unit is only for dimensions that include angle"),
        (_mutate_quaternion_term_with_dimensions, "a quaternion term has no dimensions"),
        (_mutate_layouts_on_non_joint_field, "joint_layouts are only for joint fields"),
        (_mutate_layouts_empty, "joint_layouts must be a non-empty object"),
        (_mutate_layouts_not_an_object, "joint_layouts must be a non-empty object"),
        (_mutate_layout_unknown_representation, "unknown joint representation"),
        (_mutate_layout_without_parts, "must be a non-empty list"),
        (_mutate_layout_part_not_an_object, "each layout part must be an object"),
        (_mutate_layout_part_without_scope, "each layout part needs scope"),
        (_mutate_layout_part_without_count, "each layout part needs count"),
        (_mutate_layout_part_unknown_key, "unknown layout part key"),
        (_mutate_layout_scope_without_unit_term, "with no unit term"),
        (_mutate_layout_configured_outside_d6, "only d6 scalar axes can have the count 'configured'"),
        (_mutate_layout_configured_quaternion, "only d6 scalar axes can have the count 'configured'"),
        (_mutate_layout_count_zero, "layout counts must be positive integers"),
        (_mutate_layout_count_not_an_integer, "layout counts must be positive integers"),
        (_mutate_layout_count_is_a_bool, "layout counts must be positive integers"),
        (_mutate_layout_translation_count, "a translation has 3 values"),
        (_mutate_layout_quaternion_count, "a quaternion has 4 values"),
    ],
)
def test_invalid_contracts_are_rejected(tmp_path, mutate, message):
    document = _contract()
    mutate(document)
    path = tmp_path / "read_fields.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(generator.ContractError, match=message):
        generator.load_contract(path)
