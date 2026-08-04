# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model-diff must catch what the old position-only summary missed — body
ORIENTATION and full inertia."""
import newton
import numpy as np

from . import diff_harness as dh


def _arrs(pos, quat, inertia_diag):
    n = len(pos)
    return {
        "body_label": np.array(["/b%d" % i for i in range(n)], dtype=object),
        "body_mass": np.ones(n, dtype=np.float32),
        "body_com": np.zeros((n, 3), dtype=np.float32),
        "body_q": np.hstack([np.asarray(pos, np.float32), np.asarray(quat, np.float32)]),
        "body_inertia": np.array([np.diag(d) for d in inertia_diag], dtype=np.float32),
        "joint_label": np.array([], dtype=object),
        "n_art": 0,
    }


def test_compare_catches_orientation_difference():
    ref = _arrs([[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]])
    ours = _arrs([[0, 0, 0]], [[0, 0, 1, 0]], [[1, 1, 1]])     # same pos, 180° about Z
    assert any("orient" in m.lower() for m in dh.compare_models(ref, ours, check_joints=False))


def test_compare_catches_offdiagonal_inertia_difference():
    ref = _arrs([[0, 0, 0]], [[0, 0, 0, 1]], [[1, 2, 3]])
    ours = _arrs([[0, 0, 0]], [[0, 0, 0, 1]], [[1, 2, 3]])
    ours["body_inertia"][0, 0, 1] = 0.5                        # off-diagonal differs
    assert any("inertia" in m for m in dh.compare_models(ref, ours, check_joints=False))


def _arrs_with_joint(parent_idx, child_idx, x_p, n_bodies=2):
    a = _arrs([[0, 0, 0]] * n_bodies, [[0, 0, 0, 1]] * n_bodies, [[1, 1, 1]] * n_bodies)
    a["joint_parent"] = np.array([parent_idx], dtype=np.int32)
    a["joint_child"] = np.array([child_idx], dtype=np.int32)
    a["joint_type"] = np.array([0], dtype=np.int32)
    a["joint_X_p"] = np.array([x_p], dtype=np.float32)        # (1, 7) transform
    a["joint_X_c"] = np.zeros((1, 7), dtype=np.float32)
    a["joint_label"] = np.array(["/joint_0"], dtype=object)
    return a


def test_compare_catches_joint_frame_difference_by_edge():
    # same edge (body 0 -> body 1), different parent anchor frame
    ref = _arrs_with_joint(0, 1, [0, 0, 0, 0, 0, 0, 1])
    ours = _arrs_with_joint(0, 1, [0.5, 0, 0, 0, 0, 0, 1])
    assert any("joint_X_p" in m for m in dh.compare_models(ref, ours, check_joints=True))


def test_compare_catches_joint_count_difference():
    ref = _arrs_with_joint(0, 1, [0, 0, 0, 0, 0, 0, 1])
    ours = _arrs([[0, 0, 0]] * 2, [[0, 0, 0, 1]] * 2, [[1, 1, 1]] * 2)   # 0 joints
    ours["joint_parent"] = np.array([], dtype=np.int32)
    ours["joint_child"] = np.array([], dtype=np.int32)
    assert any("joint count differs" in m for m in dh.compare_models(ref, ours, check_joints=True))


def _shape_arrays():
    arrays = _arrs([[0, 0, 0], [0, 0, 0]], [[0, 0, 0, 1]] * 2, [[1, 1, 1]] * 2)
    arrays.update(
        shape_label=np.array(["/A", "/B"], dtype=object),
        shape_body=np.array([0, 1], dtype=np.int32),
        shape_transform=np.array([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=np.float32),
        shape_type=np.array([0, 0], dtype=np.int32),
        shape_scale=np.ones((2, 3), dtype=np.float32),
        shape_flags=np.array([1, 1], dtype=np.int32),
        shape_is_solid=np.array([True, True]),
        _shape_sdf_index=np.array([-1, -1], dtype=np.int32),
        shape_collision_group=np.array([1, 1], dtype=np.int32),
        shape_collision_filter_pairs=[],
        shape_mesh=[None, None],
    )
    return arrays


def test_compare_catches_shape_geometry_difference():
    ref = _shape_arrays()
    ours = _shape_arrays()
    ours["shape_scale"][1, 0] = 2.0
    assert any("shape_scale" in m for m in dh.compare_models(ref, ours, check_joints=False, check_shapes=True))


def test_compare_catches_shape_flag_difference():
    ref = _shape_arrays()
    ours = _shape_arrays()
    ours["shape_flags"][0] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert any("shape_flags" in m for m in dh.compare_models(ref, ours, check_joints=False, check_shapes=True))


def test_compare_ignores_render_visibility_flag():
    ref = _shape_arrays()
    ours = _shape_arrays()
    ours["shape_flags"][0] &= ~int(newton.ShapeFlags.VISIBLE)
    assert not dh.compare_models(ref, ours, check_joints=False, check_shapes=True)


def test_compare_catches_shape_sdf_presence_difference():
    ref = _shape_arrays()
    ours = _shape_arrays()
    ref["_shape_sdf_index"][1] = 0
    assert any("has SDF" in m for m in dh.compare_models(ref, ours, check_joints=False, check_shapes=True))


def test_compare_ignores_shape_sdf_allocation_order():
    ref = _shape_arrays()
    ours = _shape_arrays()
    ref["_shape_sdf_index"][:] = (0, 1)
    ours["_shape_sdf_index"][:] = (1, 0)
    assert not dh.compare_models(ref, ours, check_joints=False, check_shapes=True)


def test_compare_catches_shape_filter_pair_difference():
    ref = _shape_arrays()
    ours = _shape_arrays()
    ref["shape_collision_filter_pairs"] = [(0, 1)]
    assert any("filter pairs" in m for m in dh.compare_models(ref, ours, check_joints=False, check_shapes=True))


def test_compare_catches_collision_group_behavior_difference():
    ref = _shape_arrays()
    ours = _shape_arrays()
    ours["shape_collision_group"][1] = 2
    assert any("collision groups" in m for m in dh.compare_models(ref, ours, check_joints=False, check_shapes=True))


def test_compare_catches_orphan_joint_membership_difference():
    ref = _arrs_with_joint(0, 1, [0, 0, 0, 0, 0, 0, 1])
    ours = _arrs_with_joint(0, 1, [0, 0, 0, 0, 0, 0, 1])
    ref["joint_articulation"] = np.array([-1], dtype=np.int32)
    ours["joint_articulation"] = np.array([0], dtype=np.int32)
    assert any(
        "joint_articulation" in m
        for m in dh.compare_models(ref, ours, check_joints=True, check_articulations=True)
    )
