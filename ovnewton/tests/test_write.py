# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip the body-pose write-back against the validated read.

Writes a stepped/perturbed set of body transforms to ``omni:xform`` through the
production runtime transport, reads them back at the written ordinal, and
decodes with the same ``_decode_pose`` the reader uses.
Since the decode is independently validated against ``add_usd`` (test_diff),
``encode → write → read → decode == identity`` pins the write transport.
"""

import numpy as np
import ovstage
import pytest
from ovstage import PopulationDomain, population

import ovnewton
from ovnewton._src import _build, _runtime, _stage
from ovnewton._src._errors import OvstageContractError
from ovnewton.examples import get_asset

from .runtime_helpers import lanes_tensor, read_body_state

CARTPOLE = get_asset("scene_cartpole.usda")


def test_scaled_body_pose_write_roundtrip(tmp_path):
    asset = tmp_path / "scaled-body.usda"
    asset.write_text(
        """#usda 1.0
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    double3 xformOp:scale = (-2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
""",
        encoding="utf-8",
    )

    with ovstage.Stage("ovstage-scaled-body") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, str(asset), ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        binding = ovnewton.attach_ovstage(stage)
        binding.update_to_ovstage(binding.model.state(), ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        with _stage._path_list_query(stage, pd, binding.model.body_label) as query:
            matrices = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)

    _, _, scale = _build._decode_pose(np.stack([matrices[0]]))
    np.testing.assert_allclose(scale, [[-2.0, -3.0, -4.0]])


def test_runtime_tensor_contract_rejects_wrong_type_lanes_and_rank():
    import warp as wp

    tensor = lanes_tensor(np.zeros((1, 3), dtype=np.float32), lanes=3)
    with pytest.raises(OvstageContractError, match="expected float64x3"):
        _runtime._tensor_source_view(tensor, dtype=wp.float64, lanes=3)
    with pytest.raises(OvstageContractError, match="expected float32x1"):
        _runtime._tensor_source_view(tensor, dtype=wp.float32, lanes=1)

    tensor.ndim = 2
    with pytest.raises(OvstageContractError, match="one-dimensional"):
        _runtime._tensor_source_view(tensor, dtype=wp.float32, lanes=3)


def test_wait_operations_drains_every_operation_after_failure():
    waited = []

    class Operation:
        def __init__(self, index, fails=False):
            self.index = index
            self.fails = fails

        def wait(self):
            waited.append(self.index)
            if self.fails:
                raise RuntimeError(f"operation {self.index} failed")

    with pytest.raises(RuntimeError, match="operation 1 failed"):
        _runtime._wait_operations([Operation(0), Operation(1, fails=True), Operation(2)])
    assert waited == [0, 1, 2]


def _quat_mul(a, b):
    """Hamilton product of xyzw quaternions, broadcast over the leading axis."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=1)


def test_cartpole_pose_write_roundtrip():
    import warp as wp

    with ovstage.Stage("ovstage-write") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        model = _build.build_model(stage, pd, ordinal=1)
        body_paths = model.body_label
        with _stage._path_list_query(stage, pd, body_paths) as query:
            initial = _stage.read_fixed(stage, pd, query, "omni:fabric:worldMatrix", ordinal=1)
        _, _, expected_scale = _build._decode_pose(
            np.stack([initial[i] for i in range(model.body_count)])
        )

        # Target poses: shift translation and apply a fixed 30deg-about-Y rotation
        # to every body, so both the translation and rotation encode are exercised
        # (beyond the rotations cartpole's poles already carry).
        bq = model.body_q.numpy().astype(np.float64)
        n = len(body_paths)
        target = bq.copy()
        target[:, :3] += np.array([0.123, -0.456, 0.789])
        s = np.sin(np.deg2rad(15.0))
        ry = np.tile([0.0, s, 0.0, np.cos(np.deg2rad(15.0))], (n, 1))
        target[:, 3:7] = _quat_mul(ry, bq[:, 3:7])

        # Body-state write through the production Warp encoder.
        state = model.state()
        state.body_q.assign(wp.array(target.astype(np.float32), dtype=wp.transformf, device=model.device))
        ovnewton.StageBinding(stage, model).update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        # Read the canonical local transform back at the written ordinal.
        plist = pd.create_path_list_from_strings(list(body_paths))
        query = stage.query_from_path_list(plist)
        xform = _stage.read_fixed(stage, pd, query, "omni:xform", ordinal=2)
        reset_xform_stack = _stage.read_fixed(stage, pd, query, "omni:resetXformStack", ordinal=2)
        stage.release_query(query).wait()
        pd.destroy_path_list(plist)

    ident = np.eye(4).reshape(16)
    mats = np.stack([np.asarray(xform.get(i, ident), dtype=np.float64).reshape(16) for i in range(n)])
    trans, quat, scale = _build._decode_pose(mats)

    assert np.allclose(trans, target[:, :3], atol=1e-5), \
        f"translation round-trip mismatch:\n  got ={trans}\n  want={target[:, :3]}"
    # Quaternion is sign-ambiguous; compare via |dot|.
    dots = np.abs(np.sum(quat * target[:, 3:7], axis=1))
    assert np.all(dots > 1.0 - 1e-5), f"rotation round-trip mismatch, |dot|={dots}"
    assert np.allclose(scale, expected_scale, atol=1e-5), \
        f"scale round-trip mismatch:\n  got ={scale}\n  want={expected_scale}"
    assert all(bool(reset_xform_stack[i]) for i in range(n))


def test_cartpole_velocity_write_roundtrip():
    """Write a world-frame CoM twist and convert only its angular units."""
    import warp as wp

    with ovstage.Stage("ovstage-vel") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        model = _build.build_model(stage, pd, ordinal=1)
        body_paths = model.body_label
        n = len(body_paths)

        # Deterministic non-trivial twist (linear, angular rad/s) per body.
        rng = np.arange(n * 6, dtype=np.float64).reshape(n, 6) * 0.1 - 1.0
        model.body_com.assign(
            wp.array(
                np.tile((0.3, -0.2, 0.1), (n, 1)),
                dtype=wp.vec3,
                device=model.device,
            )
        )
        state = model.state()
        state.body_qd.assign(wp.array(rng.astype(np.float32), dtype=wp.spatial_vectorf, device=model.device))

        ovnewton.StageBinding(stage, model).update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        plist = pd.create_path_list_from_strings(list(body_paths))
        query = stage.query_from_path_list(plist)
        lin = _stage.read_fixed(stage, pd, query, "physics:velocity", ordinal=2)
        ang = _stage.read_fixed(stage, pd, query, "physics:angularVelocity", ordinal=2)
        stage.release_query(query).wait()
        pd.destroy_path_list(plist)

    linear = np.stack([np.asarray(lin[i], dtype=np.float64).reshape(3) for i in range(n)])
    ang_deg = np.stack([np.asarray(ang[i], dtype=np.float64).reshape(3) for i in range(n)])
    omega = np.deg2rad(ang_deg)
    recovered = np.hstack([linear, omega])

    assert np.allclose(recovered, rng, atol=1e-4), \
        f"velocity round-trip mismatch:\n  got ={recovered}\n  want={rng}"


def test_cartpole_state_sync_roundtrip():
    """Round-trip body pose and world-frame CoM velocity."""
    import warp as wp

    with ovstage.Stage("ovstage-sync") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        model = _build.build_model(stage, pd, ordinal=1)
        body_paths = model.body_label
        n = len(body_paths)

        src = model.state()
        bq = model.body_q.numpy().astype(np.float64)
        bq[:, :3] += np.array([0.2, -0.1, 0.3])
        s = np.sin(np.deg2rad(10.0))
        ry = np.tile([0.0, s, 0.0, np.cos(np.deg2rad(10.0))], (n, 1))
        bq[:, 3:7] = _quat_mul(ry, bq[:, 3:7])
        twist = (np.arange(n * 6, dtype=np.float64).reshape(n, 6) * 0.05 - 0.3)
        src.body_q.assign(wp.array(bq.astype(np.float32), dtype=wp.transformf, device=model.device))
        src.body_qd.assign(wp.array(twist.astype(np.float32), dtype=wp.spatial_vectorf, device=model.device))

        ovnewton.StageBinding(stage, model).update_to_ovstage(src, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        dst = model.state()  # fresh (initial) state, then resync from ovstage
        read_body_state(stage, pd, body_paths, ordinal=2, state=dst, model=model)

        got_q = dst.body_q.numpy().astype(np.float64)
        got_qd = dst.body_qd.numpy().astype(np.float64)

    assert np.allclose(got_q[:, :3], bq[:, :3], atol=1e-4), \
        f"pose translation sync mismatch:\n  got ={got_q[:, :3]}\n  want={bq[:, :3]}"
    dots = np.abs(np.sum(got_q[:, 3:7] * bq[:, 3:7], axis=1))
    assert np.all(dots > 1.0 - 1e-4), f"pose rotation sync mismatch, |dot|={dots}"
    assert np.allclose(got_qd, twist, atol=1e-3), \
        f"velocity sync mismatch:\n  got ={got_qd}\n  want={twist}"


def test_cartpole_joint_state_write_roundtrip():
    """Write parsed joint coordinates by their materialized prim paths and read
    them back. Cartpole exercises both linear and angular instances."""
    import newton
    import warp as wp
    from ovstage import Filter, FilterOp, Predicate

    deg = 57.29577951308232
    with ovstage.Stage("ovstage-jstate") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        binding = ovnewton.attach_ovstage(stage)
        model = binding.model
        joint_labels = list(model.joint_label)

        q_start = model.joint_q_start.numpy()
        nq = model.joint_q.numpy().shape[0]
        jq = (np.arange(nq, dtype=np.float64) * 0.05 + 0.1).astype(np.float32)
        state = model.state()
        state.joint_q.assign(wp.array(jq, dtype=wp.float32, device=model.device))

        binding.update_to_ovstage(state, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        read_back = {}
        for prim_type, inst in [("PhysicsRevoluteJoint", "angular"),
                                ("PhysicsPrismaticJoint", "linear")]:
            with stage.query(filter=Filter([Predicate("usd-prim-type", FilterOp.IN, [prim_type])])) as q:
                count = _stage._count(stage, q)
                paths = (
                    _stage._query_paths(stage, pd, q, ordinal=2, expected_count=count)
                    if count
                    else []
                )
            if paths:
                with _stage._path_list_query(stage, pd, paths) as path_query:
                    pos = _stage.read_fixed(
                        stage, pd, path_query, f"state:{inst}:physics:position", ordinal=2
                    )
                read_back.update(
                    (path, float(np.asarray(pos[i]).reshape(-1)[0])) for i, path in enumerate(paths)
                )

    assert read_back, "no single-DOF joints found to validate"
    type_scale = {int(newton.JointType.REVOLUTE): deg,
                  int(newton.JointType.PRISMATIC): 1.0}
    for jid, label in enumerate(joint_labels):
        scale = type_scale.get(int(model.joint_type.numpy()[jid]))
        if scale is not None:
            expected = jq[int(q_start[jid])] * scale
            assert np.isclose(read_back[label], expected, atol=1e-4), \
                f"{label} joint-state mismatch: got={read_back[label]}, want={expected}"
