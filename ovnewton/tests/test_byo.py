# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bring-your-own-model: bind a labeled Newton model to an ovstage and sync
without parsing the model out of ovstage."""
import numpy as np
import ovstage
import pytest
import warp as wp
from ovstage import PopulationDomain, population

import ovnewton
from ovnewton._src import _build, _parse, _stage
from ovnewton.examples import get_asset

CARTPOLE = get_asset("scene_cartpole.usda")


def _byo_model_for(stage, pd):
    """Build a tiny model whose body_label are cartpole's body prim paths, with
    poses read from the populated stage — stands in for a Newton add_usd model
    built from the same USD (add_usd can't run in-process alongside ovstage's USD)."""
    import newton
    hierarchy = _parse.read_hierarchy(stage, pd, ordinal=1)
    _, body_paths = _parse.read_bodies(stage, pd, ordinal=1, hierarchy=hierarchy)
    plist = pd.create_path_list_from_strings(list(body_paths))
    q = stage.query_from_path_list(plist)
    wm = _stage.read_fixed(stage, pd, q, "omni:fabric:worldMatrix", 1)
    stage.release_query(q).wait(); pd.destroy_path_list(plist)
    ident = np.eye(4).reshape(16)
    mats = np.stack([np.asarray(wm.get(i, ident), np.float64).reshape(16) for i in range(len(body_paths))])
    trans, quats, _ = _build._decode_pose(mats)
    b = newton.ModelBuilder()
    for i, p in enumerate(body_paths):
        b.add_body(xform=wp.transform(wp.vec3(*trans[i]), wp.quat(*quats[i])), label=p)
    return b.finalize()


def test_byo_pose_velocity_roundtrip():
    with ovstage.Stage("byo") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        model = _byo_model_for(stage, pd)

        binding = ovnewton.attach_ovstage(stage, model=model)
        assert not hasattr(binding, "body_labels")
        assert not hasattr(binding, "stage")
        assert not hasattr(binding, "pd")

        s = model.state()
        bq = model.body_q.numpy().astype(np.float64)
        bq[:, :3] += np.array([0.1, 0.2, 0.3])
        s.body_q.assign(wp.array(bq.astype(np.float32), dtype=wp.transformf, device=model.device))
        binding.update_to_ovstage(s, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()

        dst = model.state()
        binding.update_from_ovstage(dst, control=model.control())
        got = dst.body_q.numpy().astype(np.float64)
    assert np.allclose(got[:, :3], bq[:, :3], atol=1e-4)


def test_byo_uses_requested_population_ordinal():
    import newton

    with ovstage.Stage("byo-nondefault-ordinal") as stage:
        population.open_usd(stage, CARTPOLE, ordinal=4, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=4).wait()
        builder = newton.ModelBuilder()
        builder.add_body(label="/cartPole/cart")
        binding = ovnewton.attach_ovstage(stage, model=builder.finalize(), ordinal=4)

    assert binding.model.body_label == ["/cartPole/cart"]


def test_byo_rejects_partially_unresolved_body_labels():
    """A partial path-list match must fail at bind time; otherwise later batched
    writes shift state rows onto the wrong stage prim."""
    import newton

    with ovstage.Stage("byo-invalid") as stage, ovstage.PathDictionary(stage):
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        builder = newton.ModelBuilder()
        builder.add_body(label="/cartPole/doesNotExist")
        builder.add_body(label="/cartPole/cart")
        model = builder.finalize()

        with pytest.raises(ValueError, match="did not all resolve"):
            ovnewton.attach_ovstage(stage, model=model)


def test_byo_rejects_incomplete_body_label_map():
    """The label list is a per-body mapping, so cardinality is an invariant."""
    import newton

    builder = newton.ModelBuilder()
    builder.add_body(label="/body/one")
    builder.add_body(label="/body/two")
    model = builder.finalize()
    model.body_label = model.body_label[:1]

    with ovstage.Stage("byo-short") as stage:
        with pytest.raises(ValueError, match="expected 2 body labels, got 1"):
            ovnewton.attach_ovstage(stage, model=model)


def test_byo_rejects_unresolved_absolute_joint_label():
    with ovstage.Stage("byo-invalid-joint") as stage, ovstage.PathDictionary(stage):
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        import newton

        builder = newton.ModelBuilder()
        body = builder.add_link(label="/cartPole/rail")
        joint = builder.add_joint_revolute(parent=-1, child=body, label="/cartPole/doesNotExist")
        builder.add_articulation([joint])
        model = builder.finalize()

        with pytest.raises(ValueError, match="joint labels did not all resolve"):
            ovnewton.attach_ovstage(stage, model=model)


def test_byo_joint_state_by_path():
    """Write a single revolute joint's coordinate to the stage addressed by the
    joint's prim path (the BYO path), then read it back. The model is built
    pxr-free; its revolute joint is labelled with a real cartpole joint prim path
    (stands in for an add_usd model's joint_label)."""
    import newton

    REV_JOINT = "/cartPole/cartPoleJoint"  # a real, resolvable cartpole revolute joint prim
    with ovstage.Stage("byo-j") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        # NOTE: add_body + add_joint_revolute(parent=-1) yields TWO joints — an implicit
        # FREE joint for the body AND the revolute — so index the revolute explicitly
        # (don't assume joint 0 or joint_q length 1). The FREE joint is skipped by the
        # runtime joint-channel mapping.
        b = newton.ModelBuilder()
        body = b.add_body(label="/cartPole/rail")
        b.add_joint_revolute(parent=-1, child=body, axis=(1, 0, 0), label=REV_JOINT)
        m = b.finalize()
        jtype = m.joint_type.numpy()
        rev_j = next(j for j in range(len(jtype)) if int(jtype[j]) == int(newton.JointType.REVOLUTE))
        qs = m.joint_q_start.numpy()
        s = m.state()
        jq = s.joint_q.numpy(); jq[int(qs[rev_j])] = 0.5
        s.joint_q.assign(wp.array(jq.astype(np.float32), dtype=wp.float32, device=m.device))
        binding = ovnewton.attach_ovstage(stage, model=m)
        binding.update_to_ovstage(s, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        plist = pd.create_path_list_from_strings([REV_JOINT])
        q = stage.query_from_path_list(plist)
        got = _stage.read_fixed(stage, pd, q, "state:angular:physics:position", ordinal=2)
        stage.release_query(q).wait(); pd.destroy_path_list(plist)
    val = float(np.asarray(got[0]).reshape(-1)[0])
    assert abs(val - np.rad2deg(0.5)) < 1e-3   # rad -> deg


def _read_angular_position(stage, pd, path, ordinal):
    """Read one joint prim's ``state:angular:physics:position`` scalar (deg)."""
    plist = pd.create_path_list_from_strings([path])
    q = stage.query_from_path_list(plist)
    pos = _stage.read_fixed(stage, pd, q, "state:angular:physics:position", ordinal)
    stage.release_query(q).wait(); pd.destroy_path_list(plist)
    return float(np.asarray(pos[0]).reshape(-1)[0])


def test_byo_joint_state_by_path_batches_same_instance():
    """Two revolute joints addressed by distinct prim paths write in ONE batched
    path-list query — each coordinate must land on its OWN prim (row i -> label i).
    Distinct values guard against the batched column misaligning (swapping rows)."""
    import newton

    REV1, REV2 = "/cartPole/cartPoleJoint", "/cartPole/polePoleJoint"
    with ovstage.Stage("byo-jbatch") as stage, ovstage.PathDictionary(stage) as pd:
        population.open_usd(stage, CARTPOLE, ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        b = newton.ModelBuilder()
        b.add_joint_revolute(
            parent=-1,
            child=b.add_body(label="/cartPole/rail"),
            axis=(1, 0, 0),
            label=REV1,
        )
        b.add_joint_revolute(
            parent=-1,
            child=b.add_body(label="/cartPole/cart"),
            axis=(1, 0, 0),
            label=REV2,
        )
        m = b.finalize()
        jtype = m.joint_type.numpy()
        revs = [j for j in range(len(jtype)) if int(jtype[j]) == int(newton.JointType.REVOLUTE)]
        assert len(revs) == 2
        qs = m.joint_q_start.numpy()
        s = m.state()
        jq = s.joint_q.numpy(); jq[int(qs[revs[0]])] = 0.5; jq[int(qs[revs[1]])] = -1.25
        s.joint_q.assign(wp.array(jq.astype(np.float32), dtype=wp.float32, device=m.device))
        binding = ovnewton.attach_ovstage(stage, model=m)
        binding.update_to_ovstage(s, ordinal=2)
        stage.advance_write_floor(ordinal=2).wait()
        got1 = _read_angular_position(stage, pd, REV1, 2)
        got2 = _read_angular_position(stage, pd, REV2, 2)
    assert abs(got1 - np.rad2deg(0.5)) < 1e-3, (got1, got2)
    assert abs(got2 - np.rad2deg(-1.25)) < 1e-3, (got1, got2)
