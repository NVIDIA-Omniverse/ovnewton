# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np

DT, SUBSTEPS, ITERS = 1.0 / 60.0, 2, 8

# StageBinding imports authored physics only. These options make add_usd use
# the same surface while retaining its other defaults.
ADD_USD_PHYSICS_PROFILE = {
    "apply_up_axis_from_stage": True,
    "load_visual_shapes": False,
}

_BODY_ARRAYS = ["body_mass", "body_inertia", "body_com", "body_q", "body_flags"]
_SHAPE_ARRAYS = [
    "shape_body",
    "shape_transform",
    "shape_type",
    "shape_scale",
    "shape_flags",
    "shape_is_solid",
    "shape_margin",
    "shape_gap",
    "shape_collision_group",
    "shape_material_ke",
    "shape_material_kd",
    "shape_material_kf",
    "shape_material_ka",
    "shape_material_mu",
    "shape_material_restitution",
    "shape_material_mu_torsional",
    "shape_material_mu_rolling",
    "shape_material_kh",
    "_shape_sdf_index",
]
_JOINT_ARRAYS = ["joint_type", "joint_parent", "joint_child", "joint_X_p", "joint_X_c", "joint_articulation"]
_JOINT_DOF_ARRAYS = [
    "joint_axis",
    "joint_target_qd",
    "joint_target_mode",
    "joint_target_ke",
    "joint_target_kd",
    "joint_damping",
    "joint_armature",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_friction",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_limit_ke",
    "joint_limit_kd",
]
_JOINT_COORD_ARRAYS = ["joint_target_q"]
_JOINT_STATE = ["joint_q", "joint_qd", "joint_q_start", "joint_qd_start", "joint_target_q_start"]
_MIMIC_ARRAYS = [
    "joint_mimic_joint",
    "joint_mimic_coeffs",
    "constraint_mimic_joint0",
    "constraint_mimic_joint1",
    "constraint_mimic_coef0",
    "constraint_mimic_coef1",
    "constraint_mimic_enabled",
]


def _model_arrays(model):
    """Correctness-relevant model arrays as numpy + labels for cross-model alignment."""
    out = {}
    for name in (
        _BODY_ARRAYS
        + _SHAPE_ARRAYS
        + _JOINT_ARRAYS
        + _JOINT_DOF_ARRAYS
        + _JOINT_COORD_ARRAYS
        + _JOINT_STATE
        + _MIMIC_ARRAYS
    ):
        a = getattr(model, name, None)
        if a is not None and hasattr(a, "numpy"):
            out[name] = np.asarray(a.numpy())
    for name in ("body_label", "shape_label", "joint_label", "constraint_mimic_label"):
        out[name] = np.asarray(list(getattr(model, name, [])), dtype=object)
    out["shape_collision_filter_pairs"] = sorted(getattr(model, "shape_collision_filter_pairs", ()))
    out["shape_mesh"] = [
        (
            np.asarray(source.vertices),
            np.asarray(source.indices),
        )
        if source is not None and hasattr(source, "vertices") and hasattr(source, "indices")
        else None
        for source in getattr(model, "shape_source", ())
    ]
    out["shape_mesh_maxhullvert"] = np.asarray(
        [getattr(source, "maxhullvert", -1) for source in getattr(model, "shape_source", ())],
        dtype=np.int32,
    )
    out["n_art"] = int(getattr(model, "articulation_count", 0))
    out["up_axis"] = int(getattr(model, "up_axis", -1))
    grav = getattr(model, "gravity", None)
    if grav is not None and hasattr(grav, "numpy"):
        out["gravity"] = np.asarray(grav.numpy())
    return out


def mimic_signatures(model):
    """Label-aligned mimic semantics across Newton's old and new storage."""
    labels = model["joint_label"]
    result = [
        (labels[int(follower)], labels[int(leader)], float(offset), float(multiplier), bool(enabled))
        for follower, leader, offset, multiplier, enabled in zip(
            model["constraint_mimic_joint0"],
            model["constraint_mimic_joint1"],
            model["constraint_mimic_coef0"],
            model["constraint_mimic_coef1"],
            model["constraint_mimic_enabled"],
            strict=True,
        )
    ]
    if "joint_mimic_joint" in model:
        for follower, leader in enumerate(model["joint_mimic_joint"]):
            if leader >= 0:
                offset, multiplier = model["joint_mimic_coeffs"][follower]
                result.append((labels[follower], labels[int(leader)], float(offset), float(multiplier), True))
    return result


def _quat_close(a, b, atol):
    return bool(np.all(np.abs(np.abs(np.sum(a * b, axis=-1)) - 1.0) <= atol))  # sign-insensitive


def _mesh_triangles(mesh):
    """Canonical oriented triangles, retaining duplicates but ignoring indexing."""
    vertices, indices = mesh
    triangles = np.asarray(vertices)[np.asarray(indices).reshape(-1, 3)]
    if not len(triangles):
        return np.empty((0, 9))
    # Only cyclic rotations preserve winding. Pick the lexicographically first
    # rotation, then sort faces so expansion and face order are immaterial.
    rotations = np.stack([np.roll(triangles, -i, axis=1).reshape(-1, 9) for i in range(3)], axis=1)
    first = np.lexsort(rotations[:, :, ::-1].transpose(2, 0, 1), axis=-1)[:, 0]
    faces = rotations[np.arange(len(rotations)), first]
    return faces[np.lexsort(faces[:, ::-1].T)]


def _meshes_close(ref, ours, atol, rtol):
    ref_faces, our_faces = _mesh_triangles(ref), _mesh_triangles(ours)
    if ref_faces.shape != our_faces.shape:
        return False
    if np.allclose(ref_faces, our_faces, atol=atol, rtol=rtol):
        return True
    if not np.isfinite(ref_faces).all() or not np.isfinite(our_faces).all():
        return False

    # Tiny perturbations can change the canonical corner or face order. Match
    # nearby oriented triangles within tolerance, one-to-one to keep duplicates.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.spatial import cKDTree

    ref_triangles = ref_faces.reshape(-1, 3, 3)
    our_triangles = our_faces.reshape(-1, 3, 3)
    radius = np.sqrt(3.0) * (atol + rtol * np.abs(our_faces).max())
    candidates = cKDTree(our_triangles.mean(axis=1)).query_ball_point(ref_triangles.mean(axis=1), radius)
    rows, columns = [], []
    for index, neighbors in enumerate(candidates):
        triangles = our_triangles[neighbors]
        matches = np.zeros(len(neighbors), dtype=bool)
        for rotation in range(3):
            matches |= np.isclose(
                ref_triangles[index], np.roll(triangles, rotation, axis=1), atol=atol, rtol=rtol
            ).all(axis=(1, 2))
        columns.extend(np.asarray(neighbors, dtype=int)[matches].tolist())
        rows.extend([index] * int(matches.sum()))
    graph = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(ref_faces), len(our_faces)))
    return bool(np.all(maximum_bipartite_matching(graph) >= 0))


def compare_models(
    ref,
    ours,
    *,
    atol=1e-4,
    rtol=1e-4,
    check_joints=True,
    check_articulations=False,
    check_shapes=False,
):
    mm = []
    rb = {p: i for i, p in enumerate(ref["body_label"])}
    ob = {p: i for i, p in enumerate(ours["body_label"])}
    common = [p for p in ref["body_label"] if p in ob]
    if len(common) != len(rb) or len(common) != len(ob):
        mm.append("body set differs: ref=%d ours=%d common=%d" % (len(rb), len(ob), len(common)))
    for p in common:
        i, j = rb[p], ob[p]
        for arr in ("body_mass", "body_com", "body_inertia", "body_flags"):
            if arr in ref and arr in ours and not np.allclose(ref[arr][i], ours[arr][j], atol=atol, rtol=rtol):
                mm.append("%s.%s: ref %s vs ours %s" % (p, arr, ref[arr][i], ours[arr][j]))
        rq, oq = ref["body_q"][i], ours["body_q"][j]
        if not np.allclose(rq[:3], oq[:3], atol=atol):
            mm.append("%s.pos: ref %s vs ours %s" % (p, rq[:3], oq[:3]))
        if not _quat_close(rq[3:7], oq[3:7], atol):
            mm.append("%s.orient: ref %s vs ours %s" % (p, rq[3:7], oq[3:7]))
    if check_shapes:
        rs = {p: i for i, p in enumerate(ref["shape_label"])}
        os = {p: i for i, p in enumerate(ours["shape_label"])}
        common_shapes = [p for p in ref["shape_label"] if p in os]
        if len(common_shapes) != len(rs) or len(common_shapes) != len(os):
            mm.append("shape set differs: ref=%d ours=%d common=%d" % (len(rs), len(os), len(common_shapes)))
        for p in common_shapes:
            i, j = rs[p], os[p]
            rbidx, obidx = int(ref["shape_body"][i]), int(ours["shape_body"][j])
            rbody = str(ref["body_label"][rbidx]) if rbidx >= 0 else "world"
            obody = str(ours["body_label"][obidx]) if obidx >= 0 else "world"
            if rbody != obody:
                mm.append("%s.shape_body: ref %s vs ours %s" % (p, rbody, obody))
            for arr in _SHAPE_ARRAYS[1:]:
                if arr == "shape_collision_group":
                    continue
                if arr == "_shape_sdf_index" and arr in ref and arr in ours:
                    if (int(ref[arr][i]) >= 0) != (int(ours[arr][j]) >= 0):
                        mm.append("%s has SDF: ref %s vs ours %s" % (p, ref[arr][i] >= 0, ours[arr][j] >= 0))
                    continue
                if arr == "shape_flags" and arr in ref and arr in ours:
                    import newton

                    physics_mask = ~int(newton.ShapeFlags.VISIBLE)
                    ref_flags = int(ref[arr][i]) & physics_mask
                    our_flags = int(ours[arr][j]) & physics_mask
                    if ref_flags != our_flags:
                        mm.append("%s.%s: ref %s vs ours %s" % (p, arr, ref_flags, our_flags))
                    continue
                if arr == "shape_transform" and arr in ref and arr in ours:
                    rt, ot = ref[arr][i], ours[arr][j]
                    if not np.allclose(rt[:3], ot[:3], atol=atol, rtol=rtol) or not _quat_close(
                        rt[3:7][None, :], ot[3:7][None, :], atol
                    ):
                        mm.append("%s.%s: ref %s vs ours %s" % (p, arr, rt, ot))
                    continue
                if arr in ref and arr in ours and not np.allclose(ref[arr][i], ours[arr][j], atol=atol, rtol=rtol):
                    mm.append("%s.%s: ref %s vs ours %s" % (p, arr, ref[arr][i], ours[arr][j]))
            if "shape_mesh" in ref and "shape_mesh" in ours:
                rm, om = ref["shape_mesh"][i], ours["shape_mesh"][j]
                if (rm is None) != (om is None):
                    mm.append("%s.shape_mesh: ref %s vs ours %s" % (p, rm is not None, om is not None))
                elif rm is not None:
                    if not _meshes_close(rm, om, atol, rtol):
                        mm.append("%s.shape_mesh differs" % p)
            if (
                "shape_mesh_maxhullvert" in ref
                and "shape_mesh_maxhullvert" in ours
                and ref["shape_mesh_maxhullvert"][i] != ours["shape_mesh_maxhullvert"][j]
            ):
                mm.append(
                    "%s.shape_mesh_maxhullvert: ref %s vs ours %s"
                    % (p, ref["shape_mesh_maxhullvert"][i], ours["shape_mesh_maxhullvert"][j])
                )

        def _label_pairs(model, pairs):
            labels = model["shape_label"]
            return {
                tuple(sorted((str(labels[int(a)]), str(labels[int(b)]))))
                for a, b in pairs
            }

        rfilters = _label_pairs(ref, ref.get("shape_collision_filter_pairs", ()))
        ofilters = _label_pairs(ours, ours.get("shape_collision_filter_pairs", ()))
        def _groups_collide(a, b):
            if a == 0 or b == 0:
                return False
            if a > 0:
                return a == b or b < 0
            return a != b

        if "shape_collision_group" in ref and "shape_collision_group" in ours:
            # USD group rules can be encoded as numeric groups or explicit
            # pairs. Compare the combined filtering decision, not its encoding.
            for offset, p in enumerate(common_shapes):
                for q in common_shapes[offset + 1 :]:
                    ri, rj = rs[p], rs[q]
                    oi, oj = os[p], os[q]
                    rc = _groups_collide(int(ref["shape_collision_group"][ri]), int(ref["shape_collision_group"][rj]))
                    oc = _groups_collide(int(ours["shape_collision_group"][oi]), int(ours["shape_collision_group"][oj]))
                    pair = tuple(sorted((str(p), str(q))))
                    rc = rc and pair not in rfilters
                    oc = oc and pair not in ofilters
                    if rc != oc:
                        mm.append("shape collision filtering for %s/%s: ref %s vs ours %s" % (p, q, rc, oc))
        elif rfilters != ofilters:
            mm.append("shape collision filter pairs: ref %s vs ours %s" % (sorted(rfilters), sorted(ofilters)))
    if check_joints:
        # Align joints by their (parent_label, child_label) edge: merged USD
        # siblings map many paths to one Newton D6 joint, while a body edge is a
        # unique key for the tree articulations + free joints here. Compare the per-joint
        # fields (type + anchor frames), then use each matched joint's start
        # offsets to compare its coordinate- and DOF-indexed state, limits, and
        # actuation fields.
        def _edge(m, k):
            pp = m["body_label"][m["joint_parent"][k]] if m["joint_parent"][k] >= 0 else "world"
            cc = m["body_label"][m["joint_child"][k]] if m["joint_child"][k] >= 0 else "world"
            return (str(pp), str(cc))

        rn, on = len(ref.get("joint_parent", [])), len(ours.get("joint_parent", []))
        if rn != on:
            mm.append("joint count differs: ref=%d ours=%d" % (rn, on))
        def _joint_map(model, count):
            edges = [_edge(model, k) for k in range(count)]
            labels = model.get("joint_label", ())
            result = {}
            occurrences = {}
            for k, edge in enumerate(edges):
                if edges.count(edge) == 1:
                    discriminator = ""
                else:
                    label = str(labels[k]) if k < len(labels) else ""
                    occurrence = occurrences.get(edge, 0)
                    occurrences[edge] = occurrence + 1
                    discriminator = label if label.startswith("/") else "#%d" % occurrence
                result[(*edge, discriminator)] = k
            return result

        rj = _joint_map(ref, rn)
        oj = _joint_map(ours, on)
        for key, k in rj.items():
            edge = key[:2]
            l = oj.get(key)
            if l is None:
                mm.append("joint %s missing in ours" % (key,))
                continue
            for arr in ("joint_type", "joint_X_p", "joint_X_c"):
                if (
                    arr in ref
                    and arr in ours
                    and not np.allclose(
                        np.asarray(ref[arr][k], float), np.asarray(ours[arr][l], float), atol=atol, rtol=rtol
                    )
                ):
                    mm.append("%s.%s: ref %s vs ours %s" % (edge, arr, ref[arr][k], ours[arr][l]))
            layouts = [
                ("joint_q", "joint_q_start"),
                ("joint_qd", "joint_qd_start"),
                ("joint_target_q", "joint_target_q_start"),
                *((name, "joint_qd_start") for name in _JOINT_DOF_ARRAYS),
            ]
            for values, starts in layouts:
                if values not in ref or values not in ours or starts not in ref or starts not in ours:
                    continue
                rv = ref[values][ref[starts][k] : ref[starts][k + 1]]
                ov = ours[values][ours[starts][l] : ours[starts][l + 1]]
                if rv.shape != ov.shape or not np.allclose(rv, ov, atol=atol, rtol=rtol):
                    mm.append("%s.%s: ref %s vs ours %s" % (edge, values, rv, ov))
        if check_articulations and "joint_articulation" in ref and "joint_articulation" in ours:
            common_edges = set(rj) & set(oj)

            def _membership(model, joints, edge):
                art = int(model["joint_articulation"][joints[edge]])
                if art < 0:
                    return None
                return frozenset(
                    candidate
                    for candidate in common_edges
                    if int(model["joint_articulation"][joints[candidate]]) == art
                )

            for edge in common_edges:
                rm = _membership(ref, rj, edge)
                om = _membership(ours, oj, edge)
                if rm != om:
                    mm.append("%s.joint_articulation: ref %s vs ours %s" % (edge, sorted(rm or ()), sorted(om or ())))
    if check_articulations and ref.get("n_art") != ours.get("n_art"):
        mm.append("articulation count: ref %s vs ours %s" % (ref.get("n_art"), ours.get("n_art")))
    if ref.get("up_axis") != ours.get("up_axis"):
        mm.append("up axis: ref %s vs ours %s" % (ref.get("up_axis"), ours.get("up_axis")))
    if (
        "gravity" in ref
        and "gravity" in ours
        and not np.allclose(ref["gravity"], ours["gravity"], atol=atol, rtol=rtol)
    ):
        mm.append("gravity: ref %s vs ours %s" % (ref["gravity"], ours["gravity"]))
    return mm


def _step_model(model, n, *, state=None):
    import newton

    solver = newton.solvers.SolverXPBD(model, iterations=ITERS)
    s0 = state if state is not None else model.state()
    s1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    for _ in range(n):
        for _ in range(SUBSTEPS):
            s0.clear_forces()
            collision_pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, DT / SUBSTEPS)
            s0, s1 = s1, s0
    return s0.body_q.numpy()  # (B,7): px,py,pz, qx,qy,qz,qw


def _run_reference(src, *args):
    """Run a reference snippet (which uses add_usd) in an isolated subprocess —
    the ovstage-built USD kept off LD_LIBRARY_PATH so the selected OpenUSD
    provider loads cleanly, and PYTHONPATH set so the child can ``import
    diff_harness`` — returning its ``PICKLE64 ``-prefixed payload, unpickled.
    Shared by reference_poses/reference_model."""
    env = dict(os.environ)
    for k in ("OVSTAGE_LIBRARY_PATH", "OVPOPULATION_LIBRARY_PATH", "OVHIERARCHY_LIBRARY_PATH"):
        env.pop(k, None)
    env.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
    python_paths = [str(pathlib.Path(__file__).parent)]
    newton_source = env.get("OVNEWTON_NEWTON_SOURCE")
    if newton_source:
        python_paths.insert(0, newton_source)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    proc = subprocess.run([sys.executable, "-c", src, *(str(a) for a in args)], capture_output=True, text=True, env=env)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("PICKLE64 ")), None)
    if proc.returncode != 0 or line is None:
        raise AssertionError(
            f"reference subprocess failed with exit code {proc.returncode}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return pickle.loads(base64.b64decode(line[len("PICKLE64 ") :]))


# reference: run add_usd in an isolated subprocess (selected provider, no built USD)
_REF_SRC = r"""
import sys, base64, json, pickle
import numpy as np
import newton
from newton.usd import SchemaResolverNewton
asset, n, perturb, dt, substeps, iters, options = (
    sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
    float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
    json.loads(sys.argv[7]),
)
b = newton.ModelBuilder()
ret = b.add_usd(asset, schema_resolvers=[SchemaResolverNewton()], **options)
m = b.finalize()
import newton as _newton
solver = _newton.solvers.SolverXPBD(m, iterations=iters)
s0, s1 = m.state(), m.state()
if perturb != 0.0:
    import warp as wp
    jq = s0.joint_q.numpy(); jq[:] += perturb
    s0.joint_q.assign(wp.array(jq, dtype=wp.float32, device=m.device))
control = m.control()
collision_pipeline = _newton.CollisionPipeline(m)
contacts = collision_pipeline.contacts()
for _ in range(n):
    for _ in range(substeps):
        s0.clear_forces(); collision_pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt / substeps); s0, s1 = s1, s0
q = s0.body_q.numpy()
out = {p: q[i].tolist() for p, i in ret["path_body_map"].items()}
print("PICKLE64 " + base64.b64encode(pickle.dumps(out)).decode())
"""


def reference_poses(asset, n, *, perturb_joint_q=0.0, **add_usd_options):
    options = {**ADD_USD_PHYSICS_PROFILE, **add_usd_options}
    d = _run_reference(
        _REF_SRC,
        asset,
        n,
        perturb_joint_q,
        DT,
        SUBSTEPS,
        ITERS,
        json.dumps(options),
    )
    return {p: (np.array(v[:3]), np.array(v[3:7])) for p, v in d.items()}


def max_pos_error(ref, ours):
    common = set(ref) & set(ours)
    assert common, f"no shared prim paths: ref={sorted(ref)} ours={sorted(ours)}"
    return max(float(np.linalg.norm(ref[p][0] - ours[p][0])) for p in common), common


def max_ori_error(ref, ours):
    """Max geodesic orientation error (radians) over shared prims. Quats are
    xyzw; the geodesic angle 2*acos(|dot|) is sign-agnostic (q and -q are the
    same rotation)."""
    common = set(ref) & set(ours)
    assert common, f"no shared prim paths: ref={sorted(ref)} ours={sorted(ours)}"

    def _ang(qa, qb):
        d = min(1.0, abs(float(np.dot(qa, qb))))
        return float(2.0 * np.arccos(d))

    return max(_ang(ref[p][1], ours[p][1]) for p in common), common


# ─── Model-level differential ────────────────────────────────────────
# Compare the *built* Model (mass / inertia / CoM / pose per body, type / parent
# / child / anchor frames per joint) against add_usd, keyed by prim path — no
# solver, no rendering. This isolates IMPORT correctness (what attach_ovstage
# builds) from dynamics, and is what surfaces gaps the gravity-only pose
# differential is blind to (e.g. mass on free-falling bodies).

_MODEL_REF_SRC = r"""
import sys, base64, json, pickle
import newton
from diff_harness import _model_arrays
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx
b = newton.ModelBuilder()
resolvers = [SchemaResolverNewton()]
if sys.argv[3] == "1":
    resolvers.append(SchemaResolverPhysx())
options = json.loads(sys.argv[2])
b.add_usd(
    sys.argv[1],
    schema_resolvers=resolvers,
    **options,
)
m = b.finalize(skip_validation_joints=True)

print("PICKLE64 " + base64.b64encode(pickle.dumps(_model_arrays(m))).decode())
"""


def reference_model(asset, *, include_physx_resolver=False, **add_usd_options):
    """Build ``asset`` via ``add_usd`` in an isolated subprocess and return its
    :func:`_model_arrays` dump. The physics-only profile excludes visual-only
    geometry. Focused ``JointStateAPI`` tests opt into Newton's PhysX resolver
    until that standard schema moves into the default importer path."""
    if "schema_resolvers" in add_usd_options:
        raise ValueError("use include_physx_resolver to select the reference schema resolvers")
    options = {**ADD_USD_PHYSICS_PROFILE, **add_usd_options}
    return _run_reference(_MODEL_REF_SRC, asset, json.dumps(options), int(include_physx_resolver))
