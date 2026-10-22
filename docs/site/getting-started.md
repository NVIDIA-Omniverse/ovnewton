<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting started

## Install

Install the current public release of `ovnewton` from [PyPI](https://pypi.org/project/ovnewton/).

```console
pip install ovnewton
```

## Run the installed example

Run the installed example without rendering:

```console
python -m ovnewton.examples.example_ovnewton_basic --device cpu --no-render
```

The example loads the `scene_rigid_bodies.usda` scene included in the ovnewton wheel. Install the `examples` extra for ovrtx rendering and MuJoCo-Warp:

```console
pip install "ovnewton[examples]"
python -m ovnewton.examples.example_ovnewton_basic
```

The same example can build the scene through ovnewton and advance it with the
MuJoCo-Warp or VBD backend on CUDA or CPU:

```console
python -m ovnewton.examples.example_ovnewton_basic --solver mujoco
python -m ovnewton.examples.example_ovnewton_basic --solver vbd
```

All solver choices use a model populated from the registered USD schemas.
Selecting MuJoCo does not require a scene authored with `Mjc*` schemas.
The example registers solver-owned model attributes before population and, for
VBD, colors the populated builder before finalization.
The normal MuJoCo path uses MuJoCo-Warp on both CPU and CUDA and supplies
contacts through Newton's collision pipeline. Native MuJoCo CPU execution and
MuJoCo contact generation are rare debugging options, not the recommended
application path.

The simulation runs on `cuda:0` by default. Pass `--device cpu` to simulate on the CPU. The wheel also includes a complete cartpole scene:

```console
python -m ovnewton.examples.example_ovnewton_basic --stage scene_cartpole
```

OVRTX rendering requires a compatible NVIDIA RTX-capable GPU and supported driver, even with `--device cpu`. Use `--device cpu --no-render` on a CPU-only machine.

On a machine with more than one GPU, Newton and ovrtx can use different devices. `--device` selects the Warp device used by Newton. `--render-device` selects one CUDA-visible device index for ovrtx:

```console
python -m ovnewton.examples.example_ovnewton_basic --stage scene_cartpole --device cuda:1 --render-device 0
```

The resulting `model.device` is the source of truth for Newton state and ovnewton transport allocations. In Isaac Sim, select that physics placement through `NewtonConfig.device`. OVRTX device selection remains independent.

To watch several Ant robots from Newton's installed assets fall onto a ground plane:

```console
python -m ovnewton.examples.example_ovnewton_robot
python -m ovnewton.examples.example_ovnewton_robot --solver vbd
```

OVRTX debug synchronization is disabled by default. Use `--render-debug-sync true`
for diagnostics, or `--render-debug-sync false` to explicitly use the default:

```console
python -m ovnewton.examples.example_ovnewton_basic --render-debug-sync false
python -m ovnewton.examples.example_ovnewton_robot --render-debug-sync true
```

Rendering remains blocking with debug synchronization disabled and preserves
the simulate → publish → seal → render order. For the basic example,
`--no-render` makes the option irrelevant. To measure its effect, compare the
same workload without `--timing`, which adds separate synchronization.

## Run a simulation

The `scene_path` below must point to a USD file supplied by your application.

Use `ovnewton.population_desc()` to select physics data and preserve composed
schema names, including unregistered APIs. Combine it with a rendering descriptor
as below, or use it alone for physics. Register schemas before population.

```python
import newton
import ovnewton
import ovstage
from ovstage import PopulationDomain, population

scene_path = "/path/to/scene.usd"

ovnewton.register_usd_schemas()
stage = ovstage.Stage()
population.open_usd_with_desc(
    stage, scene_path, ordinal=1, time=population.DEFAULT_TIME,
    descs=(population.Desc(domains=PopulationDomain.RENDERING), ovnewton.population_desc()),
)
stage.advance_write_floor(ordinal=1).wait()

builder = newton.ModelBuilder()
import_result = ovnewton.add_ovstage(builder, stage)
model = builder.finalize(skip_validation_joints=import_result.has_orphan_joints)
solver = newton.solvers.SolverXPBD(model)
binding = ovnewton.attach_ovstage(stage, model=model)
state_0 = model.state()
state_1 = model.state()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
control = model.control()
collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()

num_substeps = 4
collide_every_substep = False
dt = 1.0 / (60 * num_substeps)
ordinal = 2

for _ in range(240):
    for substep in range(num_substeps):
        state_0.clear_forces()
        if collide_every_substep or substep == 0:
            collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    binding.update_to_ovstage(state_0, ordinal=ordinal)
    stage.advance_write_floor(ordinal).wait()
    ordinal += 1
```

This loop generates contacts once per frame by default. Set
`collide_every_substep = True` to refresh contacts before every solver substep,
for example when debugging contact behavior or simulating fast motion. Reusing
contacts reduces collision work, but contacts can become stale between substeps;
check that the chosen cadence preserves the behavior your scene needs. This
controls the external `CollisionPipeline`, not a solver's internal collision
handling. See Newton's [collision-frequency guidance](https://newton-physics.github.io/newton/latest/concepts/collisions.html#collision-frequency-in-the-simulation-loop).

Call `binding.update_from_ovstage(state_0, control)` before a step when you intend
to resynchronize all supported state and drive targets from the sealed stage.
The default applies current stage values even if they predate the live simulation
state. For a controller that only changes drive targets, select those fields:

```python
binding.update_from_ovstage(
    state_0, control,
    attributes=("jointPositionTarget", "jointVelocityTarget"),
)
```

Either target name can be selected alone. This leaves every state array and
unselected control array unchanged. The same selection mechanism supports:

| Field | Destination |
| --- | --- |
| `body_q` | All bound body poses |
| `body_qd` | All bound body linear/angular velocities |
| `jointPosition` / `jointVelocity` | Existing single-DOF joint coordinates / velocities |
| `jointPositionTarget` / `jointVelocityTarget` | Existing single-DOF drive targets |

Fields can be combined in one read. Explicit selections update only the listed
fields and preserve unmapped joint coordinates. They do not run forward/inverse
kinematics to reconcile body and joint arrays; use the default full resync when
body-derived joint coordinates must also be reconstructed. Selection does not
add path filtering, D6 drive channels, or teleport/kinematic-target handling.
Angular joint values convert from stage degrees to Newton radians; linear values
use the stage's length units. The active drive mode determines which target
columns are required. A missing required selected column or inconsistent snapshot
fails before committing any selected state or control changes. An empty selection
reads no payloads.

If you supply `model=`, its body and joint labels must match the stage paths.

Already-populated stages must preserve the same schema metadata. `add_ovstage()`
and `attach_ovstage()` do not reconstruct discarded names, so unsupported-physics
detection requires this descriptor or equivalent population. Preserving a schema
name does not add support for its semantics.

## Configure a solver-specific model

Configure a builder that does not yet contain model data, then add the ovstage
contents and perform any solver-specific preparation before finalizing it. For
VBD:

```python
builder = newton.ModelBuilder()
newton.solvers.SolverVBD.register_custom_attributes(builder)

import_result = ovnewton.add_ovstage(builder, stage)
builder.color()

model = builder.finalize(skip_validation_joints=import_result.has_orphan_joints)
solver = newton.solvers.SolverVBD(model)
binding = ovnewton.attach_ovstage(stage, model=model)
```

MuJoCo uses the same sequence but registers
`SolverMuJoCo.register_custom_attributes` and does not need `builder.color()`.
Construct it with `SolverMuJoCo(model, use_mujoco_contacts=False)` to use Newton's collision pipeline.
`add_ovstage()` preserves builder defaults and custom attribute definitions,
applies the stage's up axis and gravity, and returns without finalizing or
creating a `StageBinding`. The current implementation supports one ovstage per
builder. Compose all USD assets for the simulation into that ovstage before
calling `add_ovstage()`.
