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

## Run a simulation

The `scene_path` below must point to a USD file supplied by your application.

```python
import newton
import ovnewton
import ovstage
from ovstage import PopulationDomain, population

scene_path = "/path/to/scene.usd"

ovnewton.register_usd_schemas()
stage = ovstage.Stage()
population.open_usd(stage, scene_path, ordinal=1, domains=PopulationDomain.ALL)
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
dt = 1.0 / (60 * num_substeps)
ordinal = 2

for _ in range(240):
    for _ in range(num_substeps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    binding.update_to_ovstage(state_0, ordinal=ordinal)
    stage.advance_write_floor(ordinal).wait()
    ordinal += 1
```

Call `binding.update_from_ovstage(state_0, control)` before a step to apply all supported state and drive target fields from sealed stage changes. This updates both `state_0` and `control`; callers cannot select a subset. If you supply `model=`, its body and joint labels must match the stage paths.

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
