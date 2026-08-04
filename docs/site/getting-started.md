<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting started

## Install

After the public release, `ovnewton` will be available from [PyPI](https://pypi.org/project/ovnewton/).

```console
pip install ovnewton
```

## Run the installed example

Run the installed example without rendering:

```console
python -m ovnewton.examples.example_ovnewton_basic --device cpu --no-render
```

The example loads the `scene_rigid_bodies.usda` scene included in the ovnewton wheel. Install the `examples` extra to render it with ovrtx:

```console
pip install "ovnewton[examples]"
python -m ovnewton.examples.example_ovnewton_basic
```

The simulation runs on `cuda:0` by default. Pass `--device cpu` to simulate on the CPU. The wheel also includes a complete cartpole scene:

```console
python -m ovnewton.examples.example_ovnewton_basic --stage scene_cartpole
```

## Run a simulation

The `scene_path` below must point to a USD file supplied by your application.

```python
import newton
import ovnewton
import ovstage
from ovstage import PopulationDomain, population

scene_path = "/path/to/scene.usd"

stage = ovstage.Stage()
population.open_usd(stage, scene_path, ordinal=1, domains=PopulationDomain.ALL)
stage.advance_write_floor(ordinal=1).wait()

binding = ovnewton.attach_ovstage(stage)
model = binding.model
solver = newton.solvers.SolverXPBD(model)
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

num_substeps = 4
dt = 1.0 / (60 * num_substeps)
ordinal = 2

for _ in range(240):
    for _ in range(num_substeps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    binding.update_to_ovstage(state_0, ordinal=ordinal)
    stage.advance_write_floor(ordinal).wait()
    ordinal += 1
```

Call `binding.update_from_ovstage(state_0, control)` before a step to apply all supported state and drive target fields from sealed stage changes. This updates both `state_0` and `control`; callers cannot select a subset. If you supply `model=`, its body and joint labels must match the stage paths.
