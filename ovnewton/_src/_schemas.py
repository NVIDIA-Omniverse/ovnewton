# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register the USD schema definitions required by ovnewton."""

from importlib.metadata import distribution


def register_usd_schemas() -> None:
    """Register ovnewton's USD schemas before the first USD schema read.

    Registration is process-wide, irreversible, and safe to repeat. It must run
    before any USD schema definitions are read in the process, including by an
    ovstage population or export call.
    """
    from ovstage import population  # noqa: PLC0415

    descriptors = [
        distribution("newton-usd-schemas").locate_file("newton_usd_schemas/plugInfo.json"),
        distribution("physx-usd-schemas").locate_file("physx_usd_schemas/PhysxSchema/resources/plugInfo.json"),
    ]
    population.register_usd_schemas(descriptors)
