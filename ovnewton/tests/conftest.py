# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec

import pytest


def pytest_configure(config):
    """Register ovnewton's schema definitions before test modules are collected."""
    if find_spec("ovstage") is None:
        return

    import ovnewton

    ovnewton.register_usd_schemas()


@pytest.fixture(params=["cpu", "cuda:0"])
def warp_device(request):
    """Exercise CPU and available CUDA devices."""
    import warp as wp

    if request.param == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    return wp.get_device(request.param)


@pytest.fixture
def populated_stage():
    """Return a callable that populates a fresh ovstage from a USD asset path
    and returns (stage, pd). The caller owns the with-context / lifecycle."""
    import ovstage
    from ovstage import PopulationDomain, population

    import ovnewton

    def _make(asset_path):
        stage = ovstage.Stage()
        pd = ovstage.PathDictionary(stage)
        population.open_usd_with_desc(
            stage, str(asset_path), ordinal=1, time=population.DEFAULT_TIME,
            descs=(population.Desc(domains=PopulationDomain.RENDERING), ovnewton.population_desc()),
        )
        stage.advance_write_floor(ordinal=1).wait()
        return stage, pd

    return _make
