# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def populated_stage():
    """Return a callable that populates a fresh ovstage from a USD asset path
    and returns (stage, pd). The caller owns the with-context / lifecycle."""
    import ovstage
    from ovstage import PopulationDomain, population

    def _make(asset_path):
        stage = ovstage.Stage()
        pd = ovstage.PathDictionary(stage)
        population.open_usd(stage, str(asset_path), ordinal=1, domains=PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        return stage, pd
    return _make
