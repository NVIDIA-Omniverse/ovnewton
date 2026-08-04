# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

from ovnewton._src import _schema_names


def test_schema_name_families():
    assert _schema_names.joint_state("angular", "position") == "state:angular:physics:position"
    assert _schema_names.joint_drive("rotX", "targetVelocity") == "drive:rotX:physics:targetVelocity"
    assert _schema_names.joint_limit("transY", "high") == "limit:transY:physics:high"


def test_schema_names_do_not_load_pxr():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ovnewton._src._schema_names; assert 'pxr' not in sys.modules",
        ],
        check=True,
    )
