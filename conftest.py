# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register test selection options before pytest parses command-line arguments."""


def pytest_addoption(parser):
    parser.getgroup("ovnewton").addoption(
        "--render",
        choices=("true", "false"),
        default=None,
        help="Select rendering tests (true) or other tests (false); omit to run both.",
    )


def pytest_collection_modifyitems(config, items):
    render = config.getoption("--render")
    if render is None:
        return

    selected = []
    deselected = []
    for item in items:
        if (item.get_closest_marker("render") is not None) == (render == "true"):
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
