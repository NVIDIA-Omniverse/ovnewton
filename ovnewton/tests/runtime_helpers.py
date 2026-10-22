# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only body-state reader for runtime transport tests."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Sequence

from ovnewton._src import _runtime


def read_body_state(
    stage: Any,
    path_dictionary: Any,
    body_paths: Sequence[str],
    ordinal: int,
    state: Any,
    model: Any,
) -> None:
    """Exercise the production runtime reader with body channels only."""
    runtime = _runtime._build_runtime_transport(model, body_paths, ())
    _runtime._capture_runtime_read_layout(stage, path_dictionary, runtime, ordinal)
    # This helper has no StageBinding to own the fixed ingress selection.
    with ExitStack() as resources:
        runtime.read_path_list = path_dictionary.create_path_list_from_strings(runtime.paths)
        resources.callback(path_dictionary.destroy_path_list, runtime.read_path_list)
        _runtime.read_runtime(
            stage,
            path_dictionary,
            ordinal,
            state,
            runtime=runtime,
            control=model.control(),
        )
