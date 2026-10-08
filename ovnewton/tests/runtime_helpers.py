# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only adapters and NumPy oracles for runtime transport tests."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ovnewton._src import _runtime


def lanes_tensor(array: np.ndarray, lanes: int):
    """Describe contiguous test data as an ``N``-element, multi-lane DLTensor."""
    from ovstage import make_dltensor, numpy_to_dldatatype

    array = np.ascontiguousarray(array)
    dtype = numpy_to_dldatatype(array.dtype)
    dtype.lanes = lanes
    return make_dltensor(array, dtype=dtype, shape=[array.shape[0]], ndim=1)


def warp_lanes_tensor(array: Any, count: int, lanes: int):
    """Describe a contiguous Warp array through ovstage's DLPack adapter."""
    from ovstage import DLDataType, DLDataTypeCode, make_dltensor

    producer = array if lanes == 1 else array.reshape((count, lanes))
    dtype = DLDataType(code=DLDataTypeCode.kDLFloat, bits=32, lanes=lanes)
    return make_dltensor(producer, dtype=dtype, shape=[count], ndim=1)


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
    _runtime.read_runtime(
        stage,
        path_dictionary,
        ordinal,
        state,
        runtime=runtime,
        control=model.control(),
    )
