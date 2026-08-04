# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ovstage column, identity, and ordinal operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from ovstage import OrdinalRange

from ._errors import OvstageContractError
from ._schema_names import USD_PATH


def _count(stage: Any, query: Any) -> int:
    query.wait()
    return stage.fetch_query_result(query).total_prim_count


def _validate_payload_ordinal(group: Any, requested_ordinal: int, subject: str) -> None:
    actual = int(group.ordinal)
    requested = int(requested_ordinal)
    if actual > requested:
        raise OvstageContractError(
            f"{subject} returned payload ordinal {actual} newer than requested ordinal {requested}; "
            "ovstage does not retain historical payloads"
        )


def _global_write_floor(stage: Any) -> int:
    with stage.get_attribute_write_floor() as query:
        return int(query.fetch())


def _oldest_change_ordinal(stage: Any) -> int:
    with stage.get_oldest_preserved_ordinal() as query:
        return int(query.fetch())


def _resolve_read_ceiling(stage: Any, ordinal: Optional[int]) -> int:
    """Resolve a sealed read ceiling covered by retained change metadata."""
    sealed = _global_write_floor(stage)
    resolved = sealed if ordinal is None else int(ordinal)
    if resolved < 0:
        raise ValueError(f"ordinal must be non-negative, got {resolved}")
    if resolved > sealed:
        raise ValueError(f"ordinal {resolved} is newer than the sealed write floor {sealed}")
    if resolved < sealed:
        oldest = _oldest_change_ordinal(stage)
        if resolved < oldest:
            raise ValueError(
                f"ordinal {resolved} is older than the retained change-membership frontier {oldest}"
            )
    return resolved


def _consume_column_group(rows: Dict[int, np.ndarray], group: Any, attr: str, *, ragged: bool) -> None:
    if group.is_delete:
        return
    if ragged and group.tensor_count:
        for local in range(group.prim_count):
            tensor_index = group.data_row_index(local)
            if tensor_index < 0 or tensor_index >= group.tensor_count:
                raise OvstageContractError(
                    f"attribute {attr!r} returned ragged row {tensor_index} outside [0, {group.tensor_count})"
                )
            rows[group.prim_index(local)] = np.array(group.array(tensor_index), copy=True)
        return
    if not ragged and group.tensor_count not in (0, 1):
        raise OvstageContractError(
            f"fixed attribute {attr!r} returned {group.tensor_count} tensors; expected at most one"
        )
    if ragged or group.tensor_count == 0:
        return

    arr = np.asarray(group.array(0))
    if group.has_data_index_map:
        tensor = group.tensor(0)
        nrows = int(tensor.shape[0]) if int(tensor.ndim) >= 1 else group.prim_count
    else:
        nrows = group.prim_count
    if nrows <= 0 or arr.size % nrows:
        raise OvstageContractError(f"fixed attribute {attr!r} has {arr.size} values for {nrows} logical rows")
    width = arr.size // nrows
    for local in range(group.prim_count):
        row = group.data_row_index(local)
        if row < 0 or row >= nrows:
            raise OvstageContractError(f"fixed attribute {attr!r} returned row {row} outside [0, {nrows})")
        rows[group.prim_index(local)] = np.array(arr[row * width : (row + 1) * width])


def read_columns(
    stage: Any,
    pd: Any,
    query: Any,
    attrs: Sequence[str],
    ordinal: int,
    *,
    ragged: Sequence[str] = (),
) -> Dict[str, Dict[int, np.ndarray]]:
    """Read fixed and ragged columns in one ovstage operation."""
    attrs = tuple(attrs)
    if len(attrs) != len(set(attrs)):
        raise ValueError("column read contains duplicate attributes")
    ragged_attrs = frozenset(ragged)
    unknown_ragged = ragged_attrs.difference(attrs)
    if unknown_ragged:
        raise ValueError(f"ragged attributes are not in the read: {sorted(unknown_ragged)!r}")
    rows: Dict[str, Dict[int, np.ndarray]] = {attr: {} for attr in attrs}
    if not attrs:
        return rows

    token_to_attr = {pd.intern_token(attr): attr for attr in attrs}
    if len(token_to_attr) != len(attrs):
        raise OvstageContractError("column names resolved to duplicate ovstage tokens")
    with stage.read_attributes(query, list(token_to_attr), OrdinalRange.latest(ordinal)) as read:
        read.wait()
        group = read.fetch_next()
        while group is not None:
            try:
                attr = token_to_attr.get(group.attribute)
                if attr is None:
                    raise OvstageContractError(f"column read returned unexpected attribute token {group.attribute}")
                _validate_payload_ordinal(group, ordinal, f"attribute {attr!r}")
                _consume_column_group(rows[attr], group, attr, ragged=attr in ragged_attrs)
            finally:
                stage.release_group(group)
            group = read.fetch_next()
    return rows


def read_fixed(stage: Any, pd: Any, query: Any, attr: str, ordinal: int) -> Dict[int, np.ndarray]:
    """Read a fixed-size column as ``{prim_index: row}``.

    Deduplicated tensors use their pool size as the logical row count; other
    groups use ``prim_count``. ``data_row_index`` selects each prim's row.
    """
    return read_columns(stage, pd, query, [attr], ordinal)[attr]


def _read_path_columns(
    stage: Any,
    pd: Any,
    query: Any,
    attrs: Sequence[str],
    ordinal: int,
) -> Dict[str, Dict[str, str]]:
    """Read path-valued metadata keyed by each group's prim identity."""
    token_to_attr = {pd.intern_token(attr): attr for attr in attrs}
    columns: Dict[str, Dict[str, str]] = {attr: {} for attr in attrs}
    with stage.read_attributes(query, list(token_to_attr), OrdinalRange.latest(ordinal)) as read:
        read.wait()
        group = read.fetch_next()
        while group is not None:
            try:
                attr = token_to_attr.get(group.attribute)
                if attr is None:
                    raise OvstageContractError(f"path read returned unexpected attribute token {group.attribute}")
                _validate_payload_ordinal(group, ordinal, f"attribute {attr!r}")
                rows: Dict[int, np.ndarray] = {}
                _consume_column_group(rows, group, attr, ragged=False)
                if not rows:
                    continue
                group_paths = pd.get_path_strings(group.prim_list)
                for prim_index, row in rows.items():
                    if prim_index < 0 or prim_index >= len(group_paths):
                        raise OvstageContractError(
                            f"attribute {attr!r} returned prim row {prim_index} outside its path list"
                        )
                    if len(row) != 1:
                        raise OvstageContractError(f"path attribute {attr!r} returned a row with {len(row)} values")
                    path = group_paths[prim_index]
                    value = pd.path_to_string(int(row[0]))
                    if attr == USD_PATH and path != value:
                        raise OvstageContractError(f"{USD_PATH!r} identifies {path} as {value}")
                    previous = columns[attr].setdefault(path, value)
                    if previous != value:
                        raise OvstageContractError(f"attribute {attr!r} returned conflicting rows for {path}")
            finally:
                stage.release_group(group)
            group = read.fetch_next()
    return columns


def _query_paths(
    stage: Any,
    pd: Any,
    query: Any,
    ordinal: int,
    *,
    expected_count: int,
) -> List[str]:
    """Materialize query identity through readable ``usd-path``."""
    paths = set(_read_path_columns(stage, pd, query, [USD_PATH], ordinal)[USD_PATH].values())
    if len(paths) != expected_count:
        raise OvstageContractError(
            f"query matched {expected_count} prims but {USD_PATH!r} materialized {len(paths)} paths"
        )
    return sorted(paths)


@contextmanager
def _path_list_query(stage: Any, pd: Any, paths: Sequence[str]):
    """Yield a query whose row indices match ``paths`` and release its handles."""
    path_list = pd.create_path_list_from_strings(paths)
    try:
        with stage.query_from_path_list(path_list) as query:
            yield query
    finally:
        pd.destroy_path_list(path_list)
