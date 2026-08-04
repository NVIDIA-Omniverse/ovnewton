# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side reads and renderer-facing writes of ovstage transforms.

A thin wrapper over transform columns for a fixed set of prims. The examples
use it to drive the interactive camera, and — while the ovrtx CUDA-write bug
lasts — to mirror body poses back to the renderer (see ``USE_TRANSFORM_RELAY``
in ``example_ovnewton_basic.py``).
"""

import numpy as np
import ovstage

WORLD_MATRIX = "omni:fabric:worldMatrix"
XFORM = "omni:xform"
RESET_XFORM_STACK = "omni:resetXformStack"
USD_PATH = "usd-path"
USD_PRIM_TYPE = "usd-prim-type"


def find_paths_by_type(stage, prim_type, ordinal=1):
    """Sorted paths of every prim of ``prim_type`` (e.g. ``"Camera"``)."""
    found = set()
    with ovstage.PathDictionary(stage) as pd:
        prim_filter = ovstage.Filter(
            [ovstage.Predicate(USD_PRIM_TYPE, ovstage.FilterOp.IN, [prim_type])])
        with stage.query(filter=prim_filter) as query:
            query.wait()
            if not stage.fetch_query_result(query).total_prim_count:
                return []
            # The paths are materialized through a readable identity column.
            token = pd.intern_token(USD_PATH)
            with stage.read_attributes(
                query, [token], ovstage.OrdinalRange.latest(ordinal)
            ) as read:
                read.wait()
                group = read.fetch_next()
                while group is not None:
                    found.update(pd.get_path_strings(group.prim_list))
                    stage.release_group(group)
                    group = read.fetch_next()
    return sorted(found)


class TransformRelay:
    """Read world matrices and write renderer-facing xforms for fixed prims.

    Row order follows the ``paths`` given to the constructor. Reads and writes
    go through one query, so the two are always consistent with each other.
    """

    def __init__(self, stage, paths):
        self._stage = stage
        self._count = len(list(paths))
        self._pd = ovstage.PathDictionary(stage)
        self._path_list = self._pd.create_path_list_from_strings(list(paths))
        self._query = stage.query_from_path_list(self._path_list)
        self._world_matrix = self._pd.intern_token(WORLD_MATRIX)
        self._xform = self._pd.intern_token(XFORM)
        self._dtype = ovstage.numpy_to_dldatatype(np.dtype(np.float64), lanes=16)
        self._scales = self._signed_scales(self.read(1))
        self._reset_written = False

    @staticmethod
    def _signed_scales(matrices):
        """Return row scales while preserving reflected transforms."""
        linear = matrices[:, :3, :3]
        scales = np.linalg.norm(linear, axis=2)
        rotations = linear / np.where(scales > 1.0e-12, scales, 1.0)[:, :, None]
        scales[np.linalg.det(rotations) < 0.0] *= -1.0
        return scales

    def read(self, ordinal):
        """Return the prims' world matrices as ``[N, 4, 4]``, in path order."""
        out = np.zeros((self._count, 4, 4), dtype=np.float64)
        with self._stage.read_attributes(
            self._query, [self._world_matrix], ovstage.OrdinalRange.latest(ordinal)
        ) as read:
            read.wait()
            group = read.fetch_next()
            while group is not None:
                rows = np.from_dlpack(group.dlpack(0)).reshape(-1, 4, 4)
                for local in range(group.prim_count):
                    out[group.prim_index(local)] = rows[local]
                self._stage.release_group(group)
                group = read.fetch_next()
        return out

    def write(self, matrices, ordinal):
        """Write ``matrices`` (``[N, 4, 4]`` or flat ``[N * 16]``) at ``ordinal``.

        The caller still owns the write floor; this does not advance it.
        """
        data = np.array(matrices, dtype=np.float64, copy=True).reshape(-1, 4, 4)
        data[:, :3, :3] *= self._scales[:, :, None]
        tensor = ovstage.make_dltensor(
            data, dtype=self._dtype, shape=[self._count], ndim=1)
        self._stage.write_attribute(
            self._query, self._xform, ordinal=ordinal, tensors=tensor,
            is_array=False, semantic=int(ovstage.AttributeSemantic.MATRIX),
        ).wait()
        if not self._reset_written:
            self._stage.write_attribute(
                self._query,
                RESET_XFORM_STACK,
                ordinal=ordinal,
                tensors=np.ones(self._count, dtype=np.uint8),
                is_array=False,
            ).wait()
            self._reset_written = True

    def close(self):
        """Release the ovstage query, path list, and path dictionary."""
        if self._query is not None:
            self._query.release().wait()
            self._query = None
        if self._path_list is not None:
            self._pd.destroy_path_list(self._path_list)
            self._path_list = None
        if self._pd is not None:
            self._pd.destroy()
            self._pd = None

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
