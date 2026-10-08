# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pyright: strict, reportMissingTypeStubs=false, reportUnusedCallResult=false

"""Static contract for the public interface in the installed ovnewton wheel."""

from typing import assert_type

import newton
import numpy as np
import numpy.typing as npt
import ovstage

import ovnewton


def check_public_interface(
    stage: ovstage.Stage,
    stage_query: ovstage.Query,
    model: newton.Model,
    state: newton.State,
    control: newton.Control,
) -> None:
    builder = newton.ModelBuilder()
    import_result = assert_type(ovnewton.add_ovstage(builder, stage), ovnewton.OvstageImportResult)
    assert_type(import_result.has_orphan_joints, bool)
    binding = assert_type(ovnewton.attach_ovstage(stage, model=model), ovnewton.StageBinding)
    assert_type(ovnewton.attach_ovstage(stage), ovnewton.StageBinding)
    assert_type(binding.model, newton.Model)

    query = assert_type(binding.query(stage_query), ovnewton.Query)
    assert_type(binding.query(paths=["/World/body"]), ovnewton.Query)
    typed_query = assert_type(
        binding.query(
            object_type=ovnewton.SimObjectType.RIGID_BODY,
            scope=ovnewton.ObjectScope.ALL,
        ),
        ovnewton.Query,
    )
    assert_type(query.attributes, tuple[int, ...])
    assert_type(query.prim_count, int)
    assert_type(typed_query.object_type, ovnewton.SimObjectType | None)
    assert_type(typed_query.scope, ovnewton.ObjectScope)

    result = assert_type(binding.read(state, query=query, attributes=["body_q"]), ovnewton.ReadResult)
    assert_type(
        binding.read(
            state,
            control=control,
            query=query,
            attributes=["jointPositionTarget"],
        ),
        ovnewton.ReadResult,
    )
    assert_type(result.groups, tuple[ovnewton.ReadGroup, ...])
    group = result.groups[0]
    assert_type(group.tensor(0), ovstage.DLTensor)
    assert_type(group.object_type, ovnewton.SimObjectType | None)
    assert_type(group.array(0), npt.NDArray[np.generic])
    assert_type(group.dlpack(0), ovstage.ManagedDLTensor)
    assert_type(group.data_index_tensor(), ovstage.DLTensor | None)
    assert_type(group.data_index_array(), npt.NDArray[np.generic] | None)
    assert_type(group.data_index_dlpack(), ovstage.ManagedDLTensor | None)
    assert_type(group.meta, ovnewton.ReadGroupMeta)
    assert_type(group.meta.attribute_write_floor_ordinal, int)
    assert_type(group.cuda_sync, ovnewton.ReadGroupCudaSync)
    assert_type(group.cuda_sync.wait_event, int)

    assert_type(binding.update_to_ovstage(state, ordinal=2), None)
    assert_type(binding.update_from_ovstage(state, control, ordinal=2), None)
