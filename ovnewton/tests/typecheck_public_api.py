# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pyright: strict, reportMissingTypeStubs=false, reportUnusedCallResult=false

"""Static contract for the public interface in the installed ovnewton wheel."""

from typing import assert_type

import newton
import numpy as np
import numpy.typing as npt
import ovstage
import warp as wp

import ovnewton

assert_type(ovnewton.population_desc(), ovstage.population.Desc)

read_field = assert_type(ovnewton.read_fields()["body_q"], ovnewton.ReadField)
assert_type(read_field.lanes, int | None)
assert_type(read_field.sources, tuple[str, ...])
assert_type(read_field.conversion, str | None)


def check_public_interface(
    stage: ovstage.Stage,
    stage_query: ovstage.Query,
    model: newton.Model,
    state: newton.State,
    control: newton.Control,
    targets: wp.array[wp.float32],
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
    assert_type(query.attributes, tuple[ovstage.Token, ...])
    assert_type(query.prim_count, int)
    assert_type(typed_query.object_type, ovnewton.SimObjectType | None)
    assert_type(typed_query.scope, ovnewton.ObjectScope)

    assert_type(binding.write({"joint_target_q": targets}, control=control, query=query), None)
    assert_type(binding.write({"joint_target_qd": targets}, control=control, query=query), None)
    result = assert_type(binding.read(state, query=query, attributes=["body_q"]), ovnewton.ReadResult)
    assert_type(binding.read(state, query=query, attributes=query.attributes), ovnewton.ReadResult)
    assert_type(
        binding.read(control=control, query=query, attributes=["joint_f"]),
        ovnewton.ReadResult,
    )
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
    assert_type(group.prim_list, ovstage.PrimPathList)
    assert_type(group.prim_count, int)
    assert_type(group.prim_index(0), int)
    assert_type(group.object_type, ovnewton.SimObjectType | None)
    assert_type(group.columns, tuple[ovnewton.ReadColumn, ...])
    assert_type(group.column_count, int)
    column = group.columns[0]
    assert_type(column.attribute, ovstage.Token)
    assert_type(column.tensor(0), ovstage.DLTensor)
    assert_type(column.data_row_index(0), int)
    assert_type(column.array(0), npt.NDArray[np.generic])
    assert_type(column.dlpack(0), ovstage.ManagedDLTensor)
    assert_type(column.data_index_tensor(), ovstage.DLTensor | None)
    assert_type(column.data_index_array(), npt.NDArray[np.generic] | None)
    assert_type(column.data_index_dlpack(), ovstage.ManagedDLTensor | None)
    assert_type(column.meta, ovnewton.ReadColumnMeta)
    assert_type(column.meta.attribute_write_floor_ordinal, int)
    assert_type(column.cuda_sync, ovnewton.ReadColumnCudaSync)
    assert_type(column.cuda_sync.wait_event, int)

    assert_type(binding.update_to_ovstage(state, ordinal=2), None)
    assert_type(binding.update_from_ovstage(state, control, ordinal=2), None)
    assert_type(binding.update_from_ovstage(state, control, attributes=("jointPosition", "jointPositionTarget")), None)
