#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the ovnewton test suite. The installed ovstage package is the default;
# OVNEWTON_OVSTAGE_PATH opts into a Kit source build for development.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NEWTON_SOURCE_PATH=""
if [[ -n "${OVNEWTON_NEWTON_SOURCE:-}" ]]; then
    if [[ ! -d "$OVNEWTON_NEWTON_SOURCE/newton" ]]; then
        echo "error: OVNEWTON_NEWTON_SOURCE is not a Newton source checkout: $OVNEWTON_NEWTON_SOURCE" >&2
        exit 1
    fi
    NEWTON_SOURCE_PATH="$OVNEWTON_NEWTON_SOURCE:"
fi
export PYTHONPATH="$NEWTON_SOURCE_PATH$REPO:$REPO/ovnewton/tests${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${OVNEWTON_OVSTAGE_PATH:-}" ]]; then
    OVS="$(cd -- "$OVNEWTON_OVSTAGE_PATH" 2>/dev/null && pwd || echo "$OVNEWTON_OVSTAGE_PATH")"
    REL="$OVS/rendering/_build/linux-x86_64/release"
    if [[ ! -f "$REL/libovstage.so" ]]; then
        echo "error: Kit ovstage is not built at $REL/libovstage.so" >&2
        echo "       build the required ovstage targets before using a source override" >&2
        exit 1
    fi

    UPD="$REL/usd_plugins"
    export OVSTAGE_LIBRARY_PATH_HINT="$REL"
    export PYTHONPATH="$OVS/rendering/ovstage/public/python:$PYTHONPATH"
    export LD_LIBRARY_PATH="$REL:$REL/plugins:$REL/plugins/usdrt:$REL/plugins/scenegraph:$REL/plugins/omni.fabric:$REL/plugins/usd:$REL/plugins/gpu.foundation:$REL/plugins/omni.client.lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OV_PXR_PLUGINPATH_2511="$UPD/omni_nurec_types:$UPD/omni_sensors:$UPD/omni_lens_distortion:$UPD/omni_projectors:$UPD/omni_semantics:$UPD/rtx_settings:$UPD/physical_lighting:$UPD/usd_particle_field:$UPD/omni_playback${OV_PXR_PLUGINPATH_2511:+:$OV_PXR_PLUGINPATH_2511}"
    export HD_ENABLE_SCENE_INDEX_EMULATION=0 USDIMAGINGGL_ENGINE_ENABLE_SCENE_INDEX=0
fi

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO/.venv/bin/python" ]]; then
        PYTHON="$REPO/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

exec "$PYTHON" -m pytest "$@"
