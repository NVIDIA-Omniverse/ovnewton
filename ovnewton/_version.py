# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed ovnewton distribution version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("ovnewton")
except PackageNotFoundError:
    __version__ = "unknown"
