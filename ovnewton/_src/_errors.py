# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors and recoverable import decisions produced while binding an ovstage."""

import logging

_LOGGER = logging.getLogger("ovnewton")


def log_diagnostic(code, message, *, path=None, level=logging.WARNING) -> None:
    """Log a recoverable import decision with machine-readable context."""
    location = f" at {path}" if path else ""
    _LOGGER.log(
        level,
        "%s%s: %s",
        code,
        location,
        message,
        extra={"diagnostic_code": code, "prim_path": path},
    )


class InvalidPhysicsError(ValueError):
    """The populated stage contains malformed or contradictory physics data."""


class OvstageContractError(RuntimeError):
    """Ovstage returned data that violates the transport contract ovnewton consumes."""


class UnsupportedPhysicsError(NotImplementedError):
    """The stage uses physics semantics that ovnewton does not implement.

    This is reserved for semantics whose loss would materially change physics
    and for which ovnewton has no exact representation or safe import policy.
    """
