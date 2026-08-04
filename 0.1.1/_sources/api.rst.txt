..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

API reference
=============

Attachment
----------

.. autofunction:: ovnewton.attach_ovstage

.. autoclass:: ovnewton.StageBinding
   :members:

Output reads
------------

.. autoclass:: ovnewton.Query
   :members:

.. autoclass:: ovnewton.ReadResult
   :members:

.. autoclass:: ovnewton.ReadGroup
   :members:

.. autoclass:: ovnewton.ReadGroupMeta
   :members:

.. autoclass:: ovnewton.ReadGroupCudaSync
   :members:

Errors
------

.. autoexception:: ovnewton.InvalidPhysicsError

.. autoexception:: ovnewton.UnsupportedPhysicsError

.. autoexception:: ovnewton.OvstageContractError
