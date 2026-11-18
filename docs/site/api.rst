..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

API reference
=============

Package
-------

.. autodata:: ovnewton.__version__

Attachment
----------

.. autofunction:: ovnewton.register_usd_schemas

.. autofunction:: ovnewton.population_desc

.. autofunction:: ovnewton.add_ovstage

.. autoclass:: ovnewton.OvstageImportResult
   :members:

.. autofunction:: ovnewton.attach_ovstage

.. autoclass:: ovnewton.StageBinding
   :members:

Output reads
------------

.. autoclass:: ovnewton.SimObjectType
   :members:
   :undoc-members:

.. autoclass:: ovnewton.ObjectScope
   :members:
   :undoc-members:

.. autoclass:: ovnewton.Query
   :members:

.. autoclass:: ovnewton.ReadResult
   :members:

.. autoclass:: ovnewton.ReadGroup
   :members:

.. autoclass:: ovnewton.ReadColumn
   :members:

.. autoclass:: ovnewton.ReadColumnMeta
   :members:

.. autoclass:: ovnewton.ReadColumnCudaSync
   :members:

.. autofunction:: ovnewton.read_fields

.. autoclass:: ovnewton.ReadField
   :members:

Attribute lookup
----------------

.. autofunction:: ovnewton.attributes.list_attributes

.. autofunction:: ovnewton.attributes.get_attribute

.. autoclass:: ovnewton.attributes.UnitTerm
   :members:

.. autoclass:: ovnewton.attributes.JointLayout
   :members:

.. autoclass:: ovnewton.attributes.JointLayoutPart
   :members:

Errors
------

.. autoexception:: ovnewton.InvalidPhysicsError

.. autoexception:: ovnewton.UnsupportedPhysicsError

.. autoexception:: ovnewton.OvstageContractError
