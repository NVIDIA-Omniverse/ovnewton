# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from importlib.metadata import version as package_version

project = "ovnewton"
# Sphinx adds the trailing period when rendering this value.
# Including it here would produce a double period.
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA"

installed_version = package_version("ovnewton")
release = version = os.environ.get("OVNEWTON_DOCS_VERSION", installed_version)

docs_commit = os.environ.get("OVNEWTON_DOCS_COMMIT")
if docs_commit is None:
    try:
        from ovnewton._build_info import COMMIT_SHA
    except ModuleNotFoundError as error:
        if error.name != "ovnewton._build_info":
            raise
        docs_commit = "source checkout"
    else:
        docs_commit = COMMIT_SHA

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"

myst_enable_extensions = ["colon_fence", "substitution"]
myst_substitutions = {
    "ovnewton_commit": docs_commit,
    "ovnewton_version": release,
}

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "collapse_navigation": False,
    "github_url": "https://github.com/NVIDIA-Omniverse/ovnewton",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/ovnewton/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
}

exclude_patterns = ["_build"]
