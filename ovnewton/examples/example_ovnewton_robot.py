# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render several uncontrolled Newton Ant robots falling onto a ground plane."""

import argparse
import os
import tempfile
from pathlib import Path

from .example_ovnewton_basic import Example


def _newton_asset(filename: str) -> Path:
    """Return an asset from the imported Newton wheel or editable checkout."""
    from newton.examples import get_asset

    path = Path(get_asset(filename))
    if not path.is_file():
        raise FileNotFoundError(f"Newton robot asset is unavailable: {path}")
    return path


def _write_scene(path: Path, robot_count: int, drop_height: float) -> None:
    """Write a small render scene that references Newton's Ant asset."""
    asset = _newton_asset("ant.usda").as_posix()
    center = 0.5 * (robot_count - 1)
    robots = "\n".join(
        f'''    def Xform "Robot_{index}" (prepend references = @{asset}@</ant>)
    {{
        double3 xformOp:translate = ({2.5 * (index - center)}, 0, {drop_height + 0.75 * (index % 2)})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}'''
        for index in range(robot_count)
    )
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    def Plane "Ground" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {{
        uniform token axis = "Z"
        double width = 20
        double length = 20
        bool doubleSided = 1
        color3f[] primvars:displayColor = [(0.35, 0.38, 0.42)]
    }}

    def PhysicsCollisionGroup "GroundCollisionGroup" (
        prepend apiSchemas = ["CollectionAPI:colliders"]
    )
    {{
        uniform token collection:colliders:expansionRule = "expandPrims"
        prepend rel collection:colliders:includes = </World/Ground>
    }}

{robots}

    def DomeLight "DomeLight"
    {{
        float inputs:intensity = 100
    }}

    def DistantLight "Sun"
    {{
        float inputs:intensity = 600
        quatd xformOp:orient = (0.92388, 0.38268, 0, 0)
        token[] xformOpOrder = ["xformOp:orient"]
    }}

    def Camera "Camera"
    {{
        float2 clippingRange = (0.1, 1000)
        float focalLength = 0.4
        float horizontalAperture = 0.36
        float verticalAperture = 0.2025
        double3 xformOp:translate = (0, -14, 6)
        float3 xformOp:rotateXYZ = (72, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}
}}

def "Render"
{{
    def RenderProduct "Camera"
    {{
        rel camera = </World/Camera>
        rel orderedVars = </Render/Camera/LdrColor>
        int2 resolution = (1280, 720)

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}
}}
''',
        encoding="utf-8",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the robot example."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--drop-height", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0", help="Warp device used for simulation")
    parser.add_argument(
        "--solver",
        choices=("xpbd", "mujoco", "vbd"),
        default="xpbd",
        help="Newton solver used to advance the scene",
    )
    parser.add_argument("--num-frames", type=int, default=120, help="Frames rendered in headless mode")
    parser.add_argument("--output-dir", default="output", help="Directory used in headless mode")
    parser.add_argument("--headless", action="store_true", help="Skip the viewer and save PNG frames")
    parser.add_argument("--save-png", action="store_true", help="Save rendered frames as PNG files")
    parser.add_argument("--timing", action="store_true", help="Print initialization and frame timings")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error("--num-frames must be at least 1")
    if args.drop_height <= 0.0:
        parser.error("--drop-height must be positive")
    if args.num_robots < 1:
        parser.error("--num-robots must be at least 1")
    if args.headless:
        args.save_png = True
    if args.save_png:
        os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ovnewton-robot-") as directory:
        scene = Path(directory) / "ant_robots.usda"
        _write_scene(scene, args.num_robots, args.drop_height)
        args.stage = str(scene)
        args.no_render = False
        example = Example(args)
        try:
            example.run()
            example.test_final()
        finally:
            example.close()


if __name__ == "__main__":
    main()
