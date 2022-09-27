#!/usr/bin/env python
"""View Bridge Server"""
import dcargs

from nerfactory.viewer.server.server import run_viewer_bridge_server

if __name__ == "__main__":
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(run_viewer_bridge_server)
