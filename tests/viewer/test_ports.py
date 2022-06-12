from __future__ import absolute_import, division, print_function

import unittest
import os
import subprocess
import sys

import meshcat
import meshcat.geometry as g


class TestPortScan(unittest.TestCase):
    """
    Test that the ZMQ server can correctly handle its default ports 
    already being in use.
    """

    def setUp(self):

        # the blocking_vis will take up the default fileserver and ZMQ ports
        self.blocking_vis = meshcat.Visualizer()

        # this should still work, by chosing a new port
        self.vis = meshcat.Visualizer()

        if "CI" in os.environ:
            port = self.vis.url().split(":")[-1].split("/")[0]
            self.dummy_proc = subprocess.Popen([sys.executable, "-m", "meshcat.tests.dummy_websocket_client", str(port)])
        else:
            self.vis.open()
            self.dummy_proc = None

        self.vis.wait()

    def runTest(self):
        v = self.vis["shapes"]
        v["cube"].set_object(g.Box([0.1, 0.2, 0.3]))

    def tearDown(self):
        if self.dummy_proc is not None:
            self.dummy_proc.kill()
