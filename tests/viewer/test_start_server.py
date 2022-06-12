from __future__ import absolute_import, division, print_function

import unittest

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

class TestStartZmqServer(unittest.TestCase):
    """
    Test the StartZmqServerAsSubprocess method.
    """

    def test_default_args(self):
        proc, zmq_url, web_url = start_zmq_server_as_subprocess()
        self.assertIn("127.0.0.1", web_url)

    def test_ngrok(self):
        proc, zmq_url, web_url = start_zmq_server_as_subprocess( server_args=["--ngrok_http_tunnel"])
        self.assertIsNotNone(web_url)
        self.assertNotIn("127.0.0.1", web_url)
