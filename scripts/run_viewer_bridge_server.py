"""View Bridge Server"""

import argparse

from nerfactory.viewer.server.server import ZMQWebSocketBridge


def main():
    """Run the viewer bridge server"""
    parser = argparse.ArgumentParser(description="Listen for ZeroMQ commands")
    parser.add_argument("--zmq-url", "-z", type=str, nargs="?", default=None)
    parser.add_argument("--websocket-port", "-wp", type=str, nargs="?", default=None)
    args = parser.parse_args()
    bridge = ZMQWebSocketBridge(zmq_url=args.zmq_url, websocket_port=args.websocket_port)
    print(bridge)
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
