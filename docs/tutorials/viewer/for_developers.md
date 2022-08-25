# For developers

The tutorial below shows you how to host the viewer yourself and how you can use it to visualize training.

(nerfactory_bridge)=
#### Connection between nerfactory and the Bridge Server

In the center, we have the Bridge Server, which facilitates the connection between nerfactory code and the Client App. This server runs on the same machine that you are using nerfactory. It has a TCP Request/Reply (REQ/REP) connection that nerfactory can connect to with the Viewer object (left). We use [ZeroMQ](https://zeromq.org/), an open-sourced messaging library, to implement this lightweight TCP connection. The Viewer class can send commands to the Bridge Server and receive replies. The Bridge Server will either dispatch commands to the Client App via a websocket or it will return information stored in the Bridge Server state.

(bridge_client)=
#### Connection between the Bridge Server and the Client App

The connection between the Bridge Server and the Client App works with WebSockets and WebRTC.

- **WebSocket connection** - The WebSocket is used by the Bridge Server to dispatch commands coming from the nerfactory TCP connection. Commands can be used for drawing primitives, for setting the transform of objects, for the setting various properties, and more.

- **WebRTC connection** - We use WebRTC to stream images being rendered from nerfactory. The websocket connection if first used to establish the WebRTC connection. Then, the Client App constantly publishes camera pose information to the Bridge Server and stores the camera information (intrinsics and extrinsics). This information is then queried from the nerfactory code, used to render an image with some Graph, and then the image is send over the TCP connection and dispatched via WebRTC to render the stream of images.

(getting_started)=
## Getting started

(bridge)=
#### Running the Bridge Server

The viewer server runs on the same machine that you use for training. The training code will connect to the server with a lightweight TCP connection.

```
cd nerfactory

# run the server on your machine
# this will run in the background
python scripts/run_viewer_bridge_server.py

It should print out something of the form:
"ZMQWebSocketBridge using zmq_url=tcp://127.0.0.1:6000 and websocket_port=8051"
```

(client)=
#### Running the Client App

> We will host the viewer online in the future, but for now we have to run it locally.

```shell
cd nerfactory/viewer/app
```

Install npm (to install yarn) and yarn

```shell
sudo apt-get install npm
npm install --global yarn
```

Install nvm and set the node version
Install nvm with [instructions](https://heynode.com/tutorial/install-nodejs-locally-nvm/).

```shell
nvm install 17.8.0
```

Now running `node --version` in the shell should print "v17.8.0".
Install package.json dependencies and start the client viewer app

```shell
yarn install
yarn start
```

Forward port and open in your local browser. The URL takes the following form, `http://localhost:<forwarded_react_port>?localhost:<forwarded_backend_tcp_port>`. Notice the "?" character.
- http://localhost:4000?localhost:8051

Possibly edit the port in `app/.env.development` file with the following to adjust the PORT, for example.

```
BROWSER=none
FAST_REFRESH=false
HOST=0.0.0.0
PORT=4000
ESLINT_NO_DEV_ERRORS=true
```

(nerfactory)=
#### Running the nerfactory Code

You can now simply run a training job and visualize progress by enabling the viewer during training:

```bash
python scripts/run_train.py --config-name=graph_instant_ngp.yaml viewer.enable=true
```

- **Notebook demo** - See [Programming the viewer](viewer_notebook.ipynb) for an overview for how to interact with the viewer with the Viewer object from nerfactory.

(ack)=
## Acknowledgements and references

We thank [Robin Deits](https://github.com/rdeits) and other contributors to the following repos, which we've started with and modified and extended for our use.

- [meshcat-python](https://github.com/rdeits/meshcat-python)
- [meshcat](https://github.com/rdeits/meshcat)

Here are other resources that we've used and/or have found helpful while building the viewer code.

- [ThreeJS](https://threejs.org/)
- [ReactJS](https://reactjs.org/)
- [WebRTC](https://webrtc.org/) - WebRTC is a framework for real-time communication that allows two peers to send video, audio, or general data to and from each other with low latency. We've adopted WebRTC to stream rendered images to our viewer.

```

```