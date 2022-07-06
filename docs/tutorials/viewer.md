# Real-time Viewer

We provide a real-time viewer that can be used during training or to view a model after training. The viewer is built using [ThreeJS](https://threejs.org/) and packed into a ReactJS application. This client application will connect via a websocket to a server running on your machine. Our web-based viewer is built off the following frameworks:

- a server using ZeroMQ with TCP (REQ/REP)
- a client app using ReactJS with a websocket connecting to the server

TODO(ethan): insert figure here

# Quick start

#### Running the viewer server

The viewer server runs on the same machine that you use for training. The training code will connect to the server with a lightweight TCP connection.

```
cd pyrad

# run the server on your machine
# this will run in the background
python scripts/run_viewer_server.py
```

#### Running the client app

```
cd pyrad/viewer/app
yarn install
yarn start
```

# Features

#### WebRTC

# References

We thank [Robin Deits](https://github.com/rdeits) and other contributors to the following repos, which we've started with and modified for our implementations.

- [meshcat-python](https://github.com/rdeits/meshcat-python)
- [meshcat](https://github.com/rdeits/meshcat)
