# Viewer

#### Backend (backend/)

The backend faciliates communication between Python code and the ReactJS client application.

```
cd backend
python run_zmq_server.py

It should print out something of the form:
"ZMQWebSocketBridge using zmq_url=tcp://127.0.0.1:6000 and websocket_port=8051"
```

#### Client (app/)

The ReactJS client code is in the `app` folder. First, install the most recent version of node.

```
cd app

# install yarn and package.json dependencies
npm install --global yarn
yarn install

# start the application
yarn start

# forward port and open in your local browser
# the URL takes the following form
# http://localhost:<forwarded_react_port>/localhost:<forwarded_backend_tcp_port>
http://localhost:4000/localhost:8051
```

Possibly edit the port in `app/.env.development` file with the following:

```
BROWSER=none
FAST_REFRESH=false
HOST=localhost
PORT=4000
```

#### Establishing the connection from Python

See `notebooks/visualize_viewer.ipynb` for a quick overview of how to interact with the viewer.

<hr>

#### How to use with VNC

> This section is more experimental and shouldn't be used right now. Eventually it would be nice to support an Electron app for local use instead of using a web browser (e.g., Chrome).

```
https://tigervnc.org/doc/Xvnc.html

# start vnc server
/opt/TurboVNC/bin/Xvnc :5017 -rfbport 5017 -SecurityTypes none -localsts

# run the application. TODO: get electron to work!
alias wgui='vglrun -d :0.0'
export DISPLAY=:5017
wgui chromium-browser http://localhost:3000/localhost:8051
```

The server communication code is in the `comms` folder.
