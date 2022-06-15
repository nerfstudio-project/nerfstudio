# Viewer

The ReactJS code is in the `app` folder.

#### Installation

Install the most recent version of node.

```
npm install --global yarn
yarn install
yarn start

# Forward port and open
# http location / tcp location
http://localhost:13000/localhost:18051
```

Create a .env file with the following

```bash
BROWSER=none
FAST_REFRESH=false
HOST=localhost
PORT=4000

# Forward port and open
# http location / tcp location
http://localhost:4000/localhost:18051
```

#### How to use with VNC

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
