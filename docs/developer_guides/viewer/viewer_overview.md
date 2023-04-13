# Viewer overview

> We have a real-time web viewer that requires no installation. It's available at [https://viewer.nerf.studio/](https://viewer.nerf.studio/), where you can connect to your training job. If you want to understand how the viewer works and contribute to it's features, this section is for you!

The viewer is built using [ThreeJS](https://threejs.org/) and packaged into a [ReactJS](https://reactjs.org/) application. This client viewer application will connect via a websocket to a server running on your machine.

## Installing and running locally

```shell
cd nerfstudio/viewer/app
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

The local webserver runs on port 4000 by default,
so when `ns-train` is running, you can connect to the viewer locally at
[http://localhost:4000/?websocket_url=ws://localhost:7007](http://localhost:4000/?websocket_url=ws://localhost:7007)

## Acknowledgements and references

We thank the authors and contributors to the following repos, which we've started, used, and modified for our use-cases.

- [meshcat-python](https://github.com/rdeits/meshcat-python) - made by [Robin Deits](https://github.com/rdeits)
- [meshcat](https://github.com/rdeits/meshcat) - made by [Robin Deits](https://github.com/rdeits)
- [ThreeJS](https://threejs.org/)
- [ReactJS](https://reactjs.org/)

## FAQ

#### Engine node incompatible

While running `yarn install`, you run into: `The engine "node" is incompatible with this module.`

**Solution**:

Install nvm with instructions at [instructions](https://heynode.com/tutorial/install-nodejs-locally-nvm/).

```shell
nvm install 17.8.0
```

If you cannot install nvm, try ignoring the engines

```
yarn install --ignore-engines
```
