# (Legacy Viewer) Local Server

**Note:** this doc only applies to the legacy version of the viewer, which was the default in in Nerfstudio versions `<=0.3.4`. It was deprecated starting Nerfstudio version `1.0.0`, where it needs to be opted into via the `--vis viewer_legacy` argument.

---

If you are unable to connect to `https://viewer.nerf.studio`, want to use Safari, or want to develop the viewer codebase, you can launch your own local viewer.

## Installing Dependencies

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
Install package.json dependencies and start the client viewer app:

```shell
yarn install
```

## Launch the web client

From the `nerfstudio/viewer/app` folder, run:

```shell
yarn start
```

The local webserver runs on port 4000 by default,
so when `ns-train` is running, you can connect to the viewer locally at
[http://localhost:4000/?websocket_url=ws://localhost:7007](http://localhost:4000/?websocket_url=ws://localhost:7007)

## FAQ

### Engine node incompatible

While running `yarn install`, you run into: `The engine "node" is incompatible with this module.`

**Solution**:

Install nvm with instructions at [instructions](https://heynode.com/tutorial/install-nodejs-locally-nvm/).

```shell
nvm install 17.8.0
```

If you cannot install nvm, try ignoring the engines

```shell
yarn install --ignore-engines
```
