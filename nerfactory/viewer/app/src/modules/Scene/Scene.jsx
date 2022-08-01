import * as THREE from 'three';

import React, { useContext, useEffect } from 'react';
import { ReactReduxContext, useDispatch, useSelector } from 'react-redux';
import { drawCameras, drawSceneBounds } from './drawing';

import { GUI } from 'dat.gui';
import { SceneNode } from '../../SceneNode';
import { WebSocketContext } from '../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

// manages setting up the scene and other logic for keeping state in sync with the server
export default function SetupScene(props) {
  let scene = null;
  let gui = null;
  let sceneTree = null;

  // the websocket context
  const socket = useContext(WebSocketContext).socket;
  const dispatch = useDispatch();

  // Scene
  scene = new THREE.Scene();

  // GUI
  gui = new GUI();
  const sceneFolder = gui.addFolder('Scene');
  sceneFolder.open();
  sceneTree = new SceneNode(scene, sceneFolder);

  // add objects to the the scene tree
  const setObject = (path, object) => {
    sceneTree.find(path.concat(['<object>'])).set_object(object);
  };

  // Axes
  const axes = new THREE.AxesHelper(5);
  setObject(['Axes'], axes);

  // Grid
  const grid = new THREE.GridHelper(20, 20);
  grid.rotateX(Math.PI / 2); // rotated to xy plane
  setObject(['Grid'], grid);

  // Lights
  const color = 0xffffff;
  const intensity = 1;
  const light = new THREE.AmbientLight(color, intensity);
  setObject(['Light'], light);

  const { store } = useContext(ReactReduxContext);

  // handle the box drawing...
  const select_box = (state) => {
    return state.sceneState.sceneBounds;
  };
  let currentSceneStateBoxValue;
  const handle_change_box = () => {
    console.log('handle_change_box');
    let previousSceneStateBoxValue = currentSceneStateBoxValue;
    currentSceneStateBoxValue = select_box(store.getState());
    if (previousSceneStateBoxValue !== currentSceneStateBoxValue && currentSceneStateBoxValue !== null) {
      const sceneBounds = currentSceneStateBoxValue;
      const line = drawSceneBounds(sceneBounds);
      setObject(['Scene Bounds'], line);
    }
  };
  store.subscribe(handle_change_box);

  // handle the camera drawing...
  const select_cameras = (state) => {
    return state.sceneState.cameras;
  };
  let currentCamerasValue;
  const handle_change_cameras = () => {
    let previousCamerasValue = currentCamerasValue;
    currentCamerasValue = select_cameras(store.getState());
    if (
      previousCamerasValue !== currentCamerasValue &&
      currentCamerasValue !== null
    ) {
      const cameras = currentCamerasValue;
      const cameraObjects = drawCameras(cameras);
      for (const [key, camera] of Object.entries(cameraObjects)) {
        // console.log(key);
        // console.log(camera);
        setObject(['Cameras', key], camera);
      }

      // for (int i = 0; i < currentCamerasValue.length; i += 1) {
      // }
      // setObject(['Cameras'], cameras);

      // TODO(ethan): draw the cameras here!!
      // setObject(['Cameras', "..."], ...);
    }
  };
  store.subscribe(handle_change_cameras);

  useEffect(() => {
    socket.addEventListener('message', (originalCmd) => {
      // set the remote description when the offer is received
      const cmd = msgpack.decode(new Uint8Array(originalCmd.data));
      if (cmd.type === 'write') {
        // write to the store
        dispatch({
          type: 'write',
          path: cmd.path,
          data: cmd.data,
        });
      }
    });
  }, []); // empty dependency array means only run once

  return scene;
}
