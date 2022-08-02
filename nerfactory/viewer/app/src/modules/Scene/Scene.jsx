import './Scene.css';

/* eslint-disable no-restricted-syntax */
import * as THREE from 'three';

import { useContext, useEffect } from 'react';
import { useDispatch } from 'react-redux';

import { GUI } from 'dat.gui';
import SceneNode from '../../SceneNode';
import { subscribe_to_changes } from '../../subscriber';
import { WebSocketContext } from '../WebSocket/WebSocket';
import { drawCamera, drawSceneBounds } from './drawing';

const msgpack = require('msgpack-lite');

// manages setting up the scene and other logic for keeping state in sync with the server
export default function SetupScene() {
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
  gui.domElement.id = 'datgui';
  const sceneFolder = gui.addFolder('Scene');
  sceneFolder.open();
  sceneTree = new SceneNode(scene, sceneFolder);

  // add objects to the the scene tree
  const setObject = (path, object) => {
    sceneTree.find(path.concat(['<object>'])).set_object(object);
  };
  const deleteObject = (path) => {
    sceneTree.delete(path);
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

  // draw scene bounds
  const selector_fn_scene_bounds = (state) => {
    return state.sceneState.sceneBounds;
  };
  const fn_value_scene_bounds = (previous, current) => {
    if (current !== null) {
      const line = drawSceneBounds(current);
      setObject(['Scene Bounds'], line);
    } else {
      deleteObject(['Scene Bounds']);
    }
  };
  subscribe_to_changes(selector_fn_scene_bounds, fn_value_scene_bounds);

  // draw camera
  // NOTE: this has some issues right now! it won't
  // update the camera on an individual change w/o deleting first
  const selector_fn_cameras = (state) => {
    return state.sceneState.cameras;
  };
  const fn_value_cameras = (previous, current) => {
    if (current !== null) {
      let prev = new Set();
      if (previous !== null) {
        prev = new Set(Object.keys(previous));
      }
      const curr = new Set(Object.keys(current));
      // valid if in current but not previous
      // invalid if in previous but not current
      for (const key of curr) {
        // valid so draw
        if (!prev.has(key)) {
          // keys_valid.push(key);
          const json = current[key];
          const camera = drawCamera(json);
          setObject(['Cameras', key], camera);
        }
      }
      for (const key of prev) {
        // invalid so delete
        if (!curr.has(key) || current[key] === null) {
          // keys_invalid.push(key);
          deleteObject(['Cameras', key]);
        }
      }
    } else {
      deleteObject(['Cameras']);
    }
  };
  subscribe_to_changes(selector_fn_cameras, fn_value_cameras);

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
