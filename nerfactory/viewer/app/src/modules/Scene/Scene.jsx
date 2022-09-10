/* eslint-disable no-restricted-syntax */
import * as THREE from 'three';

import { useContext, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { drawCamera, drawSceneBounds } from './drawing';

import SceneNode from '../../SceneNode';
import { WebSocketContext } from '../WebSocket/WebSocket';
import { subscribe_to_changes } from '../../subscriber';

const msgpack = require('msgpack-lite');

const SCENE_BOUNDS_NAME = 'Scene Bounds';
const CAMERAS_NAME = 'Training Cameras';

export function get_scene_tree() {
  let scene = null;
  let sceneTree = null;
  scene = new THREE.Scene();
  sceneTree = new SceneNode(scene);

  // add objects to the the scene tree
  const setObject = (path, object) => {
    sceneTree.find(path.concat(['<object>'])).set_object(object);
  };
  const deleteObject = (path) => {
    sceneTree.delete(path);
  };

  // Main camera
  const main_camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  setObject(['Main Camera'], main_camera);

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
      setObject([SCENE_BOUNDS_NAME], line);
    } else {
      deleteObject([SCENE_BOUNDS_NAME]);
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
          setObject([CAMERAS_NAME, key], camera);
        }
      }
      for (const key of prev) {
        // invalid so delete
        if (!curr.has(key) || current[key] === null) {
          // keys_invalid.push(key);
          deleteObject([CAMERAS_NAME, key]);
        }
      }
    } else {
      deleteObject([CAMERAS_NAME]);
    }
  };
  subscribe_to_changes(selector_fn_cameras, fn_value_cameras);

  return sceneTree;
}

// manages setting up the scene and other logic for keeping state in sync with the server
export function SceneTreeWebSocketListener() {
  const socket = useContext(WebSocketContext).socket;
  const dispatch = useDispatch();

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
  }, [socket]); // dependency to call this whenever the websocket changes
}
