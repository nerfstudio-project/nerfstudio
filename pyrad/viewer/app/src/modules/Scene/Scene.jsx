import * as THREE from 'three';
import React, { useContext } from 'react';
import { GUI } from 'dat.gui';
import { SceneNode } from '../../SceneNode';
import { WebSocketContext } from '../WebSocket/WebSocket';
import { split_path } from '../../utils';

const msgpack = require('msgpack-lite');

function findCameraObjectUnderObject3D(object) {
  // seach the tree for the camera
  if (object instanceof THREE.Camera) {
    return object;
  }
  for (const name of Object.keys(object.children)) {
    return findCameraObjectUnderObject3D(object.children[name]);
  }
  return null;
}

// manages setting up the scene and other logic for keeping state in sync with the server
//
export default function SetupScene(props) {
  const state = {
    scene: null,
    controls: null,
    renderer: null,
    gui: null,
    scene_tree: null,
    viewport_width: null,
    viewport_height: null,
  };

  let scene = null;
  let gui = null;
  let scene_tree = null;

  // the websocket context
  let websocket = useContext(WebSocketContext).socket;

  const send_training_state_over_websocket = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_training_state';
      const path = 'Training State';
      const data = {
        type: cmd,
        path,
        training_state: value,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const send_output_type_over_websocket = (value) => {
    /* update the output option in the python server
                            if the user changes selection */
    if (websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_output_type';
      const path = 'Output Type';
      const data = {
        type: cmd,
        path,
        output_type: value,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const send_min_resolution_over_websocket = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_min_resolution';
      const path = 'Min Resolution';
      const data = {
        type: cmd,
        path,
        min_resolution: value,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const send_max_resolution_over_websocket = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_max_resolution';
      const path = 'Max Resolution';
      const data = {
        type: cmd,
        path,
        max_resolution: value,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const set_object = (path, object) => {
    scene_tree.find(path.concat(['<object>'])).set_object(object);
  };

  const set_transform = (path, matrix) => {
    scene_tree.find(path).set_transform(matrix);
  };

  const delete_path = (path) => {
    if (path.length === 0) {
      console.error('Deleting the entire scene is not implemented.');
    } else {
      scene_tree.delete(path);
    }
  };

  const set_property = (path, property, value) => {
    scene_tree.find(path).set_property(property, value);
  };

  const handle_command = (cmd) => {
    // convert binary serialization format back to JSON
    cmd = msgpack.decode(new Uint8Array(cmd));
    console.log(cmd);

    // three js scene commands
    if (cmd.type === 'set_object') {
      const path = split_path(cmd.path);
      set_object_from_json(path, cmd.object);
    } else if (cmd.type === 'set_transform') {
      const path = split_path(cmd.path);
      set_transform(path, cmd.matrix);
    } else if (cmd.type === 'delete') {
      const path = split_path(cmd.path);
      delete_path(path);
    } else if (cmd.type === 'set_property') {
      const path = split_path(cmd.path);
      set_property(path, cmd.property, cmd.value);
    }
  };

  // Scene
  scene = new THREE.Scene();

  // GUI
  gui = new GUI();
  const sceneFolder = gui.addFolder('Scene');
  sceneFolder.open();
  scene_tree = new SceneNode(scene, sceneFolder);

  // Axes
  const axes = new THREE.AxesHelper(5);
  set_object(['Axes'], axes);

  // Grid
  const grid = new THREE.GridHelper(20, 20);
  grid.rotateX(Math.PI / 2); // rotated to xy plane
  set_object(['Grid'], grid);

  // Lights
  const color = 0xffffff;
  const intensity = 1;
  const light = new THREE.AmbientLight(color, intensity);
  set_object(['Light'], light);

  console.log(scene);

  return scene;
}
