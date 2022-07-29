import * as THREE from 'three';

import React, { useContext, useEffect } from 'react';
import { ReactReduxContext, useDispatch, useSelector } from 'react-redux';

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
  const dispatch = useDispatch();

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

  const { store } = useContext(ReactReduxContext);
  

  // handle the box drawing...
  const select_box = (state) => {
    return state.sceneState.sceneBounds;
  }
  let currentSceneStateBoxValue;
  const handle_change_box = () => {
    let previousSceneStateBoxValue = currentSceneStateBoxValue
    currentSceneStateBoxValue = select_box(store.getState());
    if (previousSceneStateBoxValue !== currentSceneStateBoxValue) {
      console.log("box changed!");
      console.log("draw the box!");
      // console.log(currentSceneStateBoxValue);

      const box = currentSceneStateBoxValue;
      console.log(box);

    //   const box = {
    //     "type": "aabb",
    //     "min_point": [
    //         -1,
    //         -1,
    //         -1
    //     ],
    //     "max_point": [
    //         1,
    //         1,
    //         1
    //     ]
    // }

      const material = new THREE.LineBasicMaterial( { color: 0x0000ff } );
      const w = 1.0;
      let aaa = new THREE.Vector3( w, w, w );
      let aab = new THREE.Vector3( w, w, -w );
      let aba = new THREE.Vector3( w, -w, w );
      let baa = new THREE.Vector3( -w, w, w );
      let abb = new THREE.Vector3( w, -w, -w );
      let bba = new THREE.Vector3( -w, -w, w );
      let bab = new THREE.Vector3( -w, w, -w );
      let bbb = new THREE.Vector3( -w, -w, -w );
      let camera_points = [aaa, aab, aaa, aba, aab, abb, aba, abb];
      camera_points = camera_points.concat([baa, bab, baa, bba, bab, bbb, bba, bbb]);
      camera_points = camera_points.concat([aaa, baa, aab, bab, aba, bba, abb, bbb]);

      console.log(camera_points);
      
      let max_point = new THREE.Vector3(...box["max_point"]);
      let min_point = new THREE.Vector3(...box["min_point"]);

      let lengths = max_point.clone();
      lengths.sub(min_point);

      let scalar = lengths.clone();
      scalar.divide(new THREE.Vector3(2.0, 2.0, 2.0));

      let offset = min_point.clone();
      offset.add(scalar);
      for (let i = 0; i < camera_points.length; i++) {
        camera_points[i] = (camera_points[i].multiply(scalar)).add(offset);
      }

      const geometry = new THREE.BufferGeometry().setFromPoints( camera_points );
      const line = new THREE.LineSegments( geometry, material );

      set_object(['Scene Bounds'], line);
    }
  }
  store.subscribe(handle_change_box);

  // handle the camera drawing...
  const select_cameras = (state) => {
    return state.sceneState.cameras;
  }
  let currentCamerasValue;
  const handle_change_cameras = () => {
    let previousCamerasValue = currentCamerasValue
    currentCamerasValue = select_cameras(store.getState());
    if (previousCamerasValue !== currentCamerasValue && currentCamerasValue !== null) {
      console.log("cameras changed!");
      console.log("draw the cameras!");
      console.log(currentCamerasValue);

      // TODO(ethan): draw the cameras here!!
      // set_object(['Cameras', "..."], ...);
    }
  }
  store.subscribe(handle_change_cameras);


  useEffect(() => {
    websocket.addEventListener('message', (originalCmd) => {
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
