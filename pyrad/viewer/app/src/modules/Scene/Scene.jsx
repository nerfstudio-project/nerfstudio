import * as THREE from 'three';

import { Leva, button, buttonGroup, useControls } from 'leva';
import React, {
  Component,
  MutableRefObject,
  useContext,
  useEffect,
  useRef,
} from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { ExtensibleObjectLoader } from '../../ExtensibleObjectLoader';
import { GUI } from 'dat.gui';
import { SceneNode } from '../../SceneNode';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import WebRtcWindow from '../WebRtcWindow/WebRtcWindow';
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

  // the websocket context
  state.websocket = useContext(WebSocketContext).socket;
  // state.pc = useContext(WebRtcContext).pc;
  // console.log('WebRtcContext', state.pc);

  const handleResize = () => {
    state.viewport_width = getViewportWidth();
    state.viewport_height = getViewportHeight();
    state.camera_main.aspect = state.viewport_width / state.viewport_height;
    state.camera_main.updateProjectionMatrix();
    state.renderer_main.setSize(state.viewport_width, state.viewport_height);
  };

  const update = () => {
    requestAnimationFrame(update);
    handleResize();
    const camera_main = findCameraObjectUnderObject3D(
      state.scene_tree.find(['Cameras', 'Main Camera']).object,
    );
    camera_main.updateProjectionMatrix();
    if (state.controls_main != null) {
      state.controls_main.update();
    }
    state.renderer_main.render(state.scene, camera_main);
    send_camera_over_websocket();
  };

  const send_camera_over_websocket = () => {
    /* update the camera information in the python server
                            if the websocket is connected */
    // console.log("send_camera_over_websocket");
    if (state.websocket.readyState === WebSocket.OPEN) {
      // update the camera information in the python server
      const camera_main = findCameraObjectUnderObject3D(
        state.scene_tree.find(['Cameras', 'Main Camera']).object,
      );
      // console.log(camera_main.toJSON());
      const cmd = 'set_object';
      const path = 'Cameras/Main Camera';
      const data = {
        type: cmd,
        path,
        object: camera_main.toJSON(),
      };
      const message = msgpack.encode(data);
      state.websocket.send(message);
    }
  };

  const send_training_state_over_websocket = (value) => {
    if (state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_training_state';
      const path = 'Training State';
      const data = {
        type: cmd,
        path,
        training_state: value,
      };
      const message = msgpack.encode(data);
      state.websocket.send(message);
    }
  };

  const send_output_type_over_websocket = (value) => {
    /* update the output option in the python server
                            if the user changes selection */
    if (state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_output_type';
      const path = 'Output Type';
      const data = {
        type: cmd,
        path,
        output_type: value,
      };
      const message = msgpack.encode(data);
      state.websocket.send(message);
    }
  };

  const send_min_resolution_over_websocket = (value) => {
    if (state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_min_resolution';
      const path = 'Min Resolution';
      const data = {
        type: cmd,
        path,
        min_resolution: value,
      };
      const message = msgpack.encode(data);
      state.websocket.send(message);
    }
  };

  const send_max_resolution_over_websocket = (value) => {
    if (state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_max_resolution';
      const path = 'Max Resolution';
      const data = {
        type: cmd,
        path,
        max_resolution: value,
      };
      const message = msgpack.encode(data);
      state.websocket.send(message);
    }
  };

  const set_object = (path, object) => {
    if (!(object instanceof THREE.Camera)) {
      state.scene_tree.find(path.concat(['<object>'])).set_object(object);
    }
  };

  const set_object_from_json = (path, object_json) => {
    const loader = new ExtensibleObjectLoader();
    loader.parse(object_json, (obj) => {
      if (
        obj.geometry !== undefined &&
        obj.geometry.type === 'BufferGeometry'
      ) {
        if (
          obj.geometry.attributes.normal === undefined ||
          obj.geometry.attributes.normal.count === 0
        ) {
          obj.geometry.computeVertexNormals();
        }
      }
      obj.castShadow = true;
      obj.receiveShadow = true;
      set_object(path, obj);
    });
  };

  const set_transform = (path, matrix) => {
    state.scene_tree.find(path).set_transform(matrix);
  };

  const delete_path = (path) => {
    if (path.length === 0) {
      console.error('Deleting the entire scene is not implemented.');
    } else {
      state.scene_tree.delete(path);
    }
  };

  const set_property = (path, property, value) => {
    state.scene_tree.find(path).set_property(property, value);
    // TODO(ethan): handle this issue
    if (path[0] === 'Background') {
      // The background is not an Object3d, so needs a little help.
      state.scene_tree.find(path).on_update();
    }
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
    } else if (cmd.type === 'set_output_options') {
      // setOutputOptions(cmd.output_options);
      // props.setControls({ output_options: 1 });
    }

    // possibly update the camera information in the python server
    // send_camera_over_websocket();
  };

  //   const getViewportWidth = () => {
  //     return window.innerWidth - (window.innerWidth % 2);
  //   };

  //   const getViewportHeight = () => {
  //     return window.innerHeight;
  //   };

  // Get size of screen
  //   state.viewport_width = getViewportWidth();
  //   state.viewport_height = getViewportHeight();

  // Scene
  state.scene = new THREE.Scene();

  // GUI
  state.gui = new GUI();
  const sceneFolder = state.gui.addFolder('Scene');
  sceneFolder.open();
  state.scene_tree = new SceneNode(state.scene, sceneFolder);

  // Axes display
  const axes = new THREE.AxesHelper(5);
  set_object(['Axes'], axes);

  // Grid
  const grid = new THREE.GridHelper(20, 20);
  grid.rotateX(Math.PI / 2); // now on xy plane
  set_object(['Grid'], grid);

  // Lights
  const color = 0xffffff;
  const intensity = 1;
  const light = new THREE.AmbientLight(color, intensity);
  set_object(['Light'], light);

  return [state.scene];
}
