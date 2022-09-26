/* eslint-disable no-restricted-syntax */
import * as THREE from 'three';

import { useContext, useEffect } from 'react';
import CameraControls from 'camera-controls';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer';

import { TransformControls } from 'three/examples/jsm/controls/TransformControls';
import { useDispatch } from 'react-redux';
import { drawCamera, drawSceneBounds } from './drawing';

import { CameraHelper } from '../SidePanel/CameraPanel/CameraHelper';
import SceneNode from '../../SceneNode';
import { WebSocketContext } from '../WebSocket/WebSocket';
import { subscribe_to_changes } from '../../subscriber';

const msgpack = require('msgpack-lite');

const SCENE_BOUNDS_NAME = 'Scene Bounds';
const CAMERAS_NAME = 'Training Cameras';

export function get_scene_tree() {
  const scene = new THREE.Scene();
  const sceneTree = new SceneNode(scene);

  const dispatch = useDispatch();

  // Main camera
  const main_camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  main_camera.position.x = 0.7;
  main_camera.position.y = -0.7;
  main_camera.position.z = 0.3;
  main_camera.up = new THREE.Vector3(0, 0, 1);
  sceneTree.set_object_from_path(['Cameras', 'Main Camera'], main_camera);

  sceneTree.metadata.camera = main_camera;

  // Render camera
  const render_camera = main_camera.clone();
  const render_camera_helper = new CameraHelper(render_camera, '#4eb570');
  render_camera_helper.set_visibility(false);
  sceneTree.set_object_from_path(['Cameras', 'Render Camera'], render_camera);
  sceneTree.set_object_from_path(
    ['Cameras', 'Render Camera', 'Helper'],
    render_camera_helper,
  );

  // Renderer
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  sceneTree.metadata.renderer = renderer;

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0px';
  document.body.appendChild(labelRenderer.domElement);
  sceneTree.metadata.labelRenderer = labelRenderer;

  // Camera Controls
  CameraControls.install({ THREE });

  const camera_controls = new CameraControls(main_camera, renderer.domElement);
  camera_controls.azimuthRotateSpeed = 0.3;
  camera_controls.polarRotateSpeed = 0.3;
  camera_controls.minDistance = 0.3;
  camera_controls.maxDistance = 100;

  camera_controls.dollySpeed = 0.3;
  camera_controls.saveState();

  const keyMap = [];
  const moveSpeed = 0.05;

  function moveCamera() {
    if (keyMap.ArrowLeft === true) {
      camera_controls.rotate(-0.03, 0, true);
    }
    if (keyMap.ArrowRight === true) {
      camera_controls.rotate(0.03, 0, true);
    }
    if (keyMap.ArrowUp === true) {
      camera_controls.rotate(0, -0.02, true);
    }
    if (keyMap.ArrowDown === true) {
      camera_controls.rotate(0, 0.02, true);
    }
    if (keyMap.KeyD === true) {
      camera_controls.truck(moveSpeed, 0, true);
    }
    if (keyMap.KeyA === true) {
      camera_controls.truck(-moveSpeed, 0, true);
    }
    if (keyMap.KeyW === true) {
      camera_controls.forward(moveSpeed, true);
    }
    if (keyMap.KeyS === true) {
      camera_controls.forward(-moveSpeed, true);
    }
    if (keyMap.KeyQ === true) {
      camera_controls.truck(0, -moveSpeed, true);
    }
    if (keyMap.KeyE === true) {
      camera_controls.truck(0, moveSpeed, true);
    }
  }

  function onKeyUp(event) {
    const keyCode = event.code;
    keyMap[keyCode] = false;
  }
  function onKeyDown(event) {
    const keyCode = event.code;
    keyMap[keyCode] = true;
  }

  window.addEventListener('keydown', onKeyDown, true);
  window.addEventListener('keyup', onKeyUp, true);

  window.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
      camera_controls.setTarget(0, 0, 0);
    }
  });

  sceneTree.metadata.camera_controls = camera_controls;
  sceneTree.metadata.moveCamera = moveCamera;

  // Transform Controls
  const transform_controls = new TransformControls(
    main_camera,
    renderer.domElement,
  );
  sceneTree.set_object_from_path(['Transform Controls'], transform_controls);
  transform_controls.addEventListener('dragging-changed', (event) => {
    // turn off the camera controls while transforming an object
    camera_controls.enabled = !event.value;
  });

  // if you drag the screen when the render camera is shown,
  // then snap back to the main camera
  // eslint-disable-next-line no-unused-vars
  camera_controls.addEventListener('change', (event) => {
    if (sceneTree.metadata.camera === render_camera) {
      dispatch({
        type: 'write',
        path: 'renderingState/camera_choice',
        data: 'Main Camera',
      });
    }
    // transform_controls.detach();
  });

  // Axes
  const axes = new THREE.AxesHelper(5);
  sceneTree.set_object_from_path(['Axes'], axes);

  // Grid
  const grid = new THREE.GridHelper(20, 20);
  grid.rotateX(Math.PI / 2); // rotated to xy plane
  sceneTree.set_object_from_path(['Grid'], grid);

  // Lights
  const color = 0xffffff;
  const intensity = 1;
  const light = new THREE.AmbientLight(color, intensity);
  sceneTree.set_object_from_path(['Light'], light);

  // draw scene bounds
  const selector_fn_scene_bounds = (state) => {
    return state.sceneState.sceneBounds;
  };
  const fn_value_scene_bounds = (previous, current) => {
    if (current !== null) {
      const line = drawSceneBounds(current);
      sceneTree.set_object_from_path([SCENE_BOUNDS_NAME], line);
    } else {
      sceneTree.delete([SCENE_BOUNDS_NAME]);
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
          sceneTree.set_object_from_path([CAMERAS_NAME, key], camera);
        }
      }
      for (const key of prev) {
        // invalid so delete
        if (!curr.has(key) || current[key] === null) {
          // keys_invalid.push(key);
          sceneTree.delete([CAMERAS_NAME, key]);
        }
      }
    } else {
      sceneTree.delete([CAMERAS_NAME]);
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
