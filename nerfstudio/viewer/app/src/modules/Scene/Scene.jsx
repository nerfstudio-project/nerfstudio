/* eslint-disable no-restricted-syntax */
import * as THREE from 'three';

import { useContext, useEffect } from 'react';
import CameraControls from 'camera-controls';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer';

import { TransformControls } from 'three/examples/jsm/controls/TransformControls';
import { useDispatch } from 'react-redux';
import { drawCamera, drawSceneBox } from './drawing';

import { CameraHelper } from '../SidePanel/CameraPanel/CameraHelper';
import SceneNode from '../../SceneNode';
import { WebSocketContext } from '../WebSocket/WebSocket';
import { subscribe_to_changes } from '../../subscriber';
import { snap_to_camera } from '../SidePanel/SidePanel';

const msgpack = require('msgpack-lite');

const SCENE_BOX_NAME = 'Scene Box';
const CAMERAS_NAME = 'Training Cameras';

export function get_scene_tree() {
  const scene = new THREE.Scene();

  const scene_state = {
    value: new Map(),
    callbacks: [],
    addCallback(callback, key) {
      this.callbacks.push([callback, key]);
    },
    setValue(key, value) {
      this.value.set(key, value);
      this.callbacks.forEach((cbk) => {
        const i = cbk[1];
        const callback = cbk[0];
        return callback(this.value.get(i));
      });
    },
  };

  const sceneTree = new SceneNode(scene, scene_state);

  const dispatch = useDispatch();
  const BANNER_HEIGHT = 50;

  // Main camera
  const main_camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  const start_position = new THREE.Vector3(0.7, -0.7, 0.3);
  main_camera.position.set(
    start_position.x,
    start_position.y,
    start_position.z,
  );
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
  camera_controls.dollySpeed = 0.1;
  camera_controls.infinityDolly = true;
  camera_controls.saveState();

  const keyMap = [];
  const moveSpeed = 0.008;
  const rotSpeed = 0.015;
  const EPS = 0.01;

  function rotate() {
    if (
      keyMap.ArrowLeft ||
      keyMap.ArrowRight ||
      keyMap.ArrowUp ||
      keyMap.ArrowDown
    ) {
      const curTar = camera_controls.getTarget();
      const curPos = camera_controls.getPosition();
      const diff = curTar.sub(curPos).clampLength(0, EPS);
      camera_controls.setTarget(
        curPos.x + diff.x,
        curPos.y + diff.y,
        curPos.z + diff.z,
      );

      if (keyMap.ArrowLeft === true) {
        camera_controls.rotate(rotSpeed, 0, true);
      }
      if (keyMap.ArrowRight === true) {
        camera_controls.rotate(-rotSpeed, 0, true);
      }
      if (keyMap.ArrowUp === true) {
        camera_controls.rotate(0, rotSpeed / 1.5, true);
      }
      if (keyMap.ArrowDown === true) {
        camera_controls.rotate(0, -rotSpeed / 1.5, true);
      }
    }
  }

  function translate() {
    if (keyMap.KeyD === true) {
      camera_controls.truck(moveSpeed, 0, true);
    }
    if (keyMap.KeyA === true) {
      camera_controls.truck(-moveSpeed, 0, true);
    }
    if (keyMap.KeyW === true) {
      const curPos = camera_controls.getPosition();
      const newTar = camera_controls.getTarget();
      const newDiff = newTar
        .sub(curPos)
        .normalize()
        .multiplyScalar(curPos.length());
      camera_controls.setTarget(
        curPos.x + newDiff.x,
        curPos.y + newDiff.y,
        curPos.z + newDiff.z,
      );
      camera_controls.dolly(moveSpeed, true);
    }
    if (keyMap.KeyS === true) {
      camera_controls.dolly(-moveSpeed, true);
    }
    if (keyMap.KeyQ === true) {
      camera_controls.truck(0, moveSpeed, true);
    }
    if (keyMap.KeyE === true) {
      camera_controls.truck(0, -moveSpeed, true);
    }
  }

  function moveCamera() {
    if (!scene_state.value.get('mouse_in_scene')) {
      return;
    }
    translate();
    rotate();
  }

  function onKeyUp(event) {
    const keyCode = event.code;
    keyMap[keyCode] = false;
  }
  function onKeyDown(event) {
    const keyCode = event.code;
    keyMap[keyCode] = true;
  }

  function checkVisibility(camera) {
    let curr = camera;
    while (curr !== null) {
      if (!curr.visible) return false;
      curr = curr.parent;
    }
    return true;
  }

  window.addEventListener('keydown', onKeyDown, true);
  window.addEventListener('keyup', onKeyUp, true);

  window.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
      camera_controls.setLookAt(0.7, -0.7, 0.3, 0, 0, 0);
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

  // draw scene box
  const selector_fn_scene_box = (state) => {
    return state.sceneState.sceneBox;
  };
  const fn_value_scene_box = (previous, current) => {
    if (current !== null) {
      const line = drawSceneBox(current);
      sceneTree.set_object_from_path([SCENE_BOX_NAME], line);
    } else {
      sceneTree.delete([SCENE_BOX_NAME]);
    }
  };
  subscribe_to_changes(selector_fn_scene_box, fn_value_scene_box);

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
          const camera = drawCamera(json, key);
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

  // Check for clicks on training cameras
  const mouseVector = new THREE.Vector2();
  const raycaster = new THREE.Raycaster();
  const size = new THREE.Vector2();
  let selectedCam = null;

  let drag = false;
  const onMouseDown = () => {
    drag = false;
    const curPos = camera_controls.getPosition();
    const newTar = camera_controls.getTarget();
    const newDiff = newTar
      .sub(curPos)
      .normalize()
      .multiplyScalar(curPos.length());
    camera_controls.setTarget(
      curPos.x + newDiff.x,
      curPos.y + newDiff.y,
      curPos.z + newDiff.z,
    );
  };

  const onMouseMove = (e) => {
    drag = true;

    sceneTree.metadata.renderer.getSize(size);
    mouseVector.x = 2 * (e.clientX / size.x) - 1;
    mouseVector.y = 1 - 2 * ((e.clientY - BANNER_HEIGHT) / size.y);

    const mouse_in_scene = !(
      mouseVector.x > 1 ||
      mouseVector.x < -1 ||
      mouseVector.y > 1 ||
      mouseVector.y < -1
    );

    scene_state.setValue('mouse_x', mouseVector.x);
    scene_state.setValue('mouse_y', mouseVector.y);
    scene_state.setValue('mouse_in_scene', mouse_in_scene);

    const camerasParent = sceneTree.find_no_create([CAMERAS_NAME]);
    if (camerasParent === null) {
      return;
    }
    const cameras = Object.values(camerasParent.children).map(
      (obj) => obj.object.children[0].children[1],
    );

    if (
      mouseVector.x > 1 ||
      mouseVector.x < -1 ||
      mouseVector.y > 1 ||
      mouseVector.y < -1
    ) {
      if (selectedCam !== null) {
        selectedCam.material.color = new THREE.Color(1, 1, 1);
        selectedCam = null;
      }
      return;
    }

    raycaster.setFromCamera(mouseVector, sceneTree.metadata.camera);
    const intersections = raycaster.intersectObjects(cameras, true);

    if (selectedCam !== null) {
      selectedCam.material.color = new THREE.Color(1, 1, 1);
      selectedCam = null;
    }
    const filtered_intersections = intersections.filter((isect) =>
      checkVisibility(isect.object),
    );
    if (filtered_intersections.length > 0) {
      selectedCam = filtered_intersections[0].object;
      selectedCam.material.color = new THREE.Color(0xfab300);
    }
  };

  const onMouseUp = () => {
    if (drag === true || !scene_state.value.get('mouse_in_scene')) {
      return;
    }
    if (selectedCam !== null) {
      const clickedCam = sceneTree.find_object_no_create([
        CAMERAS_NAME,
        selectedCam.name,
      ]);
      snap_to_camera(sceneTree, sceneTree.metadata.camera, clickedCam.matrix);
    }
  };
  window.addEventListener('mousedown', onMouseDown, false);
  window.addEventListener('mousemove', onMouseMove, false);
  window.addEventListener('mouseup', onMouseUp, false);
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
