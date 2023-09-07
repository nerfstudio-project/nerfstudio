/* eslint-disable no-restricted-syntax */
import * as THREE from 'three';

import CameraControls from 'camera-controls';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer';

import { TransformControls } from 'three/examples/jsm/controls/TransformControls';
import { useDispatch } from 'react-redux';
import { drawCamera, drawSceneBox} from './drawing';

import { CameraHelper } from '../SidePanel/CameraPanel/CameraHelper';
import SceneNode from '../../SceneNode';
import { subscribe_to_changes } from '../../subscriber';
import { snap_to_camera } from '../SidePanel/SidePanel';

import variables from '../../index.scss';

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
      this.callbacks.forEach((callback, callbackKey) => {
        if (callbackKey === key) {
          callback(value);
        }
      });
    },
  };

  const sceneTree = new SceneNode(scene, scene_state);

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const dispatch = useDispatch();
  const BANNER_HEIGHT = parseInt(variables.bannerHeight,10);

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
  camera_controls.azimuthRotateSpeed = 0.4;
  camera_controls.polarRotateSpeed = 0.4;
  camera_controls.dollySpeed = 0.1;
  camera_controls.infinityDolly = true;
  camera_controls.smoothTime=.05;
  camera_controls.draggingSmoothTime=.05;
  camera_controls.restThreshold = .0025;
  camera_controls.saveState();

  const keyMap = [];
  const moveSpeed = 0.005;
  const upRotSpeed = 0.04;
  const sideRotSpeed = .01;
  const EPS = 0.005;


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
        {enableTransition:true}
      );

      if (keyMap.ArrowLeft === true) {
        camera_controls.rotate(sideRotSpeed, 0, {enableTransition:true});
      }
      if (keyMap.ArrowRight === true) {
        camera_controls.rotate(-sideRotSpeed, 0, {enableTransition:true});
      }
      if (keyMap.ArrowUp === true) {
        camera_controls.rotate(0, upRotSpeed, {enableTransition:true});
      }
      if (keyMap.ArrowDown === true) {
        camera_controls.rotate(0, -upRotSpeed, {enableTransition:true});
      }
    }
  }

  function translateForward(distance) {
    const to = camera_controls.getTarget().add(
      camera_controls.getTarget()
        .sub(camera_controls.getPosition())
        .normalize()
        .multiplyScalar(distance)
    );
    camera_controls.moveTo(
      to.x,
      to.y,
      to.z,
      {enableTransition:true}
    )
  }

  function translate() {
    if (keyMap.KeyD === true) {
      camera_controls.truck(moveSpeed, 0, {enableTransition:true});
    }
    if (keyMap.KeyA === true) {
      camera_controls.truck(-moveSpeed, 0, {enableTransition:true});
    }
    if (keyMap.KeyW === true) {
      translateForward(moveSpeed);
    }
    if (keyMap.KeyS === true) {
      translateForward(-moveSpeed);
    }
    if (keyMap.KeyQ === true) {
      camera_controls.truck(0, moveSpeed, {enableTransition:true});
    }
    if (keyMap.KeyE === true) {
      camera_controls.truck(0, -moveSpeed, {enableTransition:true});
    }
  }

  function moveCamera() {
    if (!scene_state.value.get('mouse_in_scene')) {
      return;
    }
    if (keyMap.Space === true) {
      camera_controls.setLookAt(0.7, -0.7, 0.3, 0, 0, 0, {enableTransition:true});
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
      {enableTransition:true}
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
