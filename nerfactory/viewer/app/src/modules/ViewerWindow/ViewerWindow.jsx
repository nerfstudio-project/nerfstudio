import * as THREE from 'three';

import React, { useContext, useEffect, useRef } from 'react';

import { IconButton } from '@mui/material';
import OpenWithIcon from '@mui/icons-material/OpenWith';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import PublicOffSharpIcon from '@mui/icons-material/PublicOffSharp';
import PublicSharpIcon from '@mui/icons-material/PublicSharp';
import Stats from 'stats.js';
import SyncOutlinedIcon from '@mui/icons-material/SyncOutlined';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls';
import { useSelector } from 'react-redux';
import WebRtcWindow from '../WebRtcWindow/WebRtcWindow';
import { WebSocketContext } from '../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

function createStats() {
  const stats = new Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0';
  stats.domElement.style.top = '0';
  return stats;
}

function TransformIcons(props) {
  const sceneTree = props.sceneTree;
  const transform_controls = sceneTree.find_object(['Transform Controls']);
  // toggle back and forth between local and global transform
  const [world, setWorld] = React.useState(true);

  const toggleLocal = () => {
    transform_controls.setSpace(world ? 'local' : 'world');
    setWorld(!world);
  };

  return (
    <div>
      <div className="ViewerWindow-iconbutton">
        <IconButton
          size="large"
          onClick={() => {
            transform_controls.setMode('translate');
          }}
        >
          {/* translate */}
          <OpenWithIcon />
        </IconButton>
      </div>
      <div className="ViewerWindow-iconbutton">
        <IconButton
          size="large"
          onClick={() => {
            transform_controls.setMode('rotate');
          }}
        >
          {/* rotate */}
          <SyncOutlinedIcon />
        </IconButton>
      </div>
      <div className="ViewerWindow-iconbutton">
        <IconButton size="large" onClick={toggleLocal}>
          {world ? <PublicSharpIcon /> : <PublicOffSharpIcon />}
        </IconButton>
      </div>
    </div>
  );
}

// manages a camera and the web rtc stream...
export default function ViewerWindow(props) {
  // eslint-disable-next-line react/prop-types
  const sceneTree = props.sceneTree;
  const scene = props.scene;

  const myRef = useRef(null);
  const websocket = useContext(WebSocketContext).socket;
  const field_of_view = useSelector(
    (state) => state.renderingState.field_of_view,
  );
  const field_of_view_ref = useRef(field_of_view);

  let cameraControls = null;
  let transformsControls = null;
  let renderer = null;
  let viewportWidth = null;
  let viewportHeight = null;
  let stats = null;

  const camera = sceneTree.find(['Main Camera', '<object>']).object;

  const getViewportWidth = () => {
    const width = myRef.current.clientWidth;
    return width - (width % 2);
  };

  const getViewportHeight = () => {
    return myRef.current.clientHeight;
  };

  const handleResize = () => {
    viewportWidth = getViewportWidth();
    viewportHeight = getViewportHeight();
    camera.aspect = viewportWidth / viewportHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewportWidth, viewportHeight);
  };

  const sendCamera = () => {
    // update the camera information in the python server
    if (websocket.readyState === WebSocket.OPEN) {
      const cmd = 'write';
      const path = 'renderingState/camera';
      const data = {
        type: cmd,
        path,
        data: camera.toJSON(),
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  // keep sending the camera often
  useEffect(() => {
    const fps = 24;
    const interval = 1000 / fps;
    const refreshIntervalId = setInterval(sendCamera, interval);
    return () => {
      clearInterval(refreshIntervalId);
    };
  }, [websocket]);

  const update = () => {
    requestAnimationFrame(update);
    handleResize();
    camera.fov = field_of_view_ref.current;
    camera.updateProjectionMatrix();
    cameraControls.update();
    renderer.render(scene, camera);
    stats.update();
  };

  // this is run once
  useEffect(() => {
    viewportWidth = getViewportWidth();
    viewportHeight = getViewportHeight();

    stats = createStats();
    myRef.current.append(stats.domElement);

    // camera = new THREE.PerspectiveCamera(
    //   field_of_view_ref.current,
    //   viewportWidth / viewportHeight,
    //   0.01,
    //   100,
    // );
    camera.position.x = 5;
    camera.position.y = -5;
    camera.position.z = 5;
    camera.up = new THREE.Vector3(0, 0, 1);

    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(viewportWidth, viewportHeight);

    myRef.current.append(renderer.domElement);
    // add camera controls
    cameraControls = new OrbitControls(camera, renderer.domElement);
    cameraControls.rotateSpeed = 2.0;
    cameraControls.zoomSpeed = 0.3;
    cameraControls.panSpeed = 0.2;
    cameraControls.target.set(0, 0, 0); // focus point of the controls
    cameraControls.autoRotate = false;
    cameraControls.enableDamping = true;
    cameraControls.dampingFactor = 1.0;
    cameraControls.update();

    transformsControls = new TransformControls(camera, renderer.domElement);
    sceneTree.set_object_from_path(['Transform Controls'], transformsControls);

    transformsControls.addEventListener('dragging-changed', (event) => {
      cameraControls.enabled = !event.value;
    });

    update();
    // cameraControls.addEventListener('change', update);
  }, []);

  // updates the field of view inside the ref to avoid rerendering so often
  useEffect(() => {
    field_of_view_ref.current = field_of_view;
  }, [field_of_view]);

  return (
    <>
      {/* the webrtc viewer needs to know the camera pose */}
      <WebRtcWindow />
      <div className="canvas-container-main" ref={myRef} />
      <div className="ViewerWindow-buttons">
        <TransformIcons sceneTree={sceneTree} />
      </div>
    </>
  );
}
