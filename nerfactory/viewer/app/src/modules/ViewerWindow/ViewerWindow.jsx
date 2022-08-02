import './ViewerWindow.css';

import * as THREE from 'three';

import React, { useContext, useEffect, useRef } from 'react';
import { useSelector } from 'react-redux';

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import WebRtcWindow from '../WebRtcWindow/WebRtcWindow';
import { WebSocketContext } from '../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

// manages a camera and the web rtc stream...
export default function ViewerWindow(props) {
  // eslint-disable-next-line react/prop-types
  const scene = props.scene;
  let cameraControls = null;
  const myRef = useRef(null);
  const websocket = useContext(WebSocketContext).socket;
  const field_of_view = useSelector(
    (state) => state.renderingState.field_of_view,
  );

  const getViewportWidth = () => {
    return window.innerWidth - (window.innerWidth % 2);
  };

  const getViewportHeight = () => {
    return window.innerHeight;
  };

  let viewportWidth = getViewportWidth();
  let viewportHeight = getViewportHeight();

  const camera = new THREE.PerspectiveCamera(
    field_of_view,
    viewportWidth / viewportHeight,
    0.01,
    100,
  );
  camera.position.x = 5;
  camera.position.y = -5;
  camera.position.z = 5;
  camera.up = new THREE.Vector3(0, 0, 1);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(viewportWidth, viewportHeight);
  renderer.domElement.style.border = '1px solid black';

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

  const update = () => {
    handleResize();
    camera.updateProjectionMatrix();
    cameraControls.update();
    requestAnimationFrame(update);
    renderer.render(scene, camera);
    sendCamera();
  };

  // similar to componentDidMount
  useEffect(() => {
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
    update();
  }, []);

  // useEffect(() => {
  //   camera.fov = field_of_view;
  //   console.log(field_of_view);
  //   camera.updateProjectionMatrix();
  // }, [field_of_view]);

  return (
    <div>
      {/* the webrtc viewer needs to know the camera pose */}
      <WebRtcWindow />
      <div className="canvas-container-main" ref={myRef} />
    </div>
  );
}
