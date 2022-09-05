import * as THREE from 'three';

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import Stats from 'stats.js';
import { useSelector } from 'react-redux';
import React, { useContext, useEffect, useRef } from 'react';
import { WebSocketContext } from '../WebSocket/WebSocket';
import WebRtcWindow from '../WebRtcWindow/WebRtcWindow';


const msgpack = require('msgpack-lite');

function createStats() {
  const stats = new Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0';
  stats.domElement.style.top = '0';
  return stats;
}

// manages a camera and the web rtc stream...
export default function ViewerWindow(props) {
  // eslint-disable-next-line react/prop-types
  const scene = props.scene;

  const myRef = useRef(null);
  const websocket = useContext(WebSocketContext).socket;
  const field_of_view = useSelector(
    (state) => state.renderingState.field_of_view,
  );
  const field_of_view_ref = useRef(field_of_view);
  
  const camera = useRef(null);
  let cameraControls = null;
  let renderer = null;
  let viewportWidth = null;
  let viewportHeight = null;
  let stats = null;

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
    camera.current.aspect = viewportWidth / viewportHeight;
    camera.current.updateProjectionMatrix();
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
        data: camera.current.toJSON(),
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  // keep sending the camera often
  useEffect(() => {
    const fps = 60;
    const interval = 1000 / fps;
    const refreshIntervalId = setInterval(sendCamera, interval);
    return () => {
      clearInterval(refreshIntervalId);
    };
  }, [websocket]);

  const update = () => {
    requestAnimationFrame(update);
    handleResize();
    camera.current.fov = field_of_view_ref.current;
    camera.current.updateProjectionMatrix();
    cameraControls.update();
    renderer.render(scene, camera.current);
    stats.update();
  };

  // this is run once
  useEffect(() => {
    viewportWidth = getViewportWidth();
    viewportHeight = getViewportHeight();

    stats = createStats();
    myRef.current.append(stats.domElement);

    camera.current = new THREE.PerspectiveCamera(
      field_of_view_ref.current,
      viewportWidth / viewportHeight,
      0.01,
      100,
    );
    camera.current.position.x = 5;
    camera.current.position.y = -5;
    camera.current.position.z = 5;
    camera.current.up = new THREE.Vector3(0, 0, 1);

    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(viewportWidth, viewportHeight);

    myRef.current.append(renderer.domElement);
    // add camera controls
    cameraControls = new OrbitControls(camera.current, renderer.domElement);
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

  // updates the field of view inside the ref to avoid rerendering so often
  useEffect(() => {
    field_of_view_ref.current = field_of_view;
  }, [field_of_view]);

  return (
    <>
      {/* the webrtc viewer needs to know the camera pose */}
      <WebRtcWindow />
      <div className="canvas-container-main" ref={myRef} />
    </>
  );
}
