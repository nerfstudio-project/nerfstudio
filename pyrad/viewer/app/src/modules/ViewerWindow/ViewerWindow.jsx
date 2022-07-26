import * as THREE from 'three';

import React, { useEffect, useRef } from 'react';

import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls';
import WebRtcWindow from '../WebRtcWindow/WebRtcWindow';

export default function ViewerWindow(props) {
  const scene = props.scene;
  let controls_main = null;
  const myRef = useRef(null);

  const getViewportWidth = () => {
    return window.innerWidth - (window.innerWidth % 2);
  };

  const getViewportHeight = () => {
    return window.innerHeight;
  };

  let viewportWidth = getViewportWidth();
  let viewportHeight = getViewportHeight();

  let camera = new THREE.PerspectiveCamera(
    120,
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

  const update = () => {
    handleResize();
    camera.updateProjectionMatrix();
    controls_main.update();
    requestAnimationFrame(update);
    renderer.render(scene, camera);
  };

  // similar to componentDidMount
  useEffect(() => {
    myRef.current.append(renderer.domElement);

    // add controls
    controls_main = new TrackballControls(camera, renderer.domElement);
    controls_main.rotateSpeed = 2.0;
    controls_main.zoomSpeed = 0.3;
    controls_main.panSpeed = 0.2;
    controls_main.target.set(0, 0, 0); // focus point of the controls
    controls_main.autoRotate = false;
    controls_main.enableDamping = true;
    controls_main.dampingFactor = 1.0;
    controls_main.update();
    update();
  }, []);

  return (
    <div>
      {/* the webrtc viewer needs to know the camera pose */}
      <WebRtcWindow />
      <div className="canvas-container-main" ref={myRef} />
    </div>
  );
}
