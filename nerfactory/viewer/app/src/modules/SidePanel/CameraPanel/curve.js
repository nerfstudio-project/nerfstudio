// Code for creating a curve from a set of points

import * as THREE from 'three';

export function get_curve_object_from_cameras(cameras) {
  const positions = [];
  const lookats = [];
  const ups = [];

  console.log(cameras);
  for (let i = 0; i < cameras.length; i++) {
    const camera = cameras[i];
    positions.push(camera.position);
    // lookats.push(camera.lookat);
    ups.push(camera.up);
  }

  let curve = null;
  if (positions.length < 2) {
    curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-10, 0, 10),
      new THREE.Vector3(-5, 5, 5),
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(5, -5, 5),
      new THREE.Vector3(10, 0, 10),
    ]);
  } else {
    curve = new THREE.CatmullRomCurve3(positions);
  }

  const points = curve.getPoints(50);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  const material = new THREE.LineBasicMaterial({ color: 0xff0000 });

  // Create the final object to add to the scene
  const curveObject = new THREE.Line(geometry, material);

  return curveObject;
}
