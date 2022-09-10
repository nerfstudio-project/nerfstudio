// Code for creating a curve from a set of points

import * as THREE from 'three';

function get_curve(list_of_3d_vectors) {
  // TODO: add some hyperparameters to this function
  const curve = new THREE.CatmullRomCurve3(
    list_of_3d_vectors,
    false,
    'catmullrom',
  );
  return curve;
}

export function get_curve_object_from_cameras(cameras) {
  const positions = [];
  const lookats = [];
  const ups = [];

  console.log(cameras);
  for (let i = 0; i < cameras.length; i++) {
    const camera = cameras[i];

    const up = new THREE.Vector3(0, 1, 0); // y is up in local space
    const lookat = new THREE.Vector3(0, 0, -1); // -z is forward in local space
    up.applyQuaternion(camera.quaternion);
    lookat.applyQuaternion(camera.quaternion);

    positions.push(camera.position);
    ups.push(up);
    lookats.push(lookat);

    console.log(cameras);
    console.log(ups);
  }

  let curve_positions = null;
  let curve_lookats = null;
  let curve_ups = null;
  let threejs_object = null;

  curve_positions = get_curve(positions);
  curve_lookats = get_curve(lookats);
  curve_ups = get_curve(ups);

  // const points = curve.getPoints(50);
  // const geometry = new THREE.BufferGeometry().setFromPoints(points);
  // const material = new THREE.LineBasicMaterial({ color: 0xff0000 });
  // threejs_object = new THREE.Line(geometry, material);

  const curve_object = {
    curve_positions: curve_positions,
    curve_lookats: curve_lookats,
    curve_ups: curve_ups,
    // threejs_object: curveObject,
  };

  return curve_object;
}
