// Code for creating a curve from a set of points

import * as THREE from 'three';

function get_curve(list_of_3d_vectors) {
  // TODO: add some hyperparameters to this function
  const curve = new THREE.CatmullRomCurve3(
    list_of_3d_vectors,
    false,
    'centripetal',
  );
  return curve;
}

export function get_curve_object_from_cameras(cameras) {
  // interpolate positions, lookat directions, and ups
  // similar to
  // https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L281

  const positions = [];
  const lookats = [];
  const ups = [];

  for (let i = 0; i < cameras.length; i += 1) {
    const camera = cameras[i];

    const up = new THREE.Vector3(0, 1, 0); // y is up in local space
    const lookat = new THREE.Vector3(0, 0, -1); // -z is forward in local space
    up.applyQuaternion(camera.quaternion);
    lookat.applyQuaternion(camera.quaternion);

    positions.push(camera.position);
    ups.push(up);
    lookats.push(lookat);
  }

  let curve_positions = null;
  let curve_lookats = null;
  let curve_ups = null;

  curve_positions = get_curve(positions);
  curve_lookats = get_curve(lookats);
  curve_ups = get_curve(ups);

  const curve_object = {
    curve_positions,
    curve_lookats,
    curve_ups,
  };

  return curve_object;
}
