// Code for creating a curve from a set of points

import * as THREE from 'three';

function get_catmull_rom_curve(list_of_3d_vectors, is_cycle, smoothness_value) {
  // TODO: add some hyperparameters to this function
  const curve = new THREE.CatmullRomCurve3(
    list_of_3d_vectors,
    is_cycle,
    // 'centripetal'
    'catmullrom',
    smoothness_value,
  );
  return curve;
}

export function get_curve_object_from_cameras(
  cameras,
  is_cycle,
  smoothness_value,
) {
  if (cameras.length === 0) {
    return null;
  }
  // interpolate positions, lookat directions, and ups
  // similar to
  // https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L281

  const positions = [];
  const lookats = [];
  const ups = [];
  const fovs = [];
  const render_times = [];

  for (let i = 0; i < cameras.length; i += 1) {
    const camera = cameras[i];

    const up = new THREE.Vector3(0, 1, 0); // y is up in local space
    const lookat = new THREE.Vector3(0, 0, 1); // z is forward in local space

    up.applyQuaternion(camera.quaternion);
    lookat.applyQuaternion(camera.quaternion);

    positions.push(camera.position);
    ups.push(up);
    lookats.push(lookat);
    // Reuse catmullromcurve3 for 1d values. TODO fix this
    fovs.push(new THREE.Vector3(0, 0, camera.fov));
    render_times.push(new THREE.Vector3(0, 0, camera.renderTime));
  }

  let curve_positions = null;
  let curve_lookats = null;
  let curve_ups = null;
  let curve_fovs = null;
  let curve_render_times = null;

  curve_positions = get_catmull_rom_curve(positions, is_cycle, smoothness_value);
  curve_lookats = get_catmull_rom_curve(lookats, is_cycle, smoothness_value);
  curve_ups = get_catmull_rom_curve(ups, is_cycle, smoothness_value);
  curve_fovs = get_catmull_rom_curve(fovs, is_cycle, smoothness_value / 10);
  curve_render_times = get_catmull_rom_curve(render_times, is_cycle, smoothness_value);

  const curve_object = {
    curve_positions,
    curve_lookats,
    curve_ups,
    curve_fovs,
    curve_render_times,
  };
  return curve_object;
}

export function get_transform_matrix(position, lookat, up) {
  // normalize the vectors
  lookat.normalize();
  // make up orthogonal to lookat
  const up_proj = lookat.clone().multiplyScalar(up.dot(lookat));
  up.sub(up_proj);
  up.normalize();

  // create a copy of the vector up
  const up_copy = up.clone();
  const cross = up_copy.cross(lookat);
  cross.normalize();

  // create the camera transform matrix
  const mat = new THREE.Matrix4();
  mat.set(
    cross.x,
    up.x,
    lookat.x,
    position.x,
    cross.y,
    up.y,
    lookat.y,
    position.y,
    cross.z,
    up.z,
    lookat.z,
    position.z,
    0,
    0,
    0,
    1,
  );
  return mat;
}
