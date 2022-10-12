/* eslint-disable no-restricted-syntax */
// ---- code for drawing with three.js ----
import * as THREE from 'three';

export function drawSceneBox(sceneBox): THREE.Object3D {
  console.assert(sceneBox.type === 'aabb', 'The scene box must be an AABB');

  const box = sceneBox;

  const w = 1.0;
  const aaa = new THREE.Vector3(w, w, w);
  const aab = new THREE.Vector3(w, w, -w);
  const aba = new THREE.Vector3(w, -w, w);
  const baa = new THREE.Vector3(-w, w, w);
  const abb = new THREE.Vector3(w, -w, -w);
  const bba = new THREE.Vector3(-w, -w, w);
  const bab = new THREE.Vector3(-w, w, -w);
  const bbb = new THREE.Vector3(-w, -w, -w);
  let points = [aaa, aab, aaa, aba, aab, abb, aba, abb];
  points = points.concat([baa, bab, baa, bba, bab, bbb, bba, bbb]);
  points = points.concat([aaa, baa, aab, bab, aba, bba, abb, bbb]);

  const maxPoint = new THREE.Vector3(...box.max_point);
  const minPoint = new THREE.Vector3(...box.min_point);

  const lengths = maxPoint.clone();
  lengths.sub(minPoint);

  const scalar = lengths.clone();
  scalar.divide(new THREE.Vector3(2.0, 2.0, 2.0));

  const offset = minPoint.clone();
  offset.add(scalar);
  for (let i = 0; i < points.length; i += 1) {
    points[i] = points[i].clone();
    points[i].multiply(scalar).add(offset);
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x000000,
    linewidth: 1,
  });
  const lines = new THREE.LineSegments(geometry, material);
  return lines;
}

export function getCameraWireframe(
  scale = 0.3,
  focalLength = 4,
  w = 1.5,
  h = 2,
) {
  // Returns a wireframe of a 3D line-plot of a camera symbol.
  // A wireframe is a frustum.
  // https://github.com/hangg7/mvs_visual/blob/275d382a824733a3187a8e3147be184dd6f14795/mvs_visual.py#L54.
  // scale: scale of rendering
  // focalLength: this is the focal length
  // w: width
  // h: height
  const f = focalLength;

  const ul = new THREE.Vector3(-w, h, -f);
  const ur = new THREE.Vector3(w, h, -f);
  const ll = new THREE.Vector3(-w, -h, -f);
  const lr = new THREE.Vector3(w, -h, -f);
  const C = new THREE.Vector3(0, 0, 0);
  const points = [
    C,
    ul,
    C,
    ur,
    C,
    ll,
    C,
    lr,
    C,
    ul,
    ur,
    ul,
    lr,
    ur,
    lr,
    ll,
    ll,
    ul,
  ];

  const scalar = new THREE.Vector3(scale, scale, scale);
  for (let i = 0; i < points.length; i += 1) {
    points[i].multiply(scalar);
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x000000,
    linewidth: 1,
  });
  const lines = new THREE.LineSegments(geometry, material);
  return lines;
}

export function drawCameraImagePlane(width, height, imageString, name) {
  // imageString is the texture as a base64 string
  const geometry = new THREE.PlaneGeometry(width, height);
  const material = new THREE.MeshBasicMaterial({
    side: THREE.DoubleSide,
  });
  const texture = new THREE.TextureLoader().load(imageString);
  material.map = texture;
  const plane = new THREE.Mesh(geometry, material);
  plane.name = name;
  return plane;
}

function transpose(matrix) {
  return matrix[0].map((col, i) => matrix.map((row) => row[i]));
}

export function drawCamera(camera, name): THREE.Object3D {
  const group = new THREE.Group();

  console.assert(
    camera.type === 'PinholeCamera',
    'The camera should be a PinholeCamera',
  );

  const height = 0.05;
  const displayedFocalLength = height;
  const width = (height * camera.cx) / camera.cy;
  const cameraWireframeObject = getCameraWireframe(
    1.0,
    displayedFocalLength,
    width,
    height,
  );
  cameraWireframeObject.translateZ(displayedFocalLength); // move the wireframe frustum back
  group.add(cameraWireframeObject);
  const cameraImagePlaneObject = drawCameraImagePlane(
    width * 2,
    height * 2,
    camera.image,
    name,
  );
  group.add(cameraImagePlaneObject);

  // make homogeneous coordinates and then
  // transpose and flatten the matrix into an array
  let c2w = JSON.parse(JSON.stringify(camera.camera_to_world));
  c2w.push([0, 0, 0, 1]);
  c2w = transpose(c2w).flat();

  const mat = new THREE.Matrix4();
  mat.fromArray(c2w);
  mat.decompose(group.position, group.quaternion, group.scale);

  return group;
}

export function drawCameras(cameras): Record<number, THREE.Object3D> {
  const cameraObjects = {};
  for (const [key, camera] of Object.entries(cameras)) {
    cameraObjects[key] = drawCamera(camera);
  }
  return cameraObjects;
}
