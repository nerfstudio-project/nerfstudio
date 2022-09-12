import * as THREE from 'three';
import { MeshLine, MeshLineMaterial } from 'meshline';

// eslint-disable-next-line no-underscore-dangle
const _vector = new THREE.Vector3();
// eslint-disable-next-line no-underscore-dangle
const _camera = new THREE.Camera();

function setPoint(point, pointMap, geometry, camera, x, y, z) {
  _vector.set(x, y, z).unproject(camera);

  const points = pointMap[point];

  if (points !== undefined) {
    const position = geometry.getAttribute('position');

    for (let i = 0, l = points.length; i < l; i += 1) {
      position.setXYZ(points[i], _vector.x, _vector.y, _vector.z);
    }
  }
}

/**
 *	- Modified version of THREE.CameraHelper
 */

class CameraHelper extends THREE.Mesh {
  constructor(camera) {
    const line_geometry = new THREE.BufferGeometry();
    const geometry = new THREE.BufferGeometry();
    const material = new MeshLineMaterial({
      color: 0x000000,
      lineWidth: 0.01,
    });

    const vertices = [];

    const pointMap = {};

    function addPoint(id) {
      vertices.push(0, 0, 0);

      if (pointMap[id] === undefined) {
        pointMap[id] = [];
      }

      pointMap[id].push(vertices.length / 3 - 1);
    }

    function addLine(a, b) {
      addPoint(a);
      addPoint(b);
    }

    // near

    addLine('n1', 'n2');
    addLine('n2', 'n4');
    addLine('n4', 'n3');
    addLine('n3', 'n1');

    // cone

    addLine('p', 'n1');
    addLine('p', 'n2');
    addLine('p', 'n3');
    addLine('p', 'n4');

    // up

    addLine('u1', 'u2');
    addLine('u2', 'u3');
    addLine('u3', 'u1');

    line_geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(vertices, 3),
    );

    super(geometry, material);

    this.type = 'CameraHelper';

    this.camera = camera;
    if (this.camera.updateProjectionMatrix)
      this.camera.updateProjectionMatrix();

    this.matrix = camera.matrixWorld;
    this.matrixAutoUpdate = false;

    this.pointMap = pointMap;

    this.line_geometry = line_geometry;

    this.update();
  }

  update() {
    const geometry = this.line_geometry;
    const pointMap = this.pointMap;

    const w = 1;
    const h = 1;
    const z = 2;

    // we need just camera projection matrix inverse
    // world matrix must be identity

    console.log(_camera.projectionMatrixInverse);

    _camera.projectionMatrixInverse.copy(this.camera.projectionMatrixInverse);

    // near

    setPoint('n1', pointMap, geometry, _camera, -w, -h, -z);
    setPoint('n2', pointMap, geometry, _camera, w, -h, -z);
    setPoint('n3', pointMap, geometry, _camera, -w, h, -z);
    setPoint('n4', pointMap, geometry, _camera, w, h, -z);

    // up

    setPoint('u1', pointMap, geometry, _camera, w * 0.7, h * 1.1, -z);
    setPoint('u2', pointMap, geometry, _camera, -w * 0.7, h * 1.1, -z);
    setPoint('u3', pointMap, geometry, _camera, 0, h * 2, -z);

    geometry.getAttribute('position').needsUpdate = true;

    const camera_vis = new MeshLine();
    camera_vis.setGeometry(geometry);
    this.geometry = camera_vis.geometry;
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}

export { CameraHelper };
