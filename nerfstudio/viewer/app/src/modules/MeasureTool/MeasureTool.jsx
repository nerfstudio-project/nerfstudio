import * as React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer';
import { Line2 } from 'three/examples/jsm/lines/Line2';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial';

const MEASUREMENT_NAME = 'Measurement';

export default function MeasureTool(props) {
  const sceneTree = props.sceneTree;
  const renderer = sceneTree.metadata.renderer;
  const camera_controls = sceneTree.metadata.camera_controls;
  const overlayRef = React.useRef(null);

  const dispatch = useDispatch();

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  const [isMeasuring, setMeasuring] = React.useState(false);
  const [referencePoints, setReferencePoints] = React.useState([]);

  const fontSize = useSelector((state) => state.measState.fontSize);
  const color = useSelector((state) => state.measState.color);
  const markerRadius = useSelector((state) => state.measState.markerRadius);
  const lineWidth = useSelector((state) => state.measState.lineWidth);

  React.useEffect(() => {
    // camera_controls.rotatePolarTo(0, true);
    camera_controls.enabled = false;

    // const ray = new THREE.Object3D();
    // ray.name = 'ray';
    // sceneTree.set_object_from_path([MEASUREMENT_NAME, 'ray'], ray);

    return () => {
      camera_controls.enabled = true;
    };
  }, []);

  const createMarker = React.useCallback(
    (point) => {
      // Draw sphere marker
      const geom = new THREE.SphereGeometry(markerRadius);
      const mat = new THREE.MeshLambertMaterial({ color });
      const circle = new THREE.Mesh(geom, mat);
      circle.position.set(point.x, point.y, 0);

      // Draw a marker label
      const label = document.createElement('div');
      const d = `(${point.x.toFixed(3)}, ${point.y.toFixed(3)})`;
      label.style.color = color;
      label.style.fontFamily = 'sans-serif';
      label.style.fontSize = fontSize;
      label.textContent = d;
      label.style.background = 'transparent';

      const labelObj = new CSS2DObject(label);
      labelObj.position.set(point.x, point.y, 0);
      labelObj.layers.set(0);
      circle.add(labelObj);

      return circle;
    },
    [color, fontSize],
  );

  // Draw a line between last reference point and current pointer poistion,
  // https://threejs.org/examples/#webgl_lines_fat
  const createLine = React.useCallback(
    (points) => {
      const rgb = new THREE.Color(color);
      const matLine = new LineMaterial({
        color,
        linewidth: lineWidth,
        dashed: true,
        dashSize: lineWidth * 10,
        gapSize: lineWidth * 10,
        alphaToCoverage: true,
      });

      const geomLine = new LineGeometry();
      geomLine.setPositions([
        points[0].x,
        points[0].y,
        points[0].z,
        points[1].x,
        points[1].y,
        points[1].z,
      ]);
      geomLine.setColors([rgb.r, rgb.g, rgb.b]);

      const line = new Line2(geomLine, matLine);
      line.computeLineDistances();
      line.scale.set(1, 1, 1);

      return line;
    },
    [color, lineWidth],
  );

  const handleMeasStart = React.useCallback(
    (evt) => {
      const canvas = renderer.domElement;
      const canvasPos = canvas.getBoundingClientRect();

      // FIXME(rakuto): Raycaster logic is not wroking properly to get current position in Threejs scene.
      const pointer = new THREE.Vector3();
      pointer.x = ((evt.clientX - canvasPos.left) / canvas.clientWidth) * 2 - 1;
      pointer.y =
        -((evt.clientY - canvasPos.top) / canvas.clientHeight) * 2 + 1;
      pointer.z = 0;

      // raycaster.setFromCamera(pointer, sceneTree.metadata.camera);
      // const intersects = raycaster.intersectObjects(
      //   sceneTree.object.children,
      //   true,
      // );
      // console.log(
      //   sceneTree.object.children,
      //   intersects.map((o) => o.point),
      // );
      // if (intersects.length > 0) {
      // const point = intersects[0].point;

      const marker = createMarker(pointer);
      sceneTree.set_object_from_path(
        [MEASUREMENT_NAME, `marker-${referencePoints.length}`],
        marker,
      );

      referencePoints.push(pointer);
      setReferencePoints(referencePoints);
      setMeasuring(true);
    },
    [renderer, sceneTree, raycaster, renderer, referencePoints],
  );

  const handleMeasMove = React.useCallback(
    (evt) => {
      if (!isMeasuring) return;

      const canvas = renderer.domElement;
      const canvasPos = canvas.getBoundingClientRect();

      const pointer = new THREE.Vector3();
      pointer.x = ((evt.clientX - canvasPos.left) / canvas.clientWidth) * 2 - 1;
      pointer.y =
        -((evt.clientY - canvasPos.top) / canvas.clientHeight) * 2 + 1;
      pointer.z = 0;

      const points = [referencePoints[referencePoints.length - 1], pointer];
      const line = createLine(points);
      sceneTree.set_object_from_path([MEASUREMENT_NAME, `line`], line);
    },
    [renderer, sceneTree, referencePoints, isMeasuring, color, lineWidth],
  );

  const handleMeasEnd = React.useCallback(
    (evt) => {
      const canvas = renderer.domElement;
      const canvasPos = canvas.getBoundingClientRect();

      const pointer = new THREE.Vector3();
      pointer.x = ((evt.clientX - canvasPos.left) / canvas.clientWidth) * 2 - 1;
      pointer.y =
        -((evt.clientY - canvasPos.top) / canvas.clientHeight) * 2 + 1;
      pointer.z = 0;

      const name = `marker-${referencePoints.length}`;
      const marker = createMarker(pointer);
      sceneTree.set_object_from_path([MEASUREMENT_NAME, name], marker);

      // Draw line between a last reference point and current point.
      const points = [referencePoints[referencePoints.length - 1], pointer];
      const line = createLine(points);
      sceneTree.delete([MEASUREMENT_NAME, 'line']);
      sceneTree.set_object_from_path([MEASUREMENT_NAME, name, 'line'], line);

      referencePoints.push(pointer);
      setReferencePoints(referencePoints);
      setMeasuring(false);
    },
    [renderer, referencePoints, color],
  );

  return (
    <div
      ref={overlayRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: renderer.domElement.offsetWidth + 'px',
        height: renderer.domElement.offsetHeight + 'px',
        zIndex: 9999,
        background: 'transparent',
        cursor: 'crosshair',
      }}
      onPointerDown={handleMeasStart}
      onPointerMove={handleMeasMove}
      onPointerUp={handleMeasEnd}
    ></div>
  );
}
