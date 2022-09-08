import * as React from 'react';

import * as THREE from 'three';
import { Button, Slider } from '@mui/material';

function CameraList(props) {
  let camera_main = props.camera_main;
  let cameras = props.cameras;

  const set_position = (camera) => {
    console.log('setting position with camera', camera);

    const mat = new THREE.Matrix4();
    mat.fromArray(camera.matrix.elements);
    mat.decompose(
      props.camera_main.position,
      props.camera_main.quaternion,
      props.camera_main.scale,
    );

    console.log(camera_main);
  };

  console.log(cameras);

  let cameraList = cameras.map((camera, index) => {
    return (
      <div className="CameraList-row" key={index}>
        <Button onClick={() => set_position(camera)}>Camera {index}</Button>
      </div>
    );
  });
  return <div>{cameraList}</div>;
}

export default function CameraPanel(props) {
  console.log('rerendering camera panel;');

  const sceneTree = props.sceneTree;
  const main_camera = sceneTree.find_object(['Main Camera']);
  const [cameras, setCameras] = React.useState([]);

  const add_camera = () => {
    let main_camera_copy = main_camera.clone();
    // sceneTree.find(path.concat(['<object>'])).set_object(main_camera_copy);
    const newlist = cameras.concat(main_camera_copy);

    // also draw stuff

    let path = ['Camera Path'];
    main_camera_copy.far = main_camera_copy.near + 0.1;
    const helper = new THREE.CameraHelper(main_camera_copy);
    sceneTree
      .find(path.concat([cameras.length.toString(), '<object>']))
      .set_object(helper);
    setCameras(newlist);
  };

  const marks = [
    {
      value: 0,
      label: '0°C',
    },
    {
      value: 20,
      label: '20°C',
    },
    {
      value: 37,
      label: '37°C',
    },
    {
      value: 100,
      label: '100°C',
    },
  ];
  
  const valuetext = (value) => {
    return `${value}°C`;
  }

  return (
    <div className="CameraPanel">
      <Button
        id="CameraPanel-add-camera-button"
        variant="outlined"
        onClick={add_camera}
      >
        Add Camera
      </Button>
      <Slider
        aria-label="Custom marks"
        defaultValue={20}
        getAriaValueText={valuetext}
        step={10}
        valueLabelDisplay="auto"
        marks={marks}
      />
      <CameraList camera_main={main_camera} cameras={cameras} />
    </div>
  );
}
