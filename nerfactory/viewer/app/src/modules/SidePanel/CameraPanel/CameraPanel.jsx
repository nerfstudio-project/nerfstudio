import * as React from 'react';
import * as THREE from 'three';

import { Button, Slider } from '@mui/material';
import dayjs, { Dayjs } from 'dayjs';

import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import DeleteIcon from '@mui/icons-material/Delete';
import FirstPageIcon from '@mui/icons-material/FirstPage';
import LastPageIcon from '@mui/icons-material/LastPage';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import PauseIcon from '@mui/icons-material/Pause';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TextField from '@mui/material/TextField';
import { TimePicker } from '@mui/x-date-pickers/TimePicker';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { get_curve_object_from_cameras } from './curve';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';

function IntervalSlider(props) {
  const [value, setValue] = React.useState(dayjs());
  return (
    <div className="CameraList-row-time-interval">
      <TextField
        id="outlined-number"
        label="Seconds"
        type="number"
        defaultValue={1}
      />
    </div>
  );
}

function CameraList(props) {
  let sceneTree = props.sceneTree;
  let transform_controls = props.transform_controls;
  let camera_main = props.camera_main;
  let cameras = props.cameras;

  const set_transform_controls = (index) => {
    // camera helper object so grab the camera inside
    let camera = sceneTree.find([
      'Camera Path',
      index.toString(),
      'Camera',
      '<object>',
    ]).object;
    console.log(camera);
    console.log('setting transform controls with camera: ', camera);
    console.log(transform_controls);
    props.transform_controls.attach(camera);
  };

  // handle camera selection
  // const [value, setValue] = React.useState(0);

  const set_main_camera_position = (camera) => {
    const mat = new THREE.Matrix4();
    mat.fromArray(camera.matrix.elements);
    mat.decompose(
      props.camera_main.position,
      props.camera_main.quaternion,
      props.camera_main.scale,
    );
  };

  const delete_camera = (index) => {
    console.log('deleting camera: ', index);
  };

  console.log(cameras);

  const num_cameras = cameras.length;

  const slider = IntervalSlider({});

  let cameraList = cameras.map((camera, index) => {
    return (
      <>
        <div className="CameraList-row" key={index}>
          <Button onClick={() => set_transform_controls(index)}>
            Camera {index}
          </Button>
          <div className="CameraList-row-buttons">
            <Button onClick={() => set_main_camera_position(camera)}>
              <VisibilityIcon />
            </Button>
            <Button onClick={() => delete_camera(camera)}>
              <DeleteIcon />
            </Button>
          </div>
        </div>
        {index < num_cameras - 1 ? slider : null}
      </>
    );
  });
  return <div>{cameraList}</div>;
}

export default function CameraPanel(props) {
  console.log('rerendering camera panel;');

  const sceneTree = props.sceneTree;
  const main_camera = sceneTree.find_object(['Main Camera']);
  const [cameras, setCameras] = React.useState([]);

  const [is_playing, setIsPlaying] = React.useState(false);

  const transform_controls = sceneTree.find_object(['Transform Controls']);

  const add_camera = () => {
    let main_camera_copy = main_camera.clone();
    // sceneTree.find(path.concat(['<object>'])).set_object(main_camera_copy);
    const newlist = cameras.concat(main_camera_copy);

    // also draw stuff

    let path = ['Camera Path'];
    main_camera_copy.far = main_camera_copy.near + 0.1;
    const helper = new THREE.CameraHelper(main_camera_copy);
    // camera
    sceneTree
      .find(['Camera Path', cameras.length.toString(), 'Camera', '<object>'])
      .set_object(main_camera_copy);
    // helper
    sceneTree
      .find([
        'Camera Path',
        cameras.length.toString(),
        'Camera Helper',
        '<object>',
      ])
      .set_object(helper);

    setCameras(newlist);
  };

  // update the spline interpolated
  const curve = get_curve_object_from_cameras(cameras);
  sceneTree.find(['Camera Path', 'Curve', '<object>']).set_object(curve);

  let marks = [];
  for (let i = 0; i < cameras.length; i++) {
    // let camera = cameras[i];
    marks.push({ value: i, label: i.toString() });
  }

  // const marks = [
  //   {
  //     value: 0,
  //     label: '0°C',
  //   },
  //   {
  //     value: 0.5,
  //     // label: '20°C',
  //   },
  //   {
  //     value: 0.7,
  //     // label: '37°C',
  //   },
  //   {
  //     value: 1,
  //     label: '100°C',
  //   },
  // ];

  console.log(sceneTree);

  // const valuetext = (value) => {
  //   return `${value}°C`;
  // };

  return (
    <div className="CameraPanel">
      <div className="CameraPanel-top-button">
        <Button variant="outlined" onClick={add_camera}>
          Add Camera
        </Button>
      </div>
      <div className="CameraPanel-top-button">
        <Button className="CameraPanel-top-button" variant="outlined">
          Export Path
        </Button>
      </div>
      <div className="CameraPanel-slider-container">
        <Slider
          defaultValue={0}
          step={0.001}
          valueLabelDisplay="auto"
          marks={marks}
          min={0}
          max={Math.max(0, cameras.length - 1)}
          disabled={cameras.length < 2}
          valueLabelDisplay="on"
        />
      </div>
      <div className="CameraPanel-slider-button-container">
        <Button variant="outlined">
          <FirstPageIcon />
        </Button>
        <Button variant="outlined">
          <ArrowBackIosNewIcon />
        </Button>
        <Button variant="outlined" onClick={() => setIsPlaying(!is_playing)}>
          {is_playing ? <PauseIcon /> : <PlayArrowIcon />}
        </Button>
        <Button variant="outlined">
          <ArrowForwardIosIcon />
        </Button>
        <Button variant="outlined">
          <LastPageIcon />
        </Button>
      </div>
      <div className="CameraList-container">
        <CameraList
          sceneTree={sceneTree}
          transform_controls={transform_controls}
          camera_main={main_camera}
          cameras={cameras}
        />
      </div>
    </div>
  );
}
