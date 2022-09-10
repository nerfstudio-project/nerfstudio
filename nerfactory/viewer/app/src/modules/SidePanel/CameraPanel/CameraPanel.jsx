import * as React from 'react';
import * as THREE from 'three';

import { Button, Slider } from '@mui/material';
import dayjs, { Dayjs } from 'dayjs';

import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
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
import { useEffect } from 'react';

function set_camera_position(camera, matrix) {
  const mat = new THREE.Matrix4();
  mat.fromArray(matrix.elements);
  mat.decompose(camera.position, camera.quaternion, camera.scale);
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

  const set_camera_main_position = (camera) => {
    // const mat = new THREE.Matrix4();
    // mat.fromArray(camera.matrix.elements);
    // mat.decompose(
    //   props.camera_main.position,
    //   props.camera_main.quaternion,
    //   props.camera_main.scale,
    // );
    set_camera_position(props.camera_main, camera.matrix);
  };

  const delete_camera = (index) => {
    console.log('deleting camera: ', index);
  };

  console.log(cameras);

  const num_cameras = cameras.length;

  let cameraList = cameras.map((camera, index) => {
    return (
      <>
        <div className="CameraList-row" key={index}>
          <Button onClick={() => set_transform_controls(index)}>
            Camera {index}
          </Button>
          <div className="CameraList-row-buttons">
            <Button onClick={() => set_camera_main_position(camera)}>
              <VisibilityIcon />
            </Button>
            <Button onClick={() => delete_camera(camera)}>
              <DeleteIcon />
            </Button>
          </div>
        </div>
      </>
    );
  });
  return <div>{cameraList}</div>;
}

export default function CameraPanel(props) {
  console.log('rerendering camera panel;');

  const sceneTree = props.sceneTree;
  const camera_main = sceneTree.find_object(['Main Camera']);
  const [cameras, setCameras] = React.useState([]);

  const [slider_value, set_slider_value] = React.useState(0);
  const [is_playing, setIsPlaying] = React.useState(false);
  const [seconds, setSeconds] = React.useState(4);
  const [fps, setFps] = React.useState(24);

  const transform_controls = sceneTree.find_object(['Transform Controls']);

  // TODO: add listener to reupdate when the Camera Path changes

  const add_camera = () => {
    let camera_main_copy = camera_main.clone();
    const newlist = cameras.concat(camera_main_copy);
    camera_main_copy.far = camera_main_copy.near + 0.1;
    const helper = new THREE.CameraHelper(camera_main_copy);
    // camera
    sceneTree
      .find(['Camera Path', cameras.length.toString(), 'Camera', '<object>'])
      .set_object(camera_main_copy);
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
  const curve_object = get_curve_object_from_cameras(cameras);
  if (cameras.length >= 2) {
    const num_points = fps * seconds;
    const points = curve_object.curve_positions.getPoints(num_points);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0xff0000 });
    const threejs_object = new THREE.Line(geometry, material);
    sceneTree
      .find(['Camera Path', 'Curve', '<object>'])
      .set_object(threejs_object);
  }

  const marks = [];
  for (let i = 0; i < cameras.length; i++) {
    marks.push({ value: i, label: i.toString() });
  }

  const handle_slider_change = (event, value) => {
    set_slider_value(value);
  };

  // when the slider changes, update the main camera position
  useEffect(() => {
    if (cameras.length > 1) {
      // interpolate to get the points
      const position = curve_object.curve_positions.getPoint(
        slider_value / (cameras.length - 1.0),
      );
      const up = curve_object.curve_ups.getPoint(slider_value / cameras.length);
      const lookat = curve_object.curve_lookats.getPoint(
        slider_value / cameras.length,
      );

      // create a copy of the vector up
      const up_copy = up.clone();
      const cross = up_copy.cross(lookat);

      up.normalize();
      lookat.normalize();
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
      );
      set_camera_position(camera_main, mat);
    }
  }, [slider_value]);

  const total_num_steps = seconds * fps;
  const step_size = (cameras.length - 1) / total_num_steps;
  const slider_min = 0;
  const slider_max = Math.max(0, cameras.length - 1);

  // call this function whenever slider state changes
  useEffect(() => {
    if (is_playing) {
      const interval = setInterval(() => {
        console.log(slider_value);
        set_slider_value((prev) => prev + step_size);
        console.log('inside set interval\n\n');
      }, 1000 / fps);
      return () => clearInterval(interval);
    }
  }, [is_playing]);

  // make sure to pause if the slider reaches the end
  useEffect(() => {
    if (slider_value >= slider_max) {
      set_slider_value(slider_max);
      setIsPlaying(false);
    }
  }, [slider_value]);

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
          value={slider_value}
          step={step_size}
          valueLabelDisplay="on"
          marks={marks}
          min={slider_min}
          max={slider_max}
          disabled={cameras.length < 2}
          onChange={handle_slider_change}
        />
      </div>
      <div className="CameraPanel-slider-button-container">
        <Button
          variant="outlined"
          onClick={() => {
            setIsPlaying(false);
            set_slider_value(slider_min);
          }}
        >
          <FirstPageIcon />
        </Button>
        <Button
          variant="outlined"
          onClick={() => set_slider_value(slider_value - step_size)}
        >
          <ArrowBackIosNewIcon />
        </Button>

        {!is_playing ? (
          <Button
            variant="outlined"
            onClick={() => {
              setIsPlaying(true);
            }}
          >
            <PlayArrowIcon />
          </Button>
        ) : (
          <Button
            variant="outlined"
            onClick={() => {
              setIsPlaying(false);
            }}
          >
            <PauseIcon />
          </Button>
        )}
        <Button
          variant="outlined"
          onClick={() => set_slider_value(slider_value + step_size)}
        >
          <ArrowForwardIosIcon />
        </Button>
        <Button variant="outlined" onClick={() => set_slider_value(slider_max)}>
          <LastPageIcon />
        </Button>
      </div>
      <div className="CameraList-row-time-interval">
        <TextField
          id="outlined-number"
          label="Seconds"
          type="number"
          onChange={(e) => setSeconds(e.target.value)}
          defaultValue={seconds}
        />
        <TextField
          id="outlined-number"
          label="FPS"
          type="number"
          onChange={(e) => setFps(e.target.value)}
          defaultValue={fps}
        />
      </div>
      <div className="CameraList-container">
        <CameraList
          sceneTree={sceneTree}
          transform_controls={transform_controls}
          camera_main={camera_main}
          cameras={cameras}
        />
      </div>
    </div>
  );
}
