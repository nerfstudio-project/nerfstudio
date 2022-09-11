import * as React from 'react';
import * as THREE from 'three';
import { MeshLine, MeshLineMaterial } from 'meshline';

import { Button, Slider } from '@mui/material';

import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import DeleteIcon from '@mui/icons-material/Delete';
import FirstPageIcon from '@mui/icons-material/FirstPage';
import LastPageIcon from '@mui/icons-material/LastPage';
import PauseIcon from '@mui/icons-material/Pause';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TextField from '@mui/material/TextField';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { useEffect } from 'react';
import { CameraHelper } from './CameraHelper';
import { get_curve_object_from_cameras } from './curve';

function set_camera_position(camera, matrix) {
  const mat = new THREE.Matrix4();
  mat.fromArray(matrix.elements);
  mat.decompose(camera.position, camera.quaternion, camera.scale);
}

function CameraList(props) {
  const sceneTree = props.sceneTree;
  const cameras = props.cameras;
  const camera_main = props.camera_main;
  const transform_controls = props.transform_controls;
  const setCameras = props.setCameras;

  const set_transform_controls = (index) => {
    // camera helper object so grab the camera inside
    const camera = sceneTree.find([
      'Camera Path',
      'Cameras',
      index.toString(),
      'Camera',
      '<object>',
    ]).object;
    transform_controls.attach(camera);
  };

  const delete_camera = (index) => {
    console.log('TODO: deleting camera: ', index);
    setCameras([...cameras.slice(0, index), ...cameras.slice(index + 1)]);
  };

  const cameraList = cameras.map((camera, index) => {
    return (
      <div className="CameraList-row" key={camera.uuid}>
        <Button onClick={() => set_transform_controls(index)}>
          Camera {index}
        </Button>
        <div className="CameraList-row-buttons">
          <Button
            onClick={() => set_camera_position(camera_main, camera.matrix)}
          >
            <VisibilityIcon />
          </Button>
          <Button onClick={() => delete_camera(index)}>
            <DeleteIcon />
          </Button>
        </div>
      </div>
    );
  });
  return <div>{cameraList}</div>;
}

export default function CameraPanel(props) {
  // unpack props
  const sceneTree = props.sceneTree;

  // scene tree objects
  const camera_main = sceneTree.find_object(['Main Camera']);
  const transform_controls = sceneTree.find_object(['Transform Controls']);

  // react state
  const [cameras, setCameras] = React.useState([]);
  const [slider_value, set_slider_value] = React.useState(0);
  const [is_playing, setIsPlaying] = React.useState(false);
  const [seconds, setSeconds] = React.useState(4);
  const [fps, setFps] = React.useState(24);
  const total_num_steps = seconds * fps;
  const step_size = (cameras.length - 1) / total_num_steps;
  const slider_min = 0;
  const slider_max = Math.max(0, cameras.length - 1);

  const add_camera = () => {
    const camera_main_copy = camera_main.clone();
    camera_main_copy.far = camera_main_copy.near + 0.1;
    const new_camera_list = cameras.concat(camera_main_copy);
    setCameras(new_camera_list);
  };

  // force a rerender if the cameras are dragged around
  let update_cameras_interval = null;
  // eslint-disable-next-line no-unused-vars
  transform_controls.addEventListener('mouseDown', (event) => {
    // prevent multiple loops
    if (update_cameras_interval === null) {
      // hardcoded for 100 ms per udpate
      update_cameras_interval = setInterval(() => {
        setCameras(cameras);
      }, 100);
    }
  });
  // eslint-disable-next-line no-unused-vars
  transform_controls.addEventListener('mouseUp', (event) => {
    if (update_cameras_interval !== null) {
      clearInterval(update_cameras_interval);
      update_cameras_interval = null;
    }
  });

  // draw cameras and curve to the scene
  useEffect(() => {
    // draw the cameras
    sceneTree.delete(['Camera Path', 'Cameras']); // delete old cameras, which is important
    for (let i = 0; i < cameras.length; i += 1) {
      const camera = cameras[i];
      const helper = new CameraHelper(camera);
      console.log('HERERE');
      console.log(helper);
      // camera
      sceneTree.set_object_from_path(
        ['Camera Path', 'Cameras', i.toString(), 'Camera'],
        camera,
      );
      // camera helper
      sceneTree.set_object_from_path(
        ['Camera Path', 'Cameras', i.toString(), 'Camera Helper'],
        helper,
      );
    }
  }, [cameras]);

  // update the camera curve
  const curve_object = get_curve_object_from_cameras(cameras);
  if (cameras.length > 1) {
    const num_points = fps * seconds;
    const points = curve_object.curve_positions.getPoints(num_points);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const spline = new MeshLine();
    spline.setGeometry(geometry);
    const material = new MeshLineMaterial({ lineWidth: 0.05, color: 0xff5024 });
    const spline_mesh = new THREE.Mesh(spline.geometry, material);
    sceneTree.set_object_from_path(['Camera Path', 'Curve'], spline_mesh);
  }

  const marks = [];
  for (let i = 0; i < cameras.length; i += 1) {
    marks.push({ value: i, label: i.toString() });
  }

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

  // call this function whenever slider state changes
  useEffect(() => {
    if (is_playing) {
      const interval = setInterval(() => {
        set_slider_value((prev) => prev + step_size);
      }, 1000 / fps);
      return () => clearInterval(interval);
    }
    return () => {};
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
          valueLabelFormat={slider_value.toFixed(2)}
          marks={marks}
          min={slider_min}
          max={slider_max}
          disabled={cameras.length < 2}
          onChange={(event, value) => {
            set_slider_value(value);
          }}
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
          label="Seconds"
          inputProps={{
            inputMode: 'numeric',
            pattern: '[+-]?([0-9]*[.])?[0-9]+',
          }}
          size="small"
          onChange={(e) => {
            if (e.target.validity.valid) {
              setSeconds(e.target.value);
            }
          }}
          value={seconds}
          error={seconds <= 0}
          helperText={seconds <= 0 ? 'Required' : ''}
          variant="standard"
        />
        <TextField
          label="FPS"
          inputProps={{ inputMode: 'numeric', pattern: '[0-9]*' }}
          size="small"
          onChange={(e) => {
            if (e.target.validity.valid) {
              setFps(e.target.value);
            }
          }}
          value={fps}
          error={fps <= 0}
          helperText={fps <= 0 ? 'Required' : ''}
          variant="standard"
        />
      </div>
      <div className="CameraList-container">
        <CameraList
          sceneTree={sceneTree}
          transform_controls={transform_controls}
          camera_main={camera_main}
          cameras={cameras}
          setCameras={setCameras}
        />
      </div>
    </div>
  );
}
