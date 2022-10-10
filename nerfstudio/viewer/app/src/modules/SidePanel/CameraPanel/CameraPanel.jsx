import * as React from 'react';
import * as THREE from 'three';

import {
  AllInclusiveOutlined,
  ChangeHistory,
  GestureOutlined,
  LinearScaleOutlined,
  RadioButtonUnchecked,
  Replay,
  Timeline,
} from '@mui/icons-material';
import { Button, Slider } from '@mui/material';
import { MeshLine, MeshLineMaterial } from 'meshline';
import { useContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer';
import ContentPasteGoIcon from '@mui/icons-material/ContentPasteGo';
import DeleteIcon from '@mui/icons-material/Delete';
import FirstPageIcon from '@mui/icons-material/FirstPage';
import IconButton from '@mui/material/IconButton';
import LastPageIcon from '@mui/icons-material/LastPage';
import PauseIcon from '@mui/icons-material/Pause';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Stack } from '@mui/system';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { CameraHelper } from './CameraHelper';
import { get_curve_object_from_cameras, get_transform_matrix } from './curve';
import { WebSocketContext } from '../../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

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
  // eslint-disable-next-line no-unused-vars
  const [slider_value, set_slider_value] = React.useState(0);

  const set_transform_controls = (index) => {
    // camera helper object so grab the camera inside
    const camera = sceneTree.find_object_no_create([
      'Camera Path',
      'Cameras',
      index.toString(),
      'Camera',
    ]);
    if (camera !== null) {
      const viewer_buttons = document.getElementsByClassName(
        'ViewerWindow-buttons',
      )[0];
      if (camera === transform_controls.object) {
        // double click to remove controls from object
        transform_controls.detach();
        viewer_buttons.style.display = 'none';
      } else {
        transform_controls.detach();
        transform_controls.attach(camera);
        viewer_buttons.style.display = 'block';
      }
    }
  };

  const reset_slider_render_on_delete = () => {
    // set slider and render camera back to 0
    const slider_min = 0;
    const camera_render = sceneTree.find_object_no_create([
      'Cameras',
      'Render Camera',
    ]);
    const camera_render_helper = sceneTree.find_object_no_create([
      'Cameras',
      'Render Camera',
      'Helper',
    ]);
    if (cameras.length >= 1) {
      let first_camera = sceneTree.find_object_no_create([
        'Camera Path',
        'Cameras',
        0,
        'Camera',
      ]);
      if (first_camera.type !== 'PerspectiveCamera' && cameras.length > 1) {
        first_camera = sceneTree.find_object_no_create([
          'Camera Path',
          'Cameras',
          1,
          'Camera',
        ]);
      }
      set_camera_position(camera_render, first_camera.matrix);
      camera_render_helper.set_visibility(true);
    }
    set_slider_value(slider_min);
  };

  const delete_camera = (index) => {
    const camera_render_helper = sceneTree.find_object_no_create([
      'Cameras',
      'Render Camera',
      'Helper',
    ]);
    console.log('TODO: deleting camera: ', index);
    sceneTree.delete(['Camera Path', 'Cameras', index.toString(), 'Camera']);
    sceneTree.delete([
      'Camera Path',
      'Cameras',
      index.toString(),
      'Camera Helper',
    ]);

    setCameras([...cameras.slice(0, index), ...cameras.slice(index + 1)]);
    // detach and hide transform controls
    transform_controls.detach();
    const viewer_buttons = document.getElementsByClassName(
      'ViewerWindow-buttons',
    )[0];
    viewer_buttons.style.display = 'none';
    if (cameras.length < 1) {
      camera_render_helper.set_visibility(false);
    }
    reset_slider_render_on_delete();
  };

  const cameraList = cameras.map((camera, index) => {
    return (
      <div className="CameraList-row" key={camera.uuid}>
        <Button size="small" onClick={() => set_transform_controls(index)}>
          Camera {index}
        </Button>
        <div className="CameraList-row-buttons">
          <Button
            size="small"
            onClick={() => {
              set_camera_position(camera_main, camera.matrix);
            }}
          >
            <VisibilityIcon />
          </Button>
          <Button size="small" onClick={() => delete_camera(index)}>
            <DeleteIcon />
          </Button>
        </div>
      </div>
    );
  });
  return <div>{cameraList}</div>;
}

export default function CameraPanel(props) {
  // unpack relevant information
  const sceneTree = props.sceneTree;
  const camera_main = sceneTree.find_object_no_create([
    'Cameras',
    'Main Camera',
  ]);
  const camera_render = sceneTree.find_object_no_create([
    'Cameras',
    'Render Camera',
  ]);
  const camera_render_helper = sceneTree.find_object_no_create([
    'Cameras',
    'Render Camera',
    'Helper',
  ]);
  const transform_controls = sceneTree.find_object_no_create([
    'Transform Controls',
  ]);

  // redux store state
  const config_base_dir = useSelector(
    (state) => state.renderingState.config_base_dir,
  );
  const websocket = useContext(WebSocketContext).socket;

  // react state
  const [cameras, setCameras] = React.useState([]);
  const [slider_value, set_slider_value] = React.useState(0);
  const [smoothness_value, set_smoothness_value] = React.useState(0.5);
  const [is_playing, setIsPlaying] = React.useState(false);
  const [is_cycle, setIsCycle] = React.useState(false);
  const [is_linear, setIsLinear] = React.useState(false);
  const [seconds, setSeconds] = React.useState(4);
  const [fps, setFps] = React.useState(24);

  const dispatch = useDispatch();
  const render_height = useSelector(
    (state) => state.renderingState.render_height,
  );
  const render_width = useSelector(
    (state) => state.renderingState.render_width,
  );

  const field_of_view = useSelector(
    (state) => state.renderingState.field_of_view,
  );

  const setRenderHeight = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/render_height',
      data: value,
    });
  };
  const setRenderWidth = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/render_width',
      data: value,
    });
  };

  // ui state
  const [ui_render_height, setUIRenderHeight] = React.useState(render_height);
  const [ui_render_width, setUIRenderWidth] = React.useState(render_width);
  const [ui_field_of_view, setUIFieldOfView] = React.useState(field_of_view);
  const [ui_seconds, setUISeconds] = React.useState(seconds);
  const [ui_fps, setUIfps] = React.useState(fps);

  // nonlinear render option
  const slider_min = 0;
  const slider_max = Math.max(0, cameras.length - 1);

  // animation constants
  const total_num_steps = seconds * fps;
  const step_size = slider_max / total_num_steps;

  const reset_slider_render_on_add = (new_camera_list) => {
    // set slider and render camera back to 0
    if (new_camera_list.length >= 1) {
      set_camera_position(camera_render, new_camera_list[0].matrix);
      set_slider_value(slider_min);
    }
  };

  const add_camera = () => {
    const camera_main_copy = camera_main.clone();
    camera_main_copy.aspect = 1.0;
    const new_camera_list = cameras.concat(camera_main_copy);
    setCameras(new_camera_list);
    reset_slider_render_on_add(new_camera_list);
  };

  // force a rerender if the cameras are dragged around
  let update_cameras_interval = null;
  // eslint-disable-next-line no-unused-vars
  transform_controls.addEventListener('mouseDown', (event) => {
    // prevent multiple loops
    if (update_cameras_interval === null) {
      // hardcoded for 100 ms per udpate
      update_cameras_interval = setInterval(() => {}, 100);
    }
  });
  // eslint-disable-next-line no-unused-vars
  transform_controls.addEventListener('mouseUp', (event) => {
    if (update_cameras_interval !== null) {
      clearInterval(update_cameras_interval);
      update_cameras_interval = null;
      setCameras(cameras);
    }
  });

  // draw cameras and curve to the scene
  useEffect(() => {
    // draw the cameras

    const labels = Array.from(document.getElementsByClassName('label'));
    labels.forEach((label) => {
      label.remove();
    });

    sceneTree.delete(['Camera Path', 'Cameras']); // delete old cameras, which is important
    if (cameras.length < 1) {
      camera_render_helper.set_visibility(false);
    } else {
      camera_render_helper.set_visibility(true);
    }
    for (let i = 0; i < cameras.length; i += 1) {
      const camera = cameras[i];
      // camera.aspect = render_width / render_height;
      const camera_helper = new CameraHelper(camera, 0x393e46);

      const labelDiv = document.createElement('div');
      labelDiv.className = 'label';
      labelDiv.textContent = i;
      labelDiv.style.color = 'black';
      labelDiv.style.visibility = 'visible';
      const camera_label = new CSS2DObject(labelDiv);
      camera_label.position.set(0, -0.1, -0.1);
      camera_helper.add(camera_label);
      camera_label.layers.set(0);

      // camera
      sceneTree.set_object_from_path(
        ['Camera Path', 'Cameras', i.toString(), 'Camera'],
        camera,
      );
      // camera helper
      sceneTree.set_object_from_path(
        ['Camera Path', 'Cameras', i.toString(), 'Camera Helper'],
        camera_helper,
      );
    }
  }, [cameras, render_width, render_height]);

  // update the camera curve
  const curve_object = get_curve_object_from_cameras(
    cameras,
    is_cycle,
    smoothness_value,
  );

  if (cameras.length > 1) {
    const num_points = fps * seconds;
    const points = curve_object.curve_positions.getPoints(num_points);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const spline = new MeshLine();
    spline.setGeometry(geometry);
    const material = new MeshLineMaterial({ lineWidth: 0.01, color: 0xff5024 });
    const spline_mesh = new THREE.Mesh(spline.geometry, material);
    sceneTree.set_object_from_path(['Camera Path', 'Curve'], spline_mesh);

    // set the camera
    const point = Math.min(slider_value / (cameras.length - 1.0), 1);
    let position = null;
    let lookat = null;
    let up = null;
    if (!is_linear) {
      position = curve_object.curve_positions.getPoint(point);
      lookat = curve_object.curve_lookats.getPoint(point);
      up = curve_object.curve_ups.getPoint(point);
    } else {
      position = curve_object.curve_positions.getPointAt(point);
      lookat = curve_object.curve_lookats.getPointAt(point);
      up = curve_object.curve_ups.getPointAt(point);
    }
    const mat = get_transform_matrix(position, lookat, up);
    set_camera_position(camera_render, mat);
  } else {
    sceneTree.delete(['Camera Path', 'Curve']);
  }

  const marks = [];
  for (let i = 0; i < cameras.length; i += 1) {
    marks.push({ value: i, label: i.toString() });
  }

  // when the slider changes, update the main camera position
  useEffect(() => {
    if (cameras.length > 1) {
      const point = Math.min(slider_value / (cameras.length - 1.0), 1);
      let position = null;
      let lookat = null;
      let up = null;
      if (!is_linear) {
        // interpolate to get the points
        position = curve_object.curve_positions.getPoint(point);
        lookat = curve_object.curve_lookats.getPoint(point);
        up = curve_object.curve_ups.getPoint(point);
      } else {
        position = curve_object.curve_positions.getPointAt(point);
        lookat = curve_object.curve_lookats.getPointAt(point);
        up = curve_object.curve_ups.getPointAt(point);
      }
      const mat = get_transform_matrix(position, lookat, up);
      set_camera_position(camera_render, mat);
    }
  }, [slider_value, render_height, render_width]);

  // call this function whenever slider state changes
  useEffect(() => {
    if (is_playing && cameras.length > 1) {
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

  const get_camera_path = () => {
    // NOTE: currently assuming these are ints
    const num_points = fps * seconds;

    const positions = curve_object.curve_positions.getPoints(num_points);
    const lookats = curve_object.curve_lookats.getPoints(num_points);
    const ups = curve_object.curve_ups.getPoints(num_points);

    const camera_path = [];

    for (let i = 0; i < num_points; i += 1) {
      const position = positions[i];
      const lookat = lookats[i];
      const up = ups[i];

      const mat = get_transform_matrix(position, lookat, up);

      camera_path.push({
        camera_to_world: mat.transpose().elements, // convert from col-major to row-major matrix
        fov: camera_render.fov,
        aspect: camera_render.aspect,
      });
    }

    // const myData
    const camera_path_object = {
      keyframes: [],
      render_height,
      render_width,
      camera_path,
      fps,
      seconds,
    };
    return camera_path_object;
  };

  const export_camera_path = () => {
    // export the camera path
    // inspired by:
    // https://stackoverflow.com/questions/55613438/reactwrite-to-json-file-or-export-download-no-server

    const camera_path_object = get_camera_path();
    console.log(camera_render.toJSON());

    // create file in browser
    const json = JSON.stringify(camera_path_object, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const href = URL.createObjectURL(blob);

    // create "a" HTLM element with href to file
    const link = document.createElement('a');
    link.href = href;

    const filename = 'camera_path.json';
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    // clean up "a" element & remove ObjectURL
    document.body.removeChild(link);
    URL.revokeObjectURL(href);
  };

  const copy_cmd_to_clipboard = () => {
    console.log('copy_cmd_to_clipboard');

    const camera_path_object = get_camera_path();

    // Copy the text inside the text field
    const config_filename = `${config_base_dir}/config.yml`;
    const camera_path_filename = `${config_base_dir}/camera_path.json`;
    const cmd = `ns-render --load-config ${config_filename} --traj filename --camera-path-filename ${camera_path_filename} --output-path renders/output.mp4`;
    navigator.clipboard.writeText(cmd);

    const camera_path_payload = {
      camera_path_filename,
      camera_path: camera_path_object,
    };

    // send a command of the websocket to save the trajectory somewhere!
    if (websocket.readyState === WebSocket.OPEN) {
      const data = {
        type: 'write',
        path: 'camera_path_payload',
        data: camera_path_payload,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const setFOV = (fov) => {
    dispatch({
      type: 'write',
      path: 'renderingState/field_of_view',
      data: fov,
    });
  };

  const setUp = () => {
    const rot = camera_main.rotation;
    const unitY = new THREE.Vector3(0, 1, 0);
    const upVec = unitY.applyEuler(rot);

    const grid = sceneTree.find_object_no_create(['Grid']);
    grid.setRotationFromEuler(rot);

    const pos = new THREE.Vector3();
    camera_main.getWorldPosition(pos);
    camera_main.up.set(upVec.x, upVec.y, upVec.z);
    sceneTree.metadata.camera_controls.updateCameraUp();
    sceneTree.metadata.camera_controls.setLookAt(pos.x, pos.y, pos.z, 0, 0, 0);
    const points = [new THREE.Vector3(0, 0, 0), upVec.multiplyScalar(2)];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xaa46fc,
      linewidth: 1,
    });
    const line = new THREE.LineSegments(geometry, material);
    sceneTree.set_object_from_path(['Viewer Up Vector'], line);
  };

  return (
    <div className="CameraPanel">
      <div className="CameraPanel-top-button">
        <Button size="small" variant="outlined" onClick={setUp}>
          Reset Up Direction
        </Button>
      </div>
      <div>
        <div className="CameraPanel-top-button">
          <Button size="small" variant="outlined" onClick={add_camera}>
            Add Camera
          </Button>
        </div>
        <div className="CameraPanel-top-button">
          <Button
            size="small"
            className="CameraPanel-top-button"
            variant="outlined"
            onClick={export_camera_path}
          >
            Export Path
          </Button>
        </div>
        <div className="CameraPanel-top-button">
          <Tooltip title="Copy Cmd to Clipboard">
            <IconButton onClick={copy_cmd_to_clipboard}>
              <ContentPasteGoIcon />
            </IconButton>
          </Tooltip>
        </div>
      </div>
      <div>
        <div className="CameraPanel-top-button">
          <Tooltip className="curve-button" title="Close/Open camera spline">
            {!is_cycle ? (
              <Button
                size="small"
                variant="outlined"
                onClick={() => {
                  setIsCycle(true);
                }}
              >
                <GestureOutlined />
              </Button>
            ) : (
              <Button
                size="small"
                variant="outlined"
                onClick={() => {
                  setIsCycle(false);
                }}
              >
                <AllInclusiveOutlined />
              </Button>
            )}
          </Tooltip>
        </div>
        <div className="CameraPanel-top-button">
          <Tooltip title="Non-linear/Linear camera speed">
            {!is_linear ? (
              <Button
                size="small"
                variant="outlined"
                onClick={() => {
                  setIsLinear(true);
                }}
              >
                <LinearScaleOutlined />
              </Button>
            ) : (
              <Button
                size="small"
                variant="outlined"
                onClick={() => {
                  setIsLinear(false);
                }}
              >
                <Timeline />
              </Button>
            )}
          </Tooltip>
        </div>
      </div>
      <div
        className="CameraPanel-slider-container"
        style={{ marginTop: '5px' }}
      >
        <Stack spacing={2} direction="row" sx={{ mb: 1 }} alignItems="center">
          <p style={{ fontSize: 'smaller', color: '#999999' }}>Smoothness</p>
          <ChangeHistory />
          <Slider
            value={smoothness_value}
            step={step_size}
            valueLabelFormat={smoothness_value.toFixed(2)}
            min={0}
            max={1}
            onChange={(event, value) => {
              set_smoothness_value(value);
            }}
          />
          <RadioButtonUnchecked />
        </Stack>
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
          size="small"
          variant="outlined"
          onClick={() => {
            setIsPlaying(false);
            set_slider_value(slider_min);
          }}
        >
          <FirstPageIcon />
        </Button>
        <Button
          size="small"
          variant="outlined"
          onClick={() =>
            set_slider_value(Math.max(0.0, slider_value - step_size))
          }
        >
          <ArrowBackIosNewIcon />
        </Button>
        {/* eslint-disable-next-line no-nested-ternary */}
        {!is_playing && slider_max === slider_value ? (
          <Button
            size="small"
            variant="outlined"
            onClick={() => {
              set_slider_value(slider_min);
            }}
          >
            <Replay />
          </Button>
        ) : !is_playing ? (
          <Button
            size="small"
            variant="outlined"
            onClick={() => {
              if (cameras.length > 1) {
                setIsPlaying(true);
              }
            }}
          >
            <PlayArrowIcon />
          </Button>
        ) : (
          <Button
            size="small"
            variant="outlined"
            onClick={() => {
              setIsPlaying(false);
            }}
          >
            <PauseIcon />
          </Button>
        )}
        <Button
          size="small"
          variant="outlined"
          onClick={() =>
            set_slider_value(Math.min(slider_max, slider_value + step_size))
          }
        >
          <ArrowForwardIosIcon />
        </Button>
        <Button
          size="small"
          variant="outlined"
          onClick={() => set_slider_value(slider_max)}
        >
          <LastPageIcon />
        </Button>
      </div>
      <div className="CameraList-row-time-interval">
        <TextField
          label="Height"
          inputProps={{
            inputMode: 'numeric',
            pattern: '[+-]?([0-9]*[.])?[0-9]+',
          }}
          size="small"
          onChange={(e) => {
            if (e.target.validity.valid) {
              setUIRenderHeight(e.target.value);
            }
          }}
          onBlur={(e) => {
            if (e.target.validity.valid) {
              if (e.target.value !== '') {
                setRenderHeight(parseInt(e.target.value, 10));
              } else {
                setUIRenderHeight(render_height);
              }
            }
          }}
          value={ui_render_height}
          error={ui_render_height <= 0}
          helperText={ui_render_height <= 0 ? 'Required' : ''}
          variant="standard"
        />
        <TextField
          label="Width"
          inputProps={{
            inputMode: 'numeric',
            pattern: '[+-]?([0-9]*[.])?[0-9]+',
          }}
          size="small"
          onChange={(e) => {
            if (e.target.validity.valid) {
              setUIRenderWidth(e.target.value);
            }
          }}
          onBlur={(e) => {
            if (e.target.validity.valid) {
              if (e.target.value !== '') {
                setRenderWidth(parseInt(e.target.value, 10));
              } else {
                setUIRenderWidth(render_width);
              }
            }
          }}
          value={ui_render_width}
          error={ui_render_width <= 0}
          helperText={ui_render_width <= 0 ? 'Required' : ''}
          variant="standard"
        />
        <TextField
          label="FOV"
          inputProps={{
            inputMode: 'numeric',
            pattern: '[+-]?([0-9]*[.])?[0-9]+',
          }}
          onChange={(e) => {
            if (e.target.validity.valid) {
              setUIFieldOfView(e.target.value);
            }
          }}
          onBlur={(e) => {
            if (e.target.validity.valid) {
              if (e.target.value !== '') {
                setFOV(parseInt(e.target.value, 10));
              } else {
                setUIFieldOfView(field_of_view);
              }
            }
          }}
          value={ui_field_of_view}
          error={ui_field_of_view <= 0}
          helperText={ui_field_of_view <= 0 ? 'Required' : ''}
          variant="standard"
        />
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
              setUISeconds(e.target.value);
            }
          }}
          onBlur={(e) => {
            if (e.target.validity.valid) {
              if (e.target.value !== '') {
                setSeconds(parseInt(e.target.value, 10));
              } else {
                setUISeconds(seconds);
              }
            }
          }}
          value={ui_seconds}
          error={ui_seconds <= 0}
          helperText={ui_seconds <= 0 ? 'Required' : ''}
          variant="standard"
        />
        <TextField
          label="FPS"
          inputProps={{ inputMode: 'numeric', pattern: '[0-9]*' }}
          size="small"
          onChange={(e) => {
            if (e.target.validity.valid) {
              setUIfps(e.target.value);
            }
          }}
          onBlur={(e) => {
            if (e.target.validity.valid) {
              if (e.target.value !== '') {
                setFps(parseInt(e.target.value, 10));
              } else {
                setUIfps(fps);
              }
            }
          }}
          value={ui_fps}
          error={ui_fps <= 0}
          helperText={ui_fps <= 0 ? 'Required' : ''}
          variant="standard"
        />
      </div>
      <div className="CameraList-container">
        <CameraList
          sceneTree={sceneTree}
          transform_controls={transform_controls}
          camera_main={camera_render}
          cameras={cameras}
          setCameras={setCameras}
        />
      </div>
    </div>
  );
}
