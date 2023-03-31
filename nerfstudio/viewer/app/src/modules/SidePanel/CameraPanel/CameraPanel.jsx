import * as React from 'react';
import * as THREE from 'three';

import {
  ArrowBackIosNew,
  ArrowForwardIos,
  AllInclusiveOutlined,
  ChangeHistory,
  ClearAll,
  Delete,
  ExpandMore,
  FirstPage,
  GestureOutlined,
  KeyboardArrowUp,
  KeyboardArrowDown,
  LastPage,
  Pause,
  PlayArrow,
  RadioButtonUnchecked,
  Replay,
  Visibility,
  Edit,
  Animation,
} from '@mui/icons-material';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Button,
  InputAdornment,
  Slider,
} from '@mui/material';
import { MeshLine, MeshLineMaterial } from 'meshline';
import { useContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer';
import AddAPhotoIcon from '@mui/icons-material/AddAPhoto';
import FileDownloadOutlinedIcon from '@mui/icons-material/FileDownloadOutlined';
import { LevaPanel, LevaStoreProvider, useCreateStore } from 'leva';
import FileUploadOutlinedIcon from '@mui/icons-material/FileUploadOutlined';
import { Stack } from '@mui/system';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import VideoCameraBackIcon from '@mui/icons-material/VideoCameraBack';
import { CameraHelper } from './CameraHelper';
import { get_curve_object_from_cameras, get_transform_matrix } from './curve';
import { WebSocketContext } from '../../WebSocket/WebSocket';
import RenderModal from '../../RenderModal';
import LoadPathModal from '../../LoadPathModal';
import CameraPropPanel from './CameraPropPanel';
import LevaTheme from '../../../themes/leva_theme.json';

const msgpack = require('msgpack-lite');

const FOV_LABELS = {
  FOV: '°',
  MM: 'mm',
};

function set_camera_position(camera, matrix) {
  const mat = new THREE.Matrix4();
  mat.fromArray(matrix.elements);
  mat.decompose(camera.position, camera.quaternion, camera.scale);
}

function RenderTimeSelector(props) {
  const disabled = props.disabled;
  const isGlobal = props.isGlobal;
  const camera = props.camera;
  const dispatch = props.dispatch;
  const globalRenderTime = props.globalRenderTime;
  const setGlobalRenderTime = props.setGlobalRenderTime;
  const applyAll = props.applyAll;
  const changeMain = props.changeMain;
  const setAllCameraRenderTime = props.setAllCameraRenderTime;

  const getRenderTimeLabel = () => {
    if (!isGlobal) {
      return camera.renderTime;
    }
    camera.renderTime = globalRenderTime;
    return globalRenderTime;
  };

  const [UIRenderTime, setUIRenderTime] = React.useState(
    isGlobal ? globalRenderTime : getRenderTimeLabel(),
  );

  const [valid, setValid] = React.useState(true);

  useEffect(
    () => setUIRenderTime(getRenderTimeLabel()),
    [camera, globalRenderTime],
  );

  const setRndrTime = (val) => {
    if (!isGlobal) {
      camera.renderTime = val;
    } else {
      camera.renderTime = val;
      setGlobalRenderTime(val);
    }

    if (applyAll) {
      setAllCameraRenderTime(val);
    }

    if (changeMain) {
      dispatch({
        type: 'write',
        path: 'renderingState/render_time',
        data: camera.renderTime,
      });
    }
  };

  const handleValidation = (e) => {
    const valueFloat = parseFloat(e.target.value);
    let valueStr = String(valueFloat);
    if (e.target.value >= 0 && e.target.value <= 1) {
      setValid(true);
      if (valueFloat === 1.0) {
        valueStr = '1.0';
      }
      if (valueFloat === 0.0) {
        valueStr = '0.0';
      }
      setUIRenderTime(parseFloat(valueStr));
      setRndrTime(parseFloat(valueStr));
    } else {
      setValid(false);
    }
  };

  return (
    <TextField
      label="Render Time"
      InputLabelProps={{
        style: { color: '#8E8E8E' },
      }}
      inputProps={{
        inputMode: 'numeric',
      }}
      onChange={(e) => setUIRenderTime(e.target.value)}
      onBlur={(e) => handleValidation(e)}
      disabled={disabled}
      sx={{
        input: {
          WebkitTextFillColor: `${disabled ? '#24B6FF' : '#EBEBEB'} !important`,
          color: `${disabled ? '#24B6FF' : '#EBEBEB'} !important`,
        },
      }}
      value={UIRenderTime}
      error={!valid}
      helperText={!valid ? 'RenderTime should be between 0.0 and 1.0' : ''}
      variant="standard"
    />
  );
}

function FovSelector(props) {
  const fovLabel = props.fovLabel;
  const setFovLabel = props.setFovLabel;
  const camera = props.camera;
  const dispatch = props.dispatch;
  const changeMain = props.changeMain;
  const disabled = props.disabled;
  const applyAll = props.applyAll;
  const setAllCameraFOV = props.setAllCameraFOV;
  const isGlobal = props.isGlobal;
  const globalFov = props.globalFov;
  const setGlobalFov = props.setGlobalFov;

  const getFovLabel = () => {
    if (!isGlobal) {
      const label = Math.round(
        fovLabel === FOV_LABELS.FOV
          ? camera.getEffectiveFOV()
          : camera.getFocalLength() * camera.aspect,
      );
      return label;
    }
    const old_fov = camera.fov;
    camera.fov = globalFov;
    const new_focal_len = camera.getFocalLength();
    camera.fov = old_fov;
    const label = Math.round(
      fovLabel === FOV_LABELS.FOV ? globalFov : new_focal_len * camera.aspect,
    );
    return label;
  };

  const [UIFieldOfView, setUIFieldOfView] = React.useState(
    isGlobal ? globalFov : getFovLabel(),
  );

  useEffect(
    () => setUIFieldOfView(getFovLabel()),
    [camera, fovLabel, globalFov],
  );

  const setFOV = (val) => {
    if (!isGlobal) {
      if (fovLabel === FOV_LABELS.FOV) {
        camera.fov = val;
      } else {
        camera.setFocalLength(val / camera.aspect);
      }
    } else if (fovLabel === FOV_LABELS.FOV) {
      camera.fov = val;
      setGlobalFov(val);
    } else {
      camera.setFocalLength(val / camera.aspect);
      const new_fov = camera.getEffectiveFOV();
      camera.fov = new_fov;
      setGlobalFov(new_fov);
    }

    if (applyAll) {
      setAllCameraFOV(val);
    }

    if (changeMain) {
      dispatch({
        type: 'write',
        path: 'renderingState/field_of_view',
        data: camera.getEffectiveFOV(),
      });
    }
  };

  const toggleFovLabel = () => {
    if (fovLabel === FOV_LABELS.FOV) {
      setFovLabel(FOV_LABELS.MM);
    } else {
      setFovLabel(FOV_LABELS.FOV);
    }
  };

  return (
    <TextField
      label={fovLabel === FOV_LABELS.FOV ? 'FOV' : 'Focal Length'}
      InputLabelProps={{
        style: { color: '#8E8E8E' },
      }}
      inputProps={{
        inputMode: 'numeric',
        pattern: '[+-]?([0-9]*[.])?[0-9]+',
        // style: { color: 'rgb(50, 50, 50)' }
      }}
      disabled={disabled}
      // eslint-disable-next-line
      InputProps={{
        endAdornment: (
          <Tooltip title="Switch between FOV and Focal Length">
            <InputAdornment
              sx={{ cursor: 'pointer' }}
              onClick={toggleFovLabel}
              position="end"
            >
              {fovLabel === FOV_LABELS.FOV ? '°' : 'mm'}
            </InputAdornment>
          </Tooltip>
        ),
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
            setUIFieldOfView(getFovLabel());
          }
        }
      }}
      sx={{
        input: {
          WebkitTextFillColor: `${disabled ? '#24B6FF' : '#EBEBEB'} !important`,
          color: `${disabled ? '#24B6FF' : '#EBEBEB'} !important`,
        },
      }}
      value={UIFieldOfView}
      error={camera.fov <= 0}
      helperText={camera.fov <= 0 ? 'Required' : ''}
      variant="standard"
    />
  );
}

function CameraList(props) {
  const sceneTree = props.sceneTree;
  const cameras = props.cameras;
  const camera_main = props.camera_main;
  const transform_controls = props.transform_controls;
  const setCameras = props.setCameras;
  const swapCameras = props.swapCameras;
  const fovLabel = props.fovLabel;
  const setFovLabel = props.setFovLabel;
  const cameraProperties = props.cameraProperties;
  const setCameraProperties = props.setCameraProperties;
  const isAnimated = props.isAnimated;
  const dispatch = props.dispatch;
  // eslint-disable-next-line no-unused-vars
  const slider_value = props.slider_value;
  const set_slider_value = props.set_slider_value;

  const [expanded, setExpanded] = React.useState(null);

  const camera_type = useSelector((state) => state.renderingState.camera_type);

  const handleChange =
    (cameraUUID: string) =>
    (event: React.SyntheticEvent, isExpanded: boolean) => {
      setExpanded(isExpanded ? cameraUUID : false);
    };

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

  const reset_slider_render_on_change = () => {
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
      camera_render.fov = first_camera.fov;
      camera_render.renderTime = first_camera.renderTime;
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
    reset_slider_render_on_change();
  };

  // TODO: Add pencil for editing?
  const cameraList = cameras.map((camera, index) => {
    return (
      <Accordion
        className="CameraList-row"
        key={camera.uuid}
        expanded={expanded === camera.uuid}
        onChange={handleChange(camera.uuid)}
      >
        <AccordionSummary
          expandIcon={<ExpandMore sx={{ color: '#eeeeee' }} />}
          aria-controls="panel1bh-content"
          id="panel1bh-header"
        >
          <Stack spacing={0}>
            <Button
              size="small"
              onClick={(e) => {
                swapCameras(index, index - 1);
                e.stopPropagation();
              }}
              style={{
                maxWidth: '20px',
                maxHeight: '20px',
                minWidth: '20px',
                minHeight: '20px',
              }}
              disabled={index === 0}
            >
              <KeyboardArrowUp />
            </Button>
            <Button
              size="small"
              onClick={(e) => {
                swapCameras(index, index + 1);
                e.stopPropagation();
              }}
              style={{
                maxWidth: '20px',
                maxHeight: '20px',
                minWidth: '20px',
                minHeight: '20px',
              }}
              disabled={index === cameras.length - 1}
            >
              <KeyboardArrowDown />
            </Button>
          </Stack>
          <Button size="small" sx={{ ml: '3px' }}>
            <TextField
              id="standard-basic"
              value={camera.properties.get('NAME')}
              variant="standard"
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => {
                const cameraProps = new Map(cameraProperties);
                cameraProps.get(camera.uuid).set('NAME', e.target.value);
                setCameraProperties(cameraProps);
              }}
              sx={{
                alignItems: 'center',
                alignContent: 'center',
              }}
            />
          </Button>
          <Button
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              set_transform_controls(index);
            }}
          >
            <Edit />
          </Button>
          <Stack spacing={0} direction="row" justifyContent="end">
            <Button
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                set_camera_position(camera_main, camera.matrix);
                camera_main.fov = camera.fov;
                camera_main.renderTime = camera.renderTime;
                set_slider_value(camera.properties.get('TIME'));
              }}
            >
              <Visibility />
            </Button>
            <Button size="small" onClick={() => delete_camera(index)}>
              <Delete />
            </Button>
          </Stack>
        </AccordionSummary>
        <AccordionDetails>
          {isAnimated('FOV') && camera_type !== 'equirectangular' && (
            <FovSelector
              fovLabel={fovLabel}
              setFovLabel={setFovLabel}
              camera={camera}
              dispatch={dispatch}
              disabled={!isAnimated('FOV')}
              isGlobal={false}
              changeMain={false}
            />
          )}
          {isAnimated('RenderTime') && (
            <RenderTimeSelector
              camera={camera}
              dispatch={dispatch}
              disabled={!isAnimated('RenderTime')}
              isGlobal={false}
              changeMain={false}
            />
          )}
          {!isAnimated('FOV') && !isAnimated('RenderTime') && (
            <p style={{ fontSize: 'smaller', color: '#999999' }}>
              Animated camera properties will show up here!
            </p>
          )}
        </AccordionDetails>
      </Accordion>
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
  const export_path = useSelector((state) => state.renderingState.export_path);

  const websocket = useContext(WebSocketContext).socket;
  const DEFAULT_FOV = 50;
  const DEFAULT_RENDER_TIME = 0.0;

  // react state
  const [cameras, setCameras] = React.useState([]);
  // Mapping of camera id to each camera's properties
  const [cameraProperties, setCameraProperties] = React.useState(new Map());
  const [slider_value, set_slider_value] = React.useState(0);
  const [smoothness_value, set_smoothness_value] = React.useState(0.5);
  const [is_playing, setIsPlaying] = React.useState(false);
  const [is_cycle, setIsCycle] = React.useState(false);
  const [seconds, setSeconds] = React.useState(4);
  const [fps, setFps] = React.useState(24);
  const [render_modal_open, setRenderModalOpen] = React.useState(false);
  const [load_path_modal_open, setLoadPathModalOpen] = React.useState(false);
  const [animate, setAnimate] = React.useState(new Set());
  const [globalFov, setGlobalFov] = React.useState(DEFAULT_FOV);
  const [globalRenderTime, setGlobalRenderTime] =
    React.useState(DEFAULT_RENDER_TIME);

  // leva store
  const cameraPropsStore = useCreateStore();

  const scene_state = sceneTree.get_scene_state();

  // Template for sharing state between Vanilla JS Three.js and React components
  // eslint-disable-next-line no-unused-vars
  const [mouseInScene, setMouseInScene] = React.useState(false);
  React.useEffect(() => {
    scene_state.addCallback(
      (value) => setMouseInScene(value),
      'mouse_in_scene',
    );
  }, []);

  const dispatch = useDispatch();
  const render_height = useSelector(
    (state) => state.renderingState.render_height,
  );
  const render_width = useSelector(
    (state) => state.renderingState.render_width,
  );
  const camera_type = useSelector((state) => state.renderingState.camera_type);

  const crop_enabled = useSelector(
    (state) => state.renderingState.crop_enabled,
  );
  const crop_bg_color = useSelector(
    (state) => state.renderingState.crop_bg_color,
  );
  const crop_center = useSelector((state) => state.renderingState.crop_center);
  const crop_scale = useSelector((state) => state.renderingState.crop_scale);

  const [display_render_time, set_display_render_time] = React.useState(false);

  const receive_temporal_dist = (e) => {
    const msg = msgpack.decode(new Uint8Array(e.data));
    if (msg.path === '/model/has_temporal_distortion') {
      set_display_render_time(msg.data === 'true');
      websocket.removeEventListener('message', receive_temporal_dist);
    }
  };
  websocket.addEventListener('message', receive_temporal_dist);

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

  const setCameraType = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/camera_type',
      data: value,
    });
  };

  const setFieldOfView = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/field_of_view',
      data: value,
    });
  };

  const setCropEnabled = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/crop_enabled',
      data: value,
    });
  };

  const serCropBgColor = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/crop_bg_color',
      data: value,
    });
  };

  const setCropCenter = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/crop_center',
      data: value,
    });
  };

  const setCropScale = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/crop_scale',
      data: value,
    });
  };

  const setRenderTime = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/render_time',
      data: parseFloat(value),
    });
  };

  // ui state
  const [fovLabel, setFovLabel] = React.useState(FOV_LABELS.FOV);

  // nonlinear render option
  const slider_min = 0;
  const slider_max = 1;

  // animation constants
  const total_num_steps = seconds * fps;
  const step_size = slider_max / total_num_steps;

  const reset_slider_render_on_add = (new_camera_list) => {
    // set slider and render camera back to 0
    if (new_camera_list.length >= 1) {
      set_camera_position(camera_render, new_camera_list[0].matrix);
      setFieldOfView(new_camera_list[0].fov);
      setRenderTime(new_camera_list[0].renderTime);
      set_slider_value(slider_min);
    }
  };

  const add_camera = () => {
    const camera_main_copy = camera_main.clone();
    camera_main_copy.aspect = 1.0;
    camera_main_copy.fov = globalFov;
    camera_main_copy.renderTime = globalRenderTime;
    const new_camera_properties = new Map();
    camera_main_copy.properties = new_camera_properties;
    new_camera_properties.set('FOV', globalFov);
    new_camera_properties.set('NAME', `Camera ${cameras.length}`);
    // TIME VALUES ARE 0-1
    if (cameras.length === 0) {
      new_camera_properties.set('TIME', 0.0);
    } else {
      new_camera_properties.set('TIME', 1.0);
    }

    const ratio = (cameras.length - 1) / cameras.length;

    const new_properties = new Map(cameraProperties);
    new_properties.forEach((properties) => {
      properties.set('TIME', properties.get('TIME') * ratio);
    });
    new_properties.set(camera_main_copy.uuid, new_camera_properties);
    setCameraProperties(new_properties);

    const new_camera_list = cameras.concat(camera_main_copy);
    setCameras(new_camera_list);
    reset_slider_render_on_add(new_camera_list);
  };

  const setCameraProperty = (property, value, index) => {
    const activeCamera = cameras[index];
    const activeProperties = new Map(activeCamera.properties);
    activeProperties.set(property, value);
    const newProperties = new Map(cameraProperties);
    newProperties.set(activeCamera.uuid, activeProperties);
    activeCamera.properties = activeProperties;
    setCameraProperties(newProperties);
  };

  const swapCameras = (index, new_index) => {
    if (
      Math.min(index, new_index) < 0 ||
      Math.max(index, new_index) >= cameras.length
    )
      return;

    const swapCameraTime = cameras[index].properties.get('TIME');
    setCameraProperty('TIME', cameras[new_index].properties.get('TIME'), index);
    setCameraProperty('TIME', swapCameraTime, new_index);

    const new_cameras = [
      ...cameras.slice(0, index),
      ...cameras.slice(index + 1),
    ];
    setCameras([
      ...new_cameras.slice(0, new_index),
      cameras[index],
      ...new_cameras.slice(new_index),
    ]);

    // reset_slider_render_on_change();
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
      dispatch({
        type: 'write',
        path: 'renderingState/camera_choice',
        data: 'Main Camera',
      });
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
      labelDiv.textContent = camera.properties.get('NAME');
      labelDiv.style.color = 'black';
      labelDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.61)';
      labelDiv.style.backdropFilter = 'blur(5px)';
      labelDiv.style.padding = '6px';
      labelDiv.style.borderRadius = '6px';
      labelDiv.style.visibility = 'visible';
      const camera_label = new CSS2DObject(labelDiv);
      camera_label.name = 'CAMERA_LABEL';
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
  }, [cameras, cameraProperties, render_width, render_height]);

  // update the camera curve
  const curve_object = get_curve_object_from_cameras(
    cameras,
    is_cycle,
    smoothness_value,
  );

  const getKeyframePoint = (progress: Number) => {
    const times = [];
    const ratio = (cameras.length - 1) / cameras.length;
    cameras.forEach((camera) => {
      const time = camera.properties.get('TIME');
      times.push(is_cycle ? time * ratio : time);
    });

    if (is_cycle) {
      times.push(1.0);
    }

    let new_point = 0.0;
    if (progress <= times[0]) {
      new_point = 0.0;
    } else if (progress >= times[times.length - 1]) {
      new_point = 1.0;
    } else {
      let i = 0;
      while (
        i < times.length - 1 &&
        !(progress >= times[i] && progress < times[i + 1])
      ) {
        i += 1;
      }
      const percentage = (progress - times[i]) / (times[i + 1] - times[i]);
      new_point = (i + percentage) / (times.length - 1);
    }
    return new_point;
  };

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

    const point = getKeyframePoint(slider_value);
    let position = null;
    let lookat = null;
    let up = null;
    let fov = null;
    position = curve_object.curve_positions.getPoint(point);
    lookat = curve_object.curve_lookats.getPoint(point);
    up = curve_object.curve_ups.getPoint(point);
    fov = curve_object.curve_fovs.getPoint(point).z;

    const mat = get_transform_matrix(position, lookat, up);
    set_camera_position(camera_render, mat);
    setFieldOfView(fov);
  } else {
    sceneTree.delete(['Camera Path', 'Curve']);
  }

  const marks = [];
  for (let i = 0; i <= 1; i += 0.25) {
    marks.push({ value: i, label: `${(seconds * i).toFixed(1).toString()}s` });
  }

  const values = [];
  cameras.forEach((camera) => {
    const time = camera.properties.get('TIME');
    const ratio = (cameras.length - 1) / cameras.length;
    values.push(is_cycle ? time * ratio : time);
  });

  if (is_cycle && cameras.length !== 0) {
    values.push(1.0);
  }

  const handleKeyframeSlider = (
    event: Event,
    newValue: number | number[],
    activeThumb: number,
  ) => {
    if (activeThumb === cameras.length) return;
    const ratio = (cameras.length - 1) / cameras.length;
    const val = newValue[activeThumb];
    setCameraProperty(
      'TIME',
      is_cycle ? Math.min(val / ratio, 1.0) : val,
      activeThumb,
    );
  };

  // when the slider changes, update the main camera position
  useEffect(() => {
    if (cameras.length > 1) {
      const point = getKeyframePoint(slider_value);
      let position = null;
      let lookat = null;
      let up = null;
      let fov = null;
      position = curve_object.curve_positions.getPoint(point);
      lookat = curve_object.curve_lookats.getPoint(point);
      up = curve_object.curve_ups.getPoint(point);
      fov = curve_object.curve_fovs.getPoint(point).z;
      const mat = get_transform_matrix(position, lookat, up);
      set_camera_position(camera_render, mat);
      setFieldOfView(fov);
      setGlobalFov(fov);
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
    const camera_path = [];

    for (let i = 0; i < num_points; i += 1) {
      const pt = getKeyframePoint(i / num_points);

      const position = curve_object.curve_positions.getPoint(pt);
      const lookat = curve_object.curve_lookats.getPoint(pt);
      const up = curve_object.curve_ups.getPoint(pt);
      const fov = curve_object.curve_fovs.getPoint(pt).z;

      const mat = get_transform_matrix(position, lookat, up);

      if (display_render_time) {
        const renderTime = curve_object.curve_render_times.getPoint(pt).z;
        camera_path.push({
          camera_to_world: mat.transpose().elements, // convert from col-major to row-major matrix
          fov,
          aspect: camera_render.aspect,
          render_time: Math.max(Math.min(renderTime, 1.0), 0.0), // clamp time values to [0, 1]
        });
      } else {
        camera_path.push({
          camera_to_world: mat.transpose().elements, // convert from col-major to row-major matrix
          fov,
          aspect: camera_render.aspect,
        });
      }
    }

    const keyframes = [];
    for (let i = 0; i < cameras.length; i += 1) {
      const camera = cameras[i];
      keyframes.push({
        matrix: JSON.stringify(camera.matrix.toArray()),
        fov: camera.fov,
        aspect: camera_render.aspect,
        properties: JSON.stringify(Array.from(camera.properties.entries())),
      });
    }

    let crop = null;
    if (crop_enabled) {
      crop = {
        crop_bg_color,
        crop_center,
        crop_scale,
      };
    }

    // const myData
    const camera_path_object = {
      keyframes,
      camera_type,
      render_height,
      render_width,
      camera_path,
      fps,
      seconds,
      smoothness_value,
      is_cycle,
      crop,
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

  const load_camera_path = (camera_path_object) => {
    const new_camera_list = [];
    const new_properties = new Map(cameraProperties);

    setRenderHeight(camera_path_object.render_height);
    setRenderWidth(camera_path_object.render_width);
    setCameraType(camera_path_object.camera_type);

    setFps(camera_path_object.fps);
    setSeconds(camera_path_object.seconds);

    set_smoothness_value(camera_path_object.smoothness_value);
    setIsCycle(camera_path_object.is_cycle);

    for (let i = 0; i < camera_path_object.keyframes.length; i += 1) {
      const keyframe = camera_path_object.keyframes[i];
      const camera = new THREE.PerspectiveCamera(
        keyframe.fov,
        keyframe.aspect,
        0.1,
        1000,
      );

      // properties
      camera.properties = new Map(JSON.parse(keyframe.properties));
      new_properties.set(camera.uuid, camera.properties);

      const mat = new THREE.Matrix4();
      mat.fromArray(JSON.parse(keyframe.matrix));
      set_camera_position(camera, mat);
      new_camera_list.push(camera);
    }

    setCameraProperties(new_properties);
    setCameras(new_camera_list);
    reset_slider_render_on_add(new_camera_list);

    if ('crop' in camera_path_object && camera_path_object.crop !== null) {
      setCropEnabled(true);
      serCropBgColor(camera_path_object.crop.crop_bg_color);
      setCropCenter(camera_path_object.crop.crop_center);
      setCropScale(camera_path_object.crop.crop_scale);
    }
  };

  const uploadCameraPath = (e) => {
    const fileUpload = e.target.files[0];

    const fr = new FileReader();
    fr.onload = (res) => {
      const camera_path_object = JSON.parse(res.target.result);
      load_camera_path(camera_path_object);
    };

    fr.readAsText(fileUpload);
  };

  const open_render_modal = () => {
    setRenderModalOpen(true);

    const camera_path_object = get_camera_path();
    const camera_path_payload = {
      camera_path_filename: export_path,
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

  const open_load_path_modal = () => {
    if (websocket.readyState === WebSocket.OPEN) {
      const data = {
        type: 'write',
        path: 'populate_paths_payload',
        data: true,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
    setLoadPathModalOpen(true);
  };

  const isAnimated = (property) => animate.has(property);

  const toggleAnimate = (property) => {
    const new_animate = new Set(animate);
    if (animate.has(property)) {
      new_animate.delete(property);
      setAnimate(new_animate);
    } else {
      new_animate.add(property);
      setAnimate(new_animate);
    }
  };

  const setAllCameraFOV = (val) => {
    if (fovLabel === FOV_LABELS.FOV) {
      for (let i = 0; i < cameras.length; i += 1) {
        cameras[i].fov = val;
      }
    } else {
      for (let i = 0; i < cameras.length; i += 1) {
        cameras[i].setFocalLength(val / cameras[i].aspect);
      }
    }
  };

  const setAllCameraRenderTime = (val) => {
    for (let i = 0; i < cameras.length; i += 1) {
      cameras[i].renderTime = val;
    }
  };

  return (
    <div className="CameraPanel">
      <div>
        <div className="CameraPanel-path-row">
          <LoadPathModal
            open={load_path_modal_open}
            setOpen={setLoadPathModalOpen}
            pathUploadFunction={uploadCameraPath}
            loadCameraPathFunction={load_camera_path}
          />
          <Button
            size="small"
            className="CameraPanel-top-button"
            component="label"
            variant="outlined"
            startIcon={<FileUploadOutlinedIcon />}
            onClick={open_load_path_modal}
          >
            Load Path
          </Button>
        </div>
        <div className="CameraPanel-path-row">
          <Button
            size="small"
            className="CameraPanel-top-button"
            variant="outlined"
            startIcon={<FileDownloadOutlinedIcon />}
            onClick={export_camera_path}
            disabled={cameras.length === 0}
          >
            Export Path
          </Button>
        </div>
        <br />
        <RenderModal open={render_modal_open} setOpen={setRenderModalOpen} />
        <Button
          className="CameraPanel-render-button"
          variant="outlined"
          size="small"
          startIcon={<VideoCameraBackIcon />}
          onClick={open_render_modal}
          disabled={cameras.length === 0}
        >
          Render
        </Button>
      </div>
      <div className="CameraPanel-props">
        <LevaPanel
          store={cameraPropsStore}
          className="Leva-panel"
          theme={LevaTheme}
          titleBar={false}
          fill
          flat
        />
        <LevaStoreProvider store={cameraPropsStore}>
          <CameraPropPanel
            seconds={seconds}
            set_seconds={setSeconds}
            fps={fps}
            set_fps={setFps}
          />
        </LevaStoreProvider>
      </div>
      {display_render_time && (
        <div className="CameraList-row-animation-properties">
          <Tooltip title="Animate Render Time for Each Camera">
            <Button
              value="animateRenderTime"
              selected={isAnimated('RenderTime')}
              onClick={() => {
                toggleAnimate('RenderTime');
              }}
              style={{
                maxWidth: '20px',
                maxHeight: '20px',
                minWidth: '20px',
                minHeight: '20px',
                position: 'relative',
                top: '22px',
              }}
              sx={{
                mt: 1,
              }}
            >
              <Animation
                style={{
                  color: isAnimated('RenderTime') ? '#24B6FF' : '#EBEBEB',
                  maxWidth: '20px',
                  maxHeight: '20px',
                  minWidth: '20px',
                  minHeight: '20px',
                }}
              />
            </Button>
          </Tooltip>
          <RenderTimeSelector
            disabled={false}
            isGlobal
            camera={camera_main}
            dispatch={dispatch}
            globalRenderTime={globalRenderTime}
            setGlobalRenderTime={setGlobalRenderTime}
            applyAll={!isAnimated('RenderTime')}
            setAllCameraRenderTime={setAllCameraRenderTime}
            changeMain
          />
        </div>
      )}
      {camera_type !== 'equirectangular' && (
        <div className="CameraList-row-animation-properties">
          <Tooltip title="Animate FOV for Each Camera">
            <Button
              value="animatefov"
              selected={isAnimated('FOV')}
              onClick={() => {
                toggleAnimate('FOV');
              }}
              style={{
                maxWidth: '20px',
                maxHeight: '20px',
                minWidth: '20px',
                minHeight: '20px',
                position: 'relative',
                top: '22px',
              }}
              sx={{
                mt: 1,
              }}
            >
              <Animation
                style={{
                  color: isAnimated('FOV') ? '#24B6FF' : '#EBEBEB',
                  maxWidth: '20px',
                  maxHeight: '20px',
                  minWidth: '20px',
                  minHeight: '20px',
                }}
              />
            </Button>
          </Tooltip>
          <FovSelector
            fovLabel={fovLabel}
            setFovLabel={setFovLabel}
            camera={camera_main}
            cameras={cameras}
            dispatch={dispatch}
            disabled={isAnimated('FOV')}
            applyAll={!isAnimated('FOV')}
            isGlobal
            globalFov={globalFov}
            setGlobalFov={setGlobalFov}
            setAllCameraFOV={setAllCameraFOV}
            changeMain
          />
        </div>
      )}
      <div>
        <div className="CameraPanel-row">
          <Button
            size="small"
            variant="outlined"
            startIcon={<AddAPhotoIcon />}
            onClick={add_camera}
          >
            Add Camera
          </Button>
        </div>
        <div className="CameraPanel-row">
          <Tooltip
            className="curve-button"
            title="Toggle looping camera spline"
          >
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
        <div className="CameraPanel-row">
          <Tooltip title="Reset Keyframe Timing">
            <Button
              size="small"
              variant="outlined"
              onClick={() => {
                const new_properties = new Map(cameraProperties);
                cameras.forEach((camera, i) => {
                  const uuid = camera.uuid;
                  const new_time = i / (cameras.length - 1);
                  const current_cam_properties = new_properties.get(uuid);
                  current_cam_properties.set('TIME', new_time);
                });
                setCameraProperties(new_properties);
              }}
            >
              <ClearAll />
            </Button>
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
        <b style={{ fontSize: 'smaller', color: '#999999', textAlign: 'left' }}>
          Camera Keyframes
        </b>
        <Slider
          value={values}
          step={step_size}
          valueLabelDisplay="auto"
          valueLabelFormat={(value, i) => {
            if (cameras.length === 0) {
              return '';
            }
            if (i === cameras.length && is_cycle) {
              return `${cameras[0].properties.get('NAME')} @ ${parseFloat(
                (value * seconds).toFixed(2),
              )}s`;
            }
            return `${cameras[i].properties.get('NAME')} @ ${parseFloat(
              (value * seconds).toFixed(2),
            )}s`;
          }}
          marks={marks}
          min={slider_min}
          max={slider_max}
          disabled={cameras.length < 2}
          track={false}
          onChange={handleKeyframeSlider}
          sx={{
            '& .MuiSlider-thumb': {
              borderRadius: '6px',
              width: `${24.0 / Math.max(Math.sqrt(cameras.length), 2)}px`,
            },
          }}
          disableSwap
        />
        <b style={{ fontSize: 'smaller', color: '#999999', textAlign: 'left' }}>
          Playback
        </b>
        <Slider
          value={slider_value}
          step={step_size}
          valueLabelDisplay={is_playing ? 'on' : 'off'}
          valueLabelFormat={`${(Math.min(slider_value, 1.0) * seconds).toFixed(
            2,
          )}s`}
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
          <FirstPage />
        </Button>
        <Button
          size="small"
          variant="outlined"
          onClick={() =>
            set_slider_value(Math.max(0.0, slider_value - step_size))
          }
        >
          <ArrowBackIosNew />
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
            <PlayArrow />
          </Button>
        ) : (
          <Button
            size="small"
            variant="outlined"
            onClick={() => {
              setIsPlaying(false);
            }}
          >
            <Pause />
          </Button>
        )}
        <Button
          size="small"
          variant="outlined"
          onClick={() =>
            set_slider_value(Math.min(slider_max, slider_value + step_size))
          }
        >
          <ArrowForwardIos />
        </Button>
        <Button
          size="small"
          variant="outlined"
          onClick={() => set_slider_value(slider_max)}
        >
          <LastPage />
        </Button>
      </div>
      <div className="CameraList-container">
        <CameraList
          sceneTree={sceneTree}
          transform_controls={transform_controls}
          camera_main={camera_render}
          cameras={cameras}
          setCameras={setCameras}
          swapCameras={swapCameras}
          cameraProperties={cameraProperties}
          setCameraProperties={setCameraProperties}
          fovLabel={fovLabel}
          setFovLabel={setFovLabel}
          isAnimated={isAnimated}
          dispatch={dispatch}
          slider_value={slider_value}
          set_slider_value={set_slider_value}
        />
      </div>
    </div>
  );
}
