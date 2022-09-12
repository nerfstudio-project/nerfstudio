import * as THREE from 'three';

import React, { useContext, useEffect, useRef } from 'react';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import { useDispatch, useSelector } from 'react-redux';

import FormControl from '@mui/material/FormControl';
import { IconButton } from '@mui/material';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import OpenWithIcon from '@mui/icons-material/OpenWith';
import PublicOffSharpIcon from '@mui/icons-material/PublicOffSharp';
import PublicSharpIcon from '@mui/icons-material/PublicSharp';
import Stats from 'stats.js';
import SyncOutlinedIcon from '@mui/icons-material/SyncOutlined';
import WebRtcWindow from '../WebRtcWindow/WebRtcWindow';
import { WebSocketContext } from '../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

function createStats() {
  const stats = new Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0';
  stats.domElement.style.top = '0';
  return stats;
}

function CameraDropdown() {
  const dispatch = useDispatch();
  const camera_options = useSelector(
    (state) => state.renderingState.camera_options,
  );
  const camera_choice = useSelector(
    (state) => state.renderingState.camera_choice,
  );
  const set_camera_choice = (event: SelectChangeEvent) => {
    const value = event.target.value;
    dispatch({
      type: 'write',
      path: 'renderingState/camera_choice',
      data: value,
    });
  };

  const menu_items = camera_options.map((camera_option) => (
    <MenuItem key={camera_option} value={camera_option}>
      {camera_option}
    </MenuItem>
  ));

  return (
    <FormControl sx={{ m: 1, minWidth: 120 }} size="small">
      <InputLabel>Camera</InputLabel>
      <Select value={camera_choice} label="Age" onChange={set_camera_choice}>
        {menu_items}
      </Select>
    </FormControl>
  );
}

function TransformIcons(props) {
  const sceneTree = props.sceneTree;
  const transform_controls = sceneTree.find_object(['Transform Controls']);
  // toggle back and forth between local and global transform
  const [world, setWorld] = React.useState(true);

  const toggleLocal = () => {
    transform_controls.setSpace(world ? 'local' : 'world');
    setWorld(!world);
  };

  return (
    <div>
      <div className="ViewerWindow-iconbutton">
        <IconButton
          size="large"
          onClick={() => {
            transform_controls.setMode('translate');
          }}
        >
          {/* translate */}
          <OpenWithIcon />
        </IconButton>
      </div>
      <div className="ViewerWindow-iconbutton">
        <IconButton
          size="large"
          onClick={() => {
            transform_controls.setMode('rotate');
          }}
        >
          {/* rotate */}
          <SyncOutlinedIcon />
        </IconButton>
      </div>
      <div className="ViewerWindow-iconbutton">
        <IconButton size="large" onClick={toggleLocal}>
          {world ? <PublicSharpIcon /> : <PublicOffSharpIcon />}
        </IconButton>
      </div>
    </div>
  );
}

// manages a camera and the web rtc stream...
export default function ViewerWindow(props) {
  const sceneTree = props.sceneTree;
  const scene = sceneTree.object;
  const renderer = sceneTree.metadata.renderer;

  const myRef = useRef(null);
  const websocket = useContext(WebSocketContext).socket;
  const field_of_view = useSelector(
    (state) => state.renderingState.field_of_view,
  );
  const field_of_view_ref = useRef(field_of_view);

  const camera_choice = useSelector(
    (state) => state.renderingState.camera_choice,
  );

  // on change, update the camera and controls
  sceneTree.metadata.camera = sceneTree.find_object(['Cameras', camera_choice]);
  // sceneTree.metadata.camera_controls.object = sceneTree.metadata.camera;

  let viewportWidth = null;
  let viewportHeight = null;
  let stats = null;

  const get_window_width = () => {
    const width = myRef.current.clientWidth;
    return width - (width % 2);
  };

  const get_window_height = () => {
    return myRef.current.clientHeight;
  };

  const handleResize = () => {
    viewportWidth = get_window_width();
    viewportHeight = get_window_height();
    sceneTree.metadata.camera.aspect = viewportWidth / viewportHeight;
    sceneTree.metadata.camera.updateProjectionMatrix();
    renderer.setSize(viewportWidth, viewportHeight);
  };

  // update the camera information in the python server
  const sendCamera = () => {
    if (websocket.readyState === WebSocket.OPEN) {
      const cmd = 'write';
      const path = 'renderingState/camera';
      const data = {
        type: cmd,
        path,
        data: sceneTree.metadata.camera.toJSON(),
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  // keep sending the camera often
  // rerun this when the websocket changes
  useEffect(() => {
    const fps = 24;
    const interval = 1000 / fps;
    const refreshIntervalId = setInterval(sendCamera, interval);
    return () => {
      clearInterval(refreshIntervalId);
    };
  }, [websocket]);

  const render = () => {
    requestAnimationFrame(render);
    handleResize();
    sceneTree.metadata.camera.fov = field_of_view_ref.current;
    sceneTree.metadata.camera.updateProjectionMatrix();
    sceneTree.metadata.camera_controls.update();
    renderer.render(scene, sceneTree.metadata.camera);
    stats.update();
    // console.log(sceneTree.metadata.camera);
  };

  // start the three.js rendering loop
  // when the DOM is ready
  useEffect(() => {
    stats = createStats();
    myRef.current.append(stats.domElement);
    myRef.current.append(renderer.domElement);
    render();
  }, []);

  // updates the field of view inside the ref to avoid rerendering so often
  useEffect(() => {
    field_of_view_ref.current = field_of_view;
  }, [field_of_view]);

  return (
    <>
      {/* the webrtc viewer needs to know the camera pose */}
      <WebRtcWindow />
      <div className="canvas-container-main" ref={myRef} />
      <div className="ViewerWindow-camera-dropdown">
        <CameraDropdown />
      </div>
      <div className="ViewerWindow-buttons">
        <TransformIcons sceneTree={sceneTree} />
      </div>
    </>
  );
}
