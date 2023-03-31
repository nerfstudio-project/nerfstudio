import * as React from 'react';

import Button from '@mui/material/Button';
import { useDispatch, useSelector } from 'react-redux';
import PauseIcon from '@mui/icons-material/Pause';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import ImageOutlinedIcon from '@mui/icons-material/ImageOutlined';

import { WebSocketContext } from '../../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

interface StatusPanelProps {
  sceneTree: object;
}

export default function StatusPanel(props: StatusPanelProps) {
  const dispatch = useDispatch();
  const websocket = React.useContext(WebSocketContext).socket;
  const isTraining = useSelector((state) => state.renderingState.isTraining);
  const sceneTree = props.sceneTree;

  const isWebsocketConnected = useSelector(
    (state) => state.websocketState.isConnected,
  );
  const eval_res = useSelector((state) => state.renderingState.eval_res);
  const vis_train_ratio = useSelector(
    (state) => state.renderingState.vis_train_ratio,
  );
  const camera_choice = useSelector(
    (state) => state.renderingState.camera_choice,
  );

  // logic for toggling visibility of the entire scene and just the training images
  const [is_scene_visible, set_is_scene_visible] = React.useState(true);
  const [is_images_visible, set_is_images_visible] = React.useState(true);
  const scene_button = is_scene_visible ? 'Hide Scene' : 'Show Scene';
  const cameras_button = is_images_visible ? 'Hide Images' : 'Show Images';
  sceneTree.object.visible =
    is_scene_visible && camera_choice === 'Main Camera';
  if (sceneTree.find_no_create(['Training Cameras']) !== null) {
    sceneTree.find_no_create(['Training Cameras']).object.visible =
      is_images_visible;
  }

  React.useEffect(() => {
    sceneTree.object.traverse((obj) => {
      if (obj.name === 'CAMERA_LABEL') {
        // eslint-disable-next-line no-param-reassign
        obj.visible = is_scene_visible && camera_choice === 'Main Camera';
      }
    });
  }, [camera_choice, is_scene_visible]);

  const handlePlayChange = () => {
    dispatch({
      type: 'write',
      path: 'renderingState/isTraining',
      data: !isTraining,
    });
    // write to server
    const cmd = 'write';
    const path = 'renderingState/isTraining';
    const data = {
      type: cmd,
      path,
      data: !isTraining,
    };
    const message = msgpack.encode(data);
    websocket.send(message);
  };
  const is_training_text = isTraining ? 'Pause Training' : 'Resume Training';
  const training_icon = isTraining ? <PauseIcon /> : <PlayArrowIcon />;

  const websocket_connected_text = isWebsocketConnected
    ? 'Renderer Connected'
    : 'Renderer Disconnected';
  const websocket_connected_color = isWebsocketConnected ? 'success' : 'error';

  return (
    <div className="StatusPanel">
      <div className="StatusPanel-play-button">
        <Button
          className="StatusPanel-play-button"
          variant="contained"
          color="secondary"
          onClick={handlePlayChange}
          disabled={!isWebsocketConnected}
          startIcon={training_icon}
        >
          {is_training_text}
        </Button>
      </div>
      <Button
        className="StatusPanel-hide-scene-button"
        variant="outlined"
        onClick={() => {
          set_is_scene_visible(!is_scene_visible);
        }}
        style={{ textTransform: 'none' }}
        startIcon={<ViewInArIcon />}
        disabled={camera_choice === 'Render Camera'}
      >
        {scene_button}
      </Button>
      <Button
        className="StatusPanel-hide-scene-button"
        variant="outlined"
        onClick={() => {
          set_is_images_visible(!is_images_visible);
        }}
        style={{ textTransform: 'none' }}
        startIcon={<ImageOutlinedIcon />}
        disabled={camera_choice === 'Render Camera'}
      >
        {cameras_button}
      </Button>
      <Button
        className="StatusPanel-hide-scene-button"
        variant="outlined"
        onClick={() => {
          // eslint-disable-next-line no-restricted-globals
          location.reload();
        }}
        style={{ textTransform: 'none' }}
      >
        Refresh Page
      </Button>
      <div className="StatusPanel-metrics">
        <div>
          <b>Resolution:</b> {eval_res}
        </div>
        <div>
          <b>Time Allocation:</b> {vis_train_ratio}
        </div>
      </div>
      <Button
        className="StatusPanel-button"
        color={websocket_connected_color}
        style={{ textTransform: 'none' }}
      >
        {websocket_connected_text}
      </Button>
    </div>
  );
}
