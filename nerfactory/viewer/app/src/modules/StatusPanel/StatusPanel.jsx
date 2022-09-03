import * as React from 'react';

import Button from '@mui/material/Button';
import { WebSocketContext } from '../WebSocket/WebSocket';
import WebSocketUrlField from '../WebSocketUrlField';
import { useContext } from 'react';
import { useSelector } from 'react-redux';

export default function StatusPanel() {
  const isWebsocketConnected = useSelector(
    (state) => state.websocketState.isConnected,
  );
  const isWebrtcConnected = useSelector(
    (state) => state.webrtcState.isConnected,
  );
  const eval_fps = useSelector((state) => state.renderingState.eval_fps);
  const train_eta = useSelector((state) => state.renderingState.train_eta);
  const vis_train_ratio = useSelector(
    (state) => state.renderingState.vis_train_ratio,
  );

  return (
    <div className="StatusPanel">
      <div className="StatusPanel-title">Status Panel</div>
      <Button
        className="StatusPanel-button"
        variant="contained"
        disabled={!isWebsocketConnected}
        style={{ textTransform: 'none' }}
      >
        Websocket Connected
      </Button>
      <Button
        className="StatusPanel-button"
        variant="contained"
        disabled={!isWebrtcConnected}
        style={{ textTransform: 'none' }}
      >
        WebRTC Connected
      </Button>
      <WebSocketUrlField />
      <div className="StatusPanel-metrics">
        <div>
          <b>Eval FPS:</b> {eval_fps}
        </div>
        <div>
          <b>Train ETA:</b> {train_eta}
        </div>
        <div>
          <b>Time Allocation:</b> {vis_train_ratio}
        </div>
      </div>
    </div>
  );
}
