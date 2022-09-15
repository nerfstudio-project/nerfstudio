import { useContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { WebSocketContext } from '../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

export function LogPanel() {
  const websocket = useContext(WebSocketContext).socket;
  const dispatch = useDispatch();
  const gpu_oom_error_msg = 'GPU out of memory';
  const resolved_msg = 'resolved';
  let local_error = resolved_msg;
  // connection status indicators

  const set_max_train_util = () => {
    if (websocket.readyState === WebSocket.OPEN) {
      dispatch({
        type: 'write',
        path: 'renderingState/targetTrainUtil',
        data: 0.9,
      });
      const cmd = 'write';
      const path = 'renderingState/targetTrainUtil';
      const data = {
        type: cmd,
        path,
        data: 0.9,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const set_small_resolution = () => {
    if (websocket.readyState === WebSocket.OPEN) {
      dispatch({
        type: 'write',
        path: 'renderingState/maxResolution',
        data: 512,
      });
      const cmd = 'write';
      const path = 'renderingState/maxResolution';
      const data = {
        type: cmd,
        path,
        data: 512,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const set_log_message = () => {
    if (websocket.readyState === WebSocket.OPEN) {
      dispatch({
        type: 'write',
        path: 'renderingState/log_errors',
        data: resolved_msg,
      });
      const cmd = 'write';
      const path = 'renderingState/log_errors';
      const data = {
        type: cmd,
        path,
        data: resolved_msg,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const check_error = useSelector((state) => {
    local_error = state.renderingState.log_errors;
    if (local_error.includes(gpu_oom_error_msg)) {
      console.log(local_error);
      set_log_message();
      set_small_resolution();
      set_max_train_util();
    }
  });

  useEffect(() => {}, [check_error, local_error]);

  return null;
}
