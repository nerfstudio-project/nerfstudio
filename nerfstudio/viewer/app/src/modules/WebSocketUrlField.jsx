import * as React from 'react';

import { TextField } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

export default function WebSocketUrlField() {
  const websocket_port = useSelector(
    (state) => state.websocketState.websocket_port,
  );
  const dispatch = useDispatch();
  const websocket_port_onchange = (event) => {
    const value = event.target.value;
    dispatch({
      type: 'write',
      path: 'websocketState/websocket_port',
      data: value,
    });
  };

  return (
    <TextField
      className="WebSocketUrlField"
      label="WebSocket Port"
      variant="outlined"
      value={websocket_port}
      onChange={websocket_port_onchange}
      size="small"
    />
  );
}
