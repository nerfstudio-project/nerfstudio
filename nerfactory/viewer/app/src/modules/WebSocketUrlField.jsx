import * as React from 'react';

import { TextField } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

export default function WebSocketUrlField() {

  const websocket_url = useSelector(
    (state) => state.websocketState.websocket_url,
  );
  const dispatch = useDispatch();
  const websocket_url_onchange = (event) => {
    const value = event.target.value;
    dispatch({
      type: 'write',
      path: 'websocketState/websocket_url',
      data: value,
    });
  };

  return (
    <TextField
      className="WebSocketUrlField"
      label="WebSocket URL"
      variant="outlined"
      value={websocket_url}
      onChange={websocket_url_onchange}
    />
  );
}
