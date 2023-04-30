import * as React from 'react';

import { TextField, Link } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

export default function WebSocketUrlField() {
  const dispatch = useDispatch();

  // websocket url
  const websocket_url = useSelector(
    (state) => state.websocketState.websocket_url,
  );
  const websocket_url_onchange = (event) => {
    const value = event.target.value;
    dispatch({
      type: 'write',
      path: 'websocketState/websocket_url',
      data: value,
    });
  };

  const testWebSocket = (url) => {
    try {
      // eslint-disable-next-line no-new
      new WebSocket(url);
      return false;
    } catch (error) {
      return true;
    }
  };

  const currentHost = `${window.location.protocol}//${window.location.host}`;

  return (
    <div>
      <TextField
        className="WebSocketUrlField"
        label="WebSocket URL"
        variant="outlined"
        value={websocket_url}
        onChange={websocket_url_onchange}
        size="small"
        error={testWebSocket(websocket_url)}
        helperText={testWebSocket(websocket_url) ? 'Invalid websocket URL' : ''}
      />
      <Link href={`/?websocket_url=${websocket_url}`}>
        {currentHost}?websocket_url={websocket_url}
      </Link>
    </div>
  );
}
