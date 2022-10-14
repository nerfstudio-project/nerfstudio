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

  return (
    <div>
      <TextField
        className="WebSocketUrlField"
        label="WebSocket URL"
        variant="outlined"
        value={websocket_url}
        onChange={websocket_url_onchange}
        size="small"
      />
      <Link href={`/?websocket_url=${websocket_url}`}>
        viewer.nerf.studio?websocket_url={websocket_url}
      </Link>
    </div>
  );
}
