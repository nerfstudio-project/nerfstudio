import * as React from 'react';

import { TextField, Link } from '@mui/material';
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
    <div>
      <TextField
        className="WebSocketUrlField"
        label="WebSocket Port"
        variant="outlined"
        value={websocket_port}
        onChange={websocket_port_onchange}
        size="small"
      />
      <Link href={`/?websocket_port=${websocket_port}`}>
        viewer.nerf.studio?websocket_port={websocket_port}
      </Link>
    </div>
  );
}
