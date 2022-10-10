import * as React from 'react';

import { TextField, Link } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

export default function WebSocketUrlField() {
  const dispatch = useDispatch();
  
  // websocket port
  const websocket_port = useSelector(
    (state) => state.websocketState.websocket_port,
  );
  const websocket_port_onchange = (event) => {
    const value = event.target.value;
    dispatch({
      type: 'write',
      path: 'websocketState/websocket_port',
      data: value,
    });
  };

  // ip address
  const ip_address = useSelector(
    (state) => state.websocketState.ip_address,
  );
  const ip_address_onchange = (event) => {
    const value = event.target.value;
    dispatch({
      type: 'write',
      path: 'websocketState/ip_address',
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
      <TextField
        className="WebSocketUrlField"
        label="IP Address"
        variant="outlined"
        value={ip_address}
        onChange={ip_address_onchange}
        size="small"
      />
      <Link href={`/?websocket_port=${websocket_port}?ip_address=${ip_address}`}>
        viewer.nerf.studio?websocket_port={websocket_port}?ip_address=${ip_address}
      </Link>
    </div>
  );
}
