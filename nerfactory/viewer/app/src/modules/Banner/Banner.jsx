import React from 'react';
import { useDispatch } from 'react-redux';

import Button from '@mui/material/Button';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import LandingModal from '../LandingModal';

function getParam(param_name) {
  // https://stackoverflow.com/questions/831030/how-to-get-get-request-parameters-in-javascript
  const params = new RegExp(
    `[?&]${encodeURIComponent(param_name)}=([^&]*)`,
  ).exec(window.location.href);
  if (params === null) {
    return undefined;
  }
  return decodeURIComponent(params[1]);
}

function getWebsocketEndpoint() {
  const endpoint = getParam('websocket_url');
  return endpoint;
}

export default function Banner() {
  // const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [anchorEl, setAnchorEl] = React.useState(0);
  const dispatch = useDispatch();

  const open = Boolean(anchorEl);
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };

  let open_modal = true;
  const websocket_url_from_argument = getWebsocketEndpoint();
  if (websocket_url_from_argument !== undefined) {
    open_modal = false;
    dispatch({
      type: 'write',
      path: 'websocketState/websocket_url',
      data: websocket_url_from_argument,
    });
  }

  return (
    <div className="banner">
      <LandingModal initial_state={open_modal} />
      <Button
        id="basic-button"
        className="banner-button"
        variant="outlined"
        size="small"
        aria-controls={open ? 'basic-menu' : undefined}
        aria-haspopup="true"
        aria-expanded={open ? 'true' : undefined}
        onClick={handleClick}
      >
        Options
      </Button>
      <Menu
        id="basic-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        MenuListProps={{
          'aria-labelledby': 'basic-button',
        }}
      >
        <MenuItem onClick={handleClose}>Load Training Config</MenuItem>
        <MenuItem onClick={handleClose}>Export Training Config</MenuItem>
        <MenuItem onClick={handleClose}>Dark Mode</MenuItem>
      </Menu>
      <Button className="banner-button" variant="outlined" size="small">
        Download Desktop App
      </Button>
    </div>
  );
}
