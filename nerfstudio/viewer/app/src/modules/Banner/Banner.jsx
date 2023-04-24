import React from 'react';
import { useDispatch } from 'react-redux';

import Button from '@mui/material/Button';
import GitHubIcon from '@mui/icons-material/GitHub';
import LandingModal from '../LandingModal';
import ViewportControlsModal from '../ViewportControlsModal';

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

export default function Banner() {
  const dispatch = useDispatch();

  // NOTE: This disables the "Getting Started" screen
  let open_modal = false;

  // possibly set the websocket url
  const websocket_url_from_argument = getParam('websocket_url');
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
      <Button // button with view in ar icon
        className="banner-button"
        variant="outlined"
        startIcon={<GitHubIcon />}
        target="_blank"
        href="https://github.com/netdrones/nerfstudio"
        size="small"
      >
        Github
      </Button>
      <ViewportControlsModal />

      <div className="banner-logo">
        <img
          style={{ height: 30, margin: 'auto' }}
          src="https://storage.cloud.google.com/lucas.netdron.es/logo_big.png"
          alt="The favicon."
        />
      </div>
    </div>
  );
}
