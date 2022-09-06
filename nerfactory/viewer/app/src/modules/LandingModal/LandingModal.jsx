import * as React from 'react';

import { Box, Button, Modal, Typography } from '@mui/material';

import WebSocketUrlField from '../WebSocketUrlField';


interface LandingModalProps {
  initial_state: object
}

export default function LandingModel(props: LandingModalProps) {
  const [open, setOpen] = React.useState(props.initial_state);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  const text_1 = `At the top right is the Status Panel where you can connect to a WebSocket and see the train/eval status of your NeRF model. The WebSocket URL takes the form <domain-or-ip_address>:<port>.`;
  const text_2 =
    'If the WebSocket is not connecting, make sure your port is being forwarded properly!';

  return (
    <div className="LandingModal">
      <Button
        className="banner-button"
        variant="outlined"
        size="small"
        onClick={handleOpen}
      >
        Getting Started
      </Button>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box className="LandingModal-box">
          <Typography id="modal-modal-title" component="div">
            Welcome to the Nerfactory Viewer!{' '}
            <img style={{ height: 37, margin: 'auto' }} src="/favicon.png" alt="The favicon."/>
          </Typography>
          <Typography
            id="modal-modal-description"
            component="div"
            sx={{ mt: 2 }}
          >
            <div className="LandingModel-content">
              <b>Getting Started</b>
              <p>{text_1}</p>
              <p>
                You can enter your WebSocket URL in the Status Panel or here:
              </p>
              <WebSocketUrlField />
              <p>{text_2}</p>
              <b>Other Resources:</b>
              <Button
                className="LandingModal-button"
                variant="outlined"
                href="https://github.com/plenoptix/nerfactory"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </Button>
              <Button
                className="LandingModal-button"
                variant="outlined"
                href="https://plenoptix-nerfactory.readthedocs-hosted.com/en/latest/index.html"
                target="_blank"
                rel="noopener noreferrer"
              >
                Documentation
              </Button>
            </div>
          </Typography>
        </Box>
      </Modal>
    </div>
  );
}
