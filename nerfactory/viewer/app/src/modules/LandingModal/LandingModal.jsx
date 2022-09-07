import * as React from 'react';

import { Box, Button, Modal, Typography } from '@mui/material';

import WebSocketUrlField from '../WebSocketUrlField';

interface LandingModalProps {
  initial_state: object;
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
          <center>
            <img
              style={{ height: 37, margin: 'auto' }}
              src="https://plenoptix-nerfactory.readthedocs-hosted.com/en/latest/_images/logo-dark.png"
              alt="The favicon."
            />
          </center>
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
            </div>
          </Typography>
        </Box>
      </Modal>
    </div>
  );
}
