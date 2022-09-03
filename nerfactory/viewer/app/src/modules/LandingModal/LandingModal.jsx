import * as React from 'react';

import { Box, Button, Modal, TextField, Typography } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import WebSocketUrlField from '../WebSocketUrlField';

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400, // 400
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

export default function LandingModel() {
  const [open, setOpen] = React.useState(true);
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
          {/* <Box sx={style}> */}
          <Typography id="modal-modal-title" variant="h6" component="h2">
            Welcome to the Nerfactory Viewer!
          </Typography>
          <Typography id="modal-modal-description" sx={{ mt: 2 }}>
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
