/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import { Box, Button, Modal } from '@mui/material';
import KeyboardIcon from '@mui/icons-material/Keyboard';

export default function ControlsModal() {
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  return (
    <div className="LandingModal">
      <Button
        className="banner-button"
        variant="outlined"
        size="small"
        startIcon={<KeyboardIcon />}
        onClick={handleOpen}
      >
        Viewport Controls
      </Button>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box className="ViewportConstrolsModal-box">
          <center>
            <img
              style={{ height: 37, margin: 'auto' }}
              src="https://docs.nerf.studio/en/latest/_images/logo-dark.png"
              alt="The favicon."
            />
            <img
              style={{ width: '100%', paddingTop: '30px', margin: 'auto' }}
              src="https://assets.nerf.studio/keyboard_controls.png"
              alt="Controls diagram"
            />
          </center>
        </Box>
      </Modal>
    </div>
  );
}
