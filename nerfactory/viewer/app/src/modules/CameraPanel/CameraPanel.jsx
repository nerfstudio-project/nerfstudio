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

export default function CameraPanel() {
  return (
    <div className="CameraPanel">
      TODO: create a camera path panel list of cameras
      <Button
        id="CameraPanel-add-camera-button"
        variant="outlined"
      >
        Add Camera
      </Button>
    </div>
  );
}
