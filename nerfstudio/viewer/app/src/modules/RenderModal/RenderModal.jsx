/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import { Box, Button, Modal, TextField, Typography } from '@mui/material';
import { useSelector } from 'react-redux';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';

interface RenderModalProps {
  open: object;
  setOpen: object;
}

export default function RenderModal(props: RenderModalProps) {
  const open = props.open;
  const setOpen = props.setOpen;

  // redux store state
  const config_base_dir = useSelector(
    (state) => state.renderingState.config_base_dir,
  );

  // react state
  const [filename, setFilename] = React.useState('render_output');

  const handleClose = () => setOpen(false);

  // Copy the text inside the text field
  const config_filename = `${config_base_dir}/config.yml`;
  const camera_path_filename = `${config_base_dir}/camera_path.json`;
  const cmd = `ns-render --load-config ${config_filename} --traj filename --camera-path-filename ${camera_path_filename} --output-path renders/${filename}.mp4`;

  const text_intro = `To render a full resolution video, run the following command in a terminal.`;

  const handleCopy = () => {
    navigator.clipboard.writeText(cmd);
  };

  return (
    <div className="RenderModal">
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box className="RenderModal-box">
          <Typography
            id="modal-modal-description"
            component="div"
            sx={{ mt: 2 }}
          >
            <div className="RenderModel-content">
              <h2>Rendering</h2>
              <p>
                {text_intro}
                <br />
                The video will be saved to{' '}
                <code className="RenderModal-inline-code">
                  ./renders/{filename}.mp4
                </code>
                .
              </p>
              <TextField
                label="Output Name"
                inputProps={{
                  inputMode: 'numeric',
                  pattern: '[+-]?([0-9]*[.])?[0-9]+',
                }}
                onChange={(e) => {
                  setFilename(e.target.value);
                }}
                value={filename}
                variant="standard"
              />
              <div className="RenderModal-code">{cmd}</div>
              <div style={{ textAlign: 'center' }}>
                <Button
                  className="RenderModal-button"
                  variant="outlined"
                  size="small"
                  startIcon={<ContentCopyRoundedIcon />}
                  onClick={handleCopy}
                >
                  Copy Command
                </Button>
              </div>
            </div>
          </Typography>
        </Box>
      </Modal>
    </div>
  );
}
