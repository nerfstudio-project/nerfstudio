/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import { Box, Button, Modal, Typography } from '@mui/material';
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

  const export_path = useSelector(
    (state) => state.renderingState.export_path,
  );

  const data_base_dir = useSelector(
    (state) => state.renderingState.data_base_dir,
  );

  const crop_center = useSelector(
      (state) => state.renderingState.crop_center,
  );
  const crop_scale = useSelector(
      (state) => state.renderingState.crop_scale,
  );

  const crop_enabled = useSelector(
      (state) => state.renderingState.crop_enabled,
  );


  // react state

  const handleClose = () => setOpen(false);

  // Copy the text inside the text field
  const config_filename = `${config_base_dir}/config.yml`;
  const camera_path_filename = `${export_path}.json`;

  let cmd = `ns-render --load-config ${config_filename} --traj filename --camera-path-filename ${data_base_dir}/camera_paths/${camera_path_filename} --output-path renders/${data_base_dir}/${export_path}.mp4`;
  if (crop_enabled)
  {
    const bbox_min = [
      crop_center[0] - crop_scale[0] / 2,
      crop_center[1] - crop_scale[1] / 2,
      crop_center[2] - crop_scale[2] / 2,
    ];
    const bbox_max = [
      crop_center[0] + crop_scale[0] / 2,
      crop_center[1] + crop_scale[1] / 2,
      crop_center[2] + crop_scale[2] / 2,
    ];

    const bounding_box_cmd = ' --use-bounding box ' +
        ` --bounding-box-min ${bbox_min[0]} ${bbox_min[1]} ${bbox_min[2]}` +
        ` --bounding-box-max ${bbox_max[0]} ${bbox_max[1]} ${bbox_max[2]}`;

    cmd = `${cmd} ${bounding_box_cmd}`;
  }
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
                  ./renders/{data_base_dir}/{export_path}.mp4
                </code>
                .
              </p>
              
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
