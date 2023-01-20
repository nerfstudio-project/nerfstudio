/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import { Box, Button, Modal, Typography } from '@mui/material';
import { useSelector } from 'react-redux';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import { FileUpload, FileUploadOutlined } from '@mui/icons-material';

interface LoadPathModalProps {
  open: object;
  setOpen: object;
  pathUploadFunction: any;
  loadCameraPathFunction: any;
}

export default function LoadPathModal(props: LoadPathModalProps) {
  const open = props.open;
  const setOpen = props.setOpen;
  const uploadCameraPath = props.pathUploadFunction;
  const loadCameraPath = props.loadCameraPathFunction;

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

  const all_camera_paths = useSelector(
    (state) => state.renderingState.all_camera_paths,
  )

  console.log(all_camera_paths)

  const hiddenFileInput = React.useRef(null);
  const handleFileUploadClick = event => {
    hiddenFileInput.current.click();
  };

  // react state

  const handleClose = () => setOpen(false);

  return (
    <div className="LoadPathModal">
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
                <h2>Load Camera Path</h2>
                <p>Either upload a local file or select a saved camera path</p>
                <Button
                 size='small'
                 startIcon={<FileUpload/>}
                 onClick={handleFileUploadClick}
                >
                Load Local File
                <input
                  type="file"
                  accept=".json"
                  name="Camera Path"
                  onChange={uploadCameraPath}
                  hidden
                  ref={hiddenFileInput}
                />
                </Button>
                
            </div>
          </Typography>
        </Box>
      </Modal>
    </div>
  );
}
