/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import { Box, Button, Modal, Typography } from '@mui/material';
import InputLabel from '@mui/material/InputLabel';
import { useSelector } from 'react-redux';
import { FileUpload } from '@mui/icons-material';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';

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
  const all_camera_paths = useSelector(
    (state) => state.renderingState.all_camera_paths,
  );

  let camera_paths_arr = []
  if(typeof all_camera_paths === "object" && all_camera_paths !== null){
    camera_paths_arr = Object.keys(all_camera_paths).map((key) => {
      return {
        "name": key, 
        "val": all_camera_paths[key]
      }});
  }

  const hiddenFileInput = React.useRef(null);
  const handleFileUploadClick = () => {
    hiddenFileInput.current.click();
  };

  // react state

  const handleClose = () => setOpen(false);
  const handlePathSelect = event => {
    loadCameraPath(event.target.value);
  };

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
                <InputLabel id="ageInputLabel">Select Camera Path</InputLabel>
                <Select labelId='ageInputLabel' label="Camera Path" value="" onChange={handlePathSelect}>
                  {camera_paths_arr.map((obj) => {
                    return <MenuItem value={obj.val}>{obj.name}</MenuItem>
                  })}
                </Select>
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
