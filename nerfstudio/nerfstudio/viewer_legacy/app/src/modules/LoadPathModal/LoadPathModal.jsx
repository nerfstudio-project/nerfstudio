/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import { Box, Button, FormControl, Modal, Typography } from '@mui/material';
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

  const [existingPathSelect, setExistingPathSelect] = React.useState('');

  // redux store state
  const all_camera_paths = useSelector((state) => state.all_camera_paths);

  let camera_paths_arr = [];
  if (typeof all_camera_paths === 'object' && all_camera_paths !== null) {
    camera_paths_arr = Object.keys(all_camera_paths).map((key) => {
      return {
        name: key,
        val: all_camera_paths[key],
      };
    });
  }

  const hiddenFileInput = React.useRef(null);
  const handleFileUploadClick = () => {
    hiddenFileInput.current.click();
  };

  // react state

  const handleClose = () => setOpen(false);
  const handlePathSelect = (event) => {
    setExistingPathSelect(event.target.value);
  };

  const handleExistingLoadClick = () => {
    loadCameraPath(existingPathSelect);
    handleClose();
  };

  const handleFileInput = (event) => {
    uploadCameraPath(event);
    handleClose();
  };

  return (
    <div className="LoadPathModal">
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box className="LoadPathModal-box">
          <Typography
            id="modal-modal-description"
            component="div"
            sx={{ mt: 2 }}
          >
            <div>
              <h2>Load Camera Path</h2>
              {camera_paths_arr.length > 0 && (
                <>
                  <p>
                    Either upload a local file or select a saved camera path
                  </p>
                  <FormControl
                    sx={{ minWidth: '100%' }}
                    variant="filled"
                    size="small"
                  >
                    <InputLabel id="ageInputLabel">Existing Path</InputLabel>
                    <Select
                      labelId="ageInputLabel"
                      label="Camera Path"
                      value={existingPathSelect}
                      onChange={handlePathSelect}
                    >
                      {camera_paths_arr.map((obj) => {
                        return <MenuItem value={obj.val}>{obj.name}</MenuItem>;
                      })}
                    </Select>
                    <Button
                      sx={{
                        marginTop: '10px',
                        marginLeft: 'auto',
                        marginRight: 'auto',
                        width: '60%',
                      }}
                      variant="outlined"
                      size="medium"
                      disabled={existingPathSelect === ''}
                      onClick={handleExistingLoadClick}
                    >
                      Load
                    </Button>
                  </FormControl>
                  <br />
                  <center>
                    <p>OR</p>
                  </center>
                </>
              )}
              {camera_paths_arr.length === 0 && (
                <p>No existing saved paths found</p>
              )}
              <div className="LoadPathModal-upload_button">
                <Button
                  sx={{ width: '60%' }}
                  variant="outlined"
                  size="medium"
                  startIcon={<FileUpload />}
                  onClick={handleFileUploadClick}
                >
                  Upload Camera Path
                  <input
                    type="file"
                    accept=".json"
                    name="Camera Path"
                    onChange={handleFileInput}
                    hidden
                    ref={hiddenFileInput}
                  />
                </Button>
              </div>
            </div>
          </Typography>
        </Box>
      </Modal>
    </div>
  );
}
