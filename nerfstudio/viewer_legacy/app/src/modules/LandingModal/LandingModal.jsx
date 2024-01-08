/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import {
  Box,
  Button,
  Modal,
  Typography,
  Tabs,
  Tab,
  Chip,
  Link,
} from '@mui/material';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';

import WebSocketUrlField from '../WebSocketUrlField';

interface LandingModalProps {
  initial_state: object;
}

interface TabPanelProps {
  children: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      // eslint-disable-next-line react/jsx-props-no-spreading
      {...other}
    >
      <Box sx={{ p: 3, padding: 0 }}>
        <Typography component="div">{children}</Typography>
      </Box>
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

export default function LandingModel(props: LandingModalProps) {
  const [open, setOpen] = React.useState(props.initial_state);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  const text_intro = `To use our viewer, you need to connect to the WebSocket server where your training job is running. Enter your WebSocket URL below and click the link to connect to your training job.`;

  const command = `ssh -L <port>:localhost:<port> USER@REMOTE.SERVER.IP`;

  const platform = window.navigator.platform.toLowerCase();
  let initial_tab = 0;
  if (platform.includes('mac')) {
    initial_tab = 0;
  } else if (platform.includes('linux')) {
    initial_tab = 1;
  } else if (platform.includes('win')) {
    initial_tab = 2;
  }

  const [tabValue, setTabValue] = React.useState(initial_tab);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(command);
  };

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
              src="https://docs.nerf.studio/_images/logo-dark.png"
              alt="The favicon."
            />
          </center>
          <Typography
            id="modal-modal-description"
            component="div"
            sx={{ mt: 2 }}
          >
            <div className="LandingModel-content">
              <h2>Getting Started</h2>
              <p>{text_intro}</p>
              <WebSocketUrlField />
              <h3>Remote Server</h3>
              <p>
                If you are using a remote server, make sure to forward the port.
                This can be accomplished by running the following command on
                this machine.
              </p>

              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs
                  value={tabValue}
                  onChange={handleTabChange}
                  aria-label="os tabs"
                >
                  <Tab label="Mac OS" {...a11yProps(0)} />
                  <Tab label="Linux" {...a11yProps(1)} />
                  <Tab label="Windows" {...a11yProps(2)} />
                </Tabs>
              </Box>
              <TabPanel value={tabValue} index={0}>
                <p>
                  SSH must be set up on the remote machine. Then run the
                  following on this machine:
                </p>
                <Chip
                  label={command}
                  onDelete={handleCopy}
                  deleteIcon={<ContentCopyRoundedIcon />}
                />
              </TabPanel>
              <TabPanel value={tabValue} index={1}>
                <p>
                  SSH must be set up on the remote machine. Then run the
                  following on this machine:
                </p>
                <Chip
                  label={command}
                  onDelete={handleCopy}
                  deleteIcon={<ContentCopyRoundedIcon />}
                />
              </TabPanel>
              <TabPanel value={tabValue} index={2}>
                <p>
                  SSH must be set up on the remote machine. Then run the
                  following on this machine in PowerShell:
                </p>
                <Chip
                  label={command}
                  onDelete={handleCopy}
                  deleteIcon={<ContentCopyRoundedIcon />}
                />
                <p>
                  If OpenSSH is not installed, refer to this{' '}
                  <Link
                    href="http://woshub.com/using-native-ssh-client-windows/"
                    color="secondary"
                  >
                    guide
                  </Link>
                  .
                </p>
              </TabPanel>
            </div>
          </Typography>
        </Box>
      </Modal>
    </div>
  );
}
