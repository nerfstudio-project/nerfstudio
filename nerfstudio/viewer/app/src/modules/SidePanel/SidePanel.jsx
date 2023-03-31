/* eslint-disable react/jsx-props-no-spreading */

import * as React from 'react';
import * as THREE from 'three';

import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import { Leva } from 'leva';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import {
  CameraAltRounded,
  TuneRounded,
  WidgetsRounded,
  ImportExportRounded,
} from '@mui/icons-material/';
import { useSelector } from 'react-redux';
import StatusPanel from './StatusPanel';
import LevaTheme from '../../themes/leva_theme.json';
import CameraPanel from './CameraPanel';
import ScenePanel from './ScenePanel';
import { RenderControls } from '../ConfigPanel/ConfigPanel';
import ExportPanel from './ExportPanel';

export const snap_to_camera = (sceneTree, camera, matrix) => {
  const mat = new THREE.Matrix4();
  mat.fromArray(matrix.elements);
  mat.decompose(camera.position, camera.quaternion, camera.scale);
  const unit = new THREE.Vector3(0, 0, -1);
  const viewDirection = unit.applyMatrix4(mat);
  sceneTree.metadata.camera_controls.setLookAt(
    camera.position.x,
    camera.position.y,
    camera.position.z,
    viewDirection.x,
    viewDirection.y,
    viewDirection.z,
  );
};

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

interface BasicTabsProps {
  sceneTree: object;
}

export function BasicTabs(props: BasicTabsProps) {
  const sceneTree = props.sceneTree;

  const [value, setValue] = React.useState(0);
  const [showExportBox, setShowExportBox] = React.useState(false);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
    setShowExportBox(newValue === 3);
  };
  const camera_choice = useSelector(
    (state) => state.renderingState.camera_choice,
  );

  React.useEffect(() => {
    if (camera_choice === 'Render Camera') {
      setValue(1);
    }
  }, [camera_choice]);

  return (
    <div>
      <StatusPanel sceneTree={sceneTree} />
      <Divider />
      <Box sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={value}
            onChange={handleChange}
            aria-label="panel tabs"
            centered
          >
            <Tab icon={<TuneRounded />} label="Controls" {...a11yProps(0)} />
            <Tab icon={<CameraAltRounded />} label="Render" {...a11yProps(1)} />
            <Tab
              icon={<WidgetsRounded />}
              label="Scene"
              disabled={camera_choice === 'Render Camera'}
              {...a11yProps(2)}
            />
            <Tab
              icon={<ImportExportRounded />}
              label="Export"
              disabled={camera_choice === 'Render Camera'}
              {...a11yProps(3)}
            />
          </Tabs>
        </Box>
        <TabPanel value={value} index={0}>
          <div className="Leva-container">
            <RenderControls />
            <Leva
              className="Leva-panel"
              theme={LevaTheme}
              titleBar={false}
              fill
              flat
            />
          </div>
        </TabPanel>
        <TabPanel value={value} index={1}>
          <CameraPanel
            sceneTree={sceneTree}
            // camera_controls={sceneTree.metadata.camera_controls}
          />
        </TabPanel>
        <TabPanel value={value} index={2}>
          <div className="Scene-container">
            <ScenePanel sceneTree={sceneTree} />
          </div>
        </TabPanel>

        <TabPanel value={value} index={3}>
          <ExportPanel sceneTree={sceneTree} showExportBox={showExportBox} />
        </TabPanel>
      </Box>
    </div>
  );
}
