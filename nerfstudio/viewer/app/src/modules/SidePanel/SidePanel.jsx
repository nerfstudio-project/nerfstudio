/* eslint-disable react/jsx-props-no-spreading */

import * as React from 'react';
import * as THREE from 'three';

import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import {
  CameraAltRounded,
  TuneRounded,
  WidgetsRounded,
  ImportExportRounded,
} from '@mui/icons-material/';
import { useDispatch, useSelector } from 'react-redux';
import StatusPanel from './StatusPanel';
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

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

interface PanelTabContentsProps {
  children: React.ReactNode;
  index: number;
  value: number;
}

/** One tab in the control panel. */
function PanelTabContents(props: PanelTabContentsProps) {
  const { children, value, index, ...other } = props;

  return (
    <Box
      component="div"
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {children}
    </Box>
  );
}

interface PanelContentsProps {
  children: React.ReactNode;
}

function PanelContents(props: PanelContentsProps) {
  const dispatch = useDispatch();
  const [tabState, setTabState] = React.useState(0);
  const handleChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabState(newValue);
    dispatch({
      type: 'write',
      path: 'show_export_box',
      data: newValue === 3,
    });
  };
  const camera_choice = useSelector(
    (state) => state.renderingState.camera_choice,
  );
  const arrayChildren = React.Children.toArray(props.children);

  React.useEffect(() => {
    if (camera_choice === 'Render Camera') {
      setTabState(1);
    }
  }, [camera_choice]);

  return (
    <>
      <Box component="div" sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={tabState}
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

      {arrayChildren.map((child, index) => (
        // eslint-disable-next-line react/no-array-index-key
        <PanelTabContents value={tabState} index={index} key={index}>
          {child}
        </PanelTabContents>
      ))}
    </>
  );
}

interface BasicTabsProps {
  sceneTree: object;
}

export function BasicTabs(props: BasicTabsProps) {
  const sceneTree = props.sceneTree;

  return (
    <div>
      <StatusPanel sceneTree={sceneTree} />
      <Divider />
      <Box sx={{ width: '100%' }}>
        <PanelContents>
          <RenderControls />
          <CameraPanel sceneTree={sceneTree} />
          <ScenePanel sceneTree={sceneTree} />
          <ExportPanel sceneTree={sceneTree} />
        </PanelContents>
      </Box>
    </div>
  );
}
