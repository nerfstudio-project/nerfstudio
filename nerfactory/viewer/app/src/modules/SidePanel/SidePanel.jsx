/* eslint-disable react/jsx-props-no-spreading */

import * as React from 'react';

import { FaTractor } from 'react-icons/fa';

import Box from '@mui/material/Box';
import { Collapse, IconButton } from '@mui/material';
import Divider from '@mui/material/Divider';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import { Leva } from 'leva';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import {
  CameraAltRounded,
  ExpandLess,
  ExpandMore,
  ReceiptLongRounded,
  TuneRounded,
  Visibility,
  VisibilityOff,
  WidgetsRounded,
} from '@mui/icons-material/';
import StatusPanel from './StatusPanel';
import SceneNode from '../../SceneNode';
import LevaTheme from '../ConfigPanel/leva_theme.json';
import CameraPanel from './CameraPanel';
import { RenderControls } from '../ConfigPanel/ConfigPanel';
import { LogPanel } from '../LogPanel/LogPanel';
// import { object } from 'prop-types';

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

interface ListItemProps {
  name: String;
  scene_node: SceneNode;
  level: Number;
  groupVisible: Boolean;
}

function MenuItems(props: ListItemProps) {
  const name = props.name;
  const scene_node = props.scene_node;
  const level = props.level;
  const groupVisible = props.groupVisible;

  // TODO: sort the keys by string
  const terminal = Object.keys(scene_node.children).includes('<object>');

  const num_children = Object.keys(scene_node.children).length;
  if (num_children === 0) {
    return null;
  }

  const [open, setOpen] = React.useState(true);
  const toggleOpen = () => {
    setOpen(!open);
  };

  const [visible, setVisible] = React.useState(groupVisible);
  if (terminal) {
    scene_node.object.visible = visible;
  }

  const toggleVisible = (e) => {
    e.stopPropagation();
    setVisible(!visible);
  };

  React.useEffect(() => {
    setVisible(groupVisible);
  }, [groupVisible]);

  return (
    <>
      <ListItemButton
        onClick={terminal ? null : toggleOpen}
        sx={
          terminal
            ? {
                pl: 2 + level * 2,
                color: visible
                  ? LevaTheme.colors.accent2
                  : LevaTheme.colors.disabled,
              }
            : {
                pl: 2 + level * 2,
                bgcolor: open
                  ? LevaTheme.colors.elevation3
                  : LevaTheme.colors.elevation1,
                color: visible
                  ? LevaTheme.colors.accent2
                  : LevaTheme.colors.disabled,
              }
        }
      >
        <ListItemIcon
          sx={{
            color: visible
              ? LevaTheme.colors.accent2
              : LevaTheme.colors.disabled,
          }}
        >
          <FaTractor />
        </ListItemIcon>
        <ListItemText primary={name} />

        <IconButton aria-label="visibility" onClick={toggleVisible}>
          {visible ? <Visibility /> : <VisibilityOff />}
        </IconButton>

        {terminal
          ? null
          : (() => {
              if (open) {
                return <ExpandLess />;
              }
              return <ExpandMore />;
            })()}
      </ListItemButton>

      {terminal ? null : (
        <Collapse in={open} timeout="auto">
          <List>
            {Object.keys(scene_node.children).filter( (key) => {
              if (key === "Camera") {
                return false; // skip
              }
              return true;
            }).map((key) => (
              <MenuItems
                name={key}
                scene_node={scene_node.children[key]}
                level={level + 1}
                groupVisible={visible}
              />
            ))}
          </List>
        </Collapse>
      )}
    </>
  );
}

interface ClickableListProps {
  sceneTree: object;
}

function ClickableList(props: ClickableListProps) {
  const sceneTree = props.sceneTree;

  return (
    <List sx={{ color: LevaTheme.colors.accent2 }}>
      <MenuItems name="Scene" scene_node={sceneTree} level={0} groupVisible />
    </List>
  );
}

interface BasicTabsProps {
  sceneTree: object;
}

export function BasicTabs(props: BasicTabsProps) {
  const sceneTree = props.sceneTree;

  const [value, setValue] = React.useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  return (
    <div>
      <StatusPanel sceneTree={sceneTree} />
      <Divider />
      <Box sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={value}
            onChange={handleChange}
            aria-label="basic tabs example"
          >
            <Tab icon={<TuneRounded />} label="Controls" {...a11yProps(0)} />
            <Tab icon={<WidgetsRounded />} label="Scene" {...a11yProps(1)} />
            <Tab icon={<CameraAltRounded />} label="Render" {...a11yProps(2)} />
            <Tab icon={<ReceiptLongRounded />} label="Logs" {...a11yProps(3)} />
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
          <div className="Scene-container">
            <ClickableList sceneTree={sceneTree} />
          </div>
        </TabPanel>
        <TabPanel value={value} index={2}>
          <CameraPanel
            sceneTree={sceneTree}
            // camera_controls={sceneTree.metadata.camera_controls}
          />
        </TabPanel>

        <TabPanel value={value} index={3}>
          <LogPanel>TODO: Something maybe?</LogPanel>
        </TabPanel>
      </Box>
    </div>
  );
}
