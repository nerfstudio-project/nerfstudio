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
  name: object;
  object: object;
}

function ListItem(props: ListItemProps) {
  const name = props.name;
  const object = props.object;
  const groupVisible = props.groupVisible;
  const level = props.level;

  const [visible, setVisible] = React.useState(groupVisible);
  object.visible = visible;

  React.useEffect(() => {
    setVisible(groupVisible);
  }, [props.groupVisible]);

  const handleClick = () => {
    setVisible(!visible);
  };

  return (
    // TODO: maybe this isn't the cleanest way to handle color changes
    <ListItemButton
      sx={{
        pl: 2 + level * 2,
        color: visible ? LevaTheme.colors.accent2 : LevaTheme.colors.disabled,
      }}
    >
      <ListItemIcon
        sx={{
          color: visible ? LevaTheme.colors.accent2 : LevaTheme.colors.disabled,
        }}
      >
        <FaTractor />
      </ListItemIcon>
      <ListItemText primary={name} />
      <IconButton aria-label="visibility" onClick={handleClick}>
        {visible ? <Visibility /> : <VisibilityOff />}{' '}
      </IconButton>
    </ListItemButton>
  );
}

interface ClickableListProps {
  sceneTree: object;
}

function ClickableList(props: ClickableListProps) {
  const sceneTree = props.sceneTree;

  const get_menu_items = (
    name: String,
    scene_node: SceneNode,
    level: Number,
    groupVisible: Boolean,
  ) => {
    // TODO: sort the keys by string
    const isTerminal = (object) => {
      return Object.keys(object.children).includes('<object>');
    };

    const num_children = Object.keys(scene_node.children).length;
    if (num_children === 0) {
      return;
    }

    if (isTerminal(scene_node)) {
      return (
        <ListItem
          name={name}
          object={scene_node.object}
          groupVisible={groupVisible}
          level={level}
        />
      );
    }
    const [open, setOpen] = React.useState(true);
    const [visible, setVisible] = React.useState(true);
    const handleClick = () => {
      setOpen(!open);
    };
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
          onClick={handleClick}
          sx={{
            pl: 2 + level * 2,
            bgcolor: open
              ? LevaTheme.colors.elevation3
              : LevaTheme.colors.elevation1,
            color: visible
              ? LevaTheme.colors.accent2
              : LevaTheme.colors.disabled,
          }}
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
          {open ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
        <Collapse in={open} timeout="auto">
          <List>
            {Object.keys(scene_node.children).map((key) =>
              get_menu_items(key, scene_node.children[key], level + 1, visible),
            )}
          </List>
        </Collapse>
      </>
    );
  };

  const menu_items = get_menu_items('Scene', sceneTree, 0, true);

  return <List sx={{ color: LevaTheme.colors.accent2 }}>{menu_items}</List>;
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
