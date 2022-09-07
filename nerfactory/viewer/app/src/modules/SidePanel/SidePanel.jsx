/* eslint-disable react/jsx-props-no-spreading */

import * as React from 'react';

import { FaLightbulb, FaTractor } from 'react-icons/fa';
import { Menu, MenuItem, ProSidebar, SubMenu } from 'react-pro-sidebar';

import TuneRoundedIcon from '@mui/icons-material/TuneRounded';
import WidgetsRoundedIcon from '@mui/icons-material/WidgetsRounded';
import CameraAltRoundedIcon from '@mui/icons-material/CameraAltRounded';
import ReceiptLongRoundedIcon from '@mui/icons-material/ReceiptLongRounded';

import Box from '@mui/material/Box';
import { Leva } from 'leva';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import { useEffect } from 'react';
import Divider from '@mui/material/Divider';
import StatusPanel from './StatusPanel';
import SceneNode from '../../SceneNode';
import LevaTheme from '../ConfigPanel/leva_theme.json';
import CameraPanel from './CameraPanel';

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
      {value === index && (
        <Box sx={{ p: 3, padding: 0 }}>
          <Typography component="div">{children}</Typography>
        </Box>
      )}
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

  const [value, setValue] = React.useState(1);
  object.visible = value;

  function handleClick() {
    setValue(!value);
  }

  return (
    <button type="button" onClick={handleClick}>
      {name}
    </button>
  );
}

interface ClickableListProps {
  sceneTree: object;
}

function ClickableList(props: ClickableListProps) {
  const sceneTree = props.sceneTree;

  const get_menu_items = (name: String, scene_node: SceneNode) => {
    // TODO: sort the keys by string
    const num_children = Object.keys(scene_node.children).length;
    if (num_children === 0) {
      return (
        <MenuItem icon={<FaLightbulb />}>
          <ListItem name={name} object={scene_node.object} />
        </MenuItem>
      );
    }
    return (
      <SubMenu title={name} icon={<FaTractor />} defaultOpen>
        {Object.keys(scene_node.children).map((key) =>
          get_menu_items(key, scene_node.children[key]),
        )}
      </SubMenu>
    );
  };

  const menu_items = get_menu_items('Scene', sceneTree);

  return (
    <ProSidebar>
      <Menu iconShape="square">{menu_items}</Menu>
    </ProSidebar>
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

  // set the panel pane index
  useEffect(() => {
    const panel_index = 0;
    setValue(panel_index);
  }, []);

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
            <Tab
              icon={<TuneRoundedIcon />}
              label="Controls"
              {...a11yProps(0)}
            />
            <Tab
              icon={<WidgetsRoundedIcon />}
              label="Scene"
              {...a11yProps(1)}
            />
            <Tab
              icon={<CameraAltRoundedIcon />}
              label="Render"
              {...a11yProps(2)}
            />
            <Tab
              icon={<ReceiptLongRoundedIcon />}
              label="Logs"
              {...a11yProps(3)}
            />
          </Tabs>
        </Box>
        <TabPanel value={value} index={0}>
          <div className="Leva-container">
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
          <CameraPanel />
        </TabPanel>

        <TabPanel value={value} index={3}>
          TODO: add a logging panel
        </TabPanel>
      </Box>
    </div>
  );
}
