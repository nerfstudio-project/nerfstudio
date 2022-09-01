import './Sidebar.scss';

import * as React from 'react';

import {
  FaBeer,
  FaCloud,
  FaHeart,
  FaLightbulb,
  FaTractor,
} from 'react-icons/fa';
import { Menu, MenuItem, ProSidebar, SubMenu } from 'react-pro-sidebar';
import { useContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
// import { UIOutliner } from 'three/editor/js/libs/ui.three.js';
// import { Editor } from '../../libs/three.js/editor/js/Editor.js';
import { Leva } from 'leva';
import LevaTheme from '../ConfigPanel/leva_theme.json';
import { Object3D } from 'three';
import { Resizable } from 're-resizable';
import { SidebarScene } from '../../libs/three.js/editor/js/Sidebar.Scene.js';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import { WebSocketContext } from '../WebSocket/WebSocket';
import { object } from 'prop-types';

// import './SidebarCustom.scss';

interface TabPanelProps {
  children?: React.ReactNode;
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
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          <Typography>{children}</Typography>
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

const style = {
  position: 'absolute',
  top: '0px',
  right: '0px',
  display: 'block', // flex
  alignItems: 'center',
  justifyContent: 'center',
  border: 'solid 1px #ddd',
  background: '#f0f0f080', // with alpha transparency set
  zIndex: 999,
};

function Editor() {
  this.scene = null;
}

function ListItem(props) {
  const name = props.name;
  const object = props.object;

  const [value, setValue] = React.useState(1);
  object.visible = value;

  function handleClick() {
    console.log('handle click!');
    console.log(object);
    setValue(!value);
  }

  console.log(object.name);
  console.log(object);

  return <button onClick={handleClick}>{name}</button>;
}

function ClickableList(props) {
  const sceneTree = props.sceneTree;

  const get_menu_items = (name: String, scene_node: SceneNode) => {
    // TODO: sort the keys by string
    const num_children = Object.keys(scene_node.children).length;
    if (num_children === 0) {
      return (
        <MenuItem icon={<FaLightbulb />}>
          <ListItem name={name} object={scene_node.object}></ListItem>
        </MenuItem>
      );
    }
    return (
      <SubMenu title={name} icon={<FaTractor />} defaultOpen={true}>
        {Object.keys(scene_node.children).map((key) =>
          get_menu_items((name = key), scene_node.children[key]),
        )}
      </SubMenu>
    );
  };

  const menu_items = get_menu_items('Scene', sceneTree);
  console.log(menu_items);

  return (
    <ProSidebar>
      <Menu iconShape="square">{menu_items}</Menu>
    </ProSidebar>
  );
}

function StatusPanel() {
  const websocket = useContext(WebSocketContext).socket;
  const isWebsocketConnected = useSelector(
    (state) => state.websocketState.isConnected,
  );
  const isWebrtcConnected = useSelector(
    (state) => state.webrtcState.isConnected,
  );
  let eval_fps = useSelector((state) => state.renderingState.eval_fps);
  let train_eta = useSelector((state) => state.renderingState.train_eta);
  let vis_train_ratio = useSelector(
    (state) => state.renderingState.vis_train_ratio,
  );

  return (
    <div className="StatusPanel">
      <Button
        variant="contained"
        disabled={!isWebsocketConnected}
        style={{ textTransform: 'none' }}
      >
        Websocket Connected
      </Button>
      <br></br>
      <br></br>
      <Button
        variant="contained"
        disabled={!isWebrtcConnected}
        style={{ textTransform: 'none' }}
      >
        WebRTC Connected
      </Button>
      <br></br>
      <br></br>
      <span>Eval FPS: {eval_fps}</span>
      <br></br>
      <span>Train ETA: {train_eta}</span>
      <br></br>
      <span>Time Allocation: {vis_train_ratio}</span>
    </div>
  );
}

export function BasicTabs(props) {
  const sceneTree = props.sceneTree;
  // const scene = sceneTree.object;

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
    <Resizable
      style={style}
      defaultSize={{
        width: 500,
        height: 1000,
      }}
    >
      <StatusPanel></StatusPanel>
      <Box sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={value}
            onChange={handleChange}
            aria-label="basic tabs example"
          >
            <Tab label="Controls" {...a11yProps(0)} />
            <Tab label="Scene" {...a11yProps(1)} />
            <Tab label="Camera Path" {...a11yProps(2)} />
            <Tab label="Logs" {...a11yProps(3)} />
          </Tabs>
        </Box>
        <TabPanel value={value} index={0}>
          <div className="Leva-container">
            <Leva className="Leva-panel" theme={LevaTheme} titleBar={false} />
          </div>
        </TabPanel>
        <TabPanel value={value} index={1}>
          <div className="Scene-container">
            <ClickableList sceneTree={sceneTree}></ClickableList>
          </div>
        </TabPanel>
        <TabPanel value={value} index={2}>
          TODO: create a camera path panel

          list of cameras
        </TabPanel>
        <TabPanel value={value} index={3}>
          TODO: add a logging panel
        </TabPanel>
      </Box>
    </Resizable>
  );
}
