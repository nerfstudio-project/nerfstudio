import { FaBeer, FaGem, FaHeart } from 'react-icons/fa';
import { SceneTreeWebSocketListener, get_scene_tree } from './modules/Scene/Scene';

import { BasicTabs } from './modules/Sidebar/Sidebar';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import LandingModal from './modules/LandingModal';
import { Leva } from 'leva';
import LevaTheme from './modules/ConfigPanel/leva_theme.json';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import React from 'react';
import { RenderControls } from './modules/ConfigPanel/ConfigPanel';
import { Resizable } from 're-resizable';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';

function Banner() {
  // const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [anchorEl, setAnchorEl] = React.useState(0);

  const open = Boolean(anchorEl);
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <div className="banner">
      <LandingModal />
      <Button
        id="basic-button"
        className="banner-button"
        variant="outlined"
        size="small"
        aria-controls={open ? 'basic-menu' : undefined}
        aria-haspopup="true"
        aria-expanded={open ? 'true' : undefined}
        onClick={handleClick}
      >
        Options
      </Button>
      <Menu
        id="basic-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        MenuListProps={{
          'aria-labelledby': 'basic-button',
        }}
      >
        <MenuItem onClick={handleClose}>Load Training Config</MenuItem>
        <MenuItem onClick={handleClose}>Export Training Config</MenuItem>
        <MenuItem onClick={handleClose}>Dark Mode</MenuItem>
      </Menu>
      <Button className="banner-button" variant="outlined" size="small">
        Download Desktop App
      </Button>
    </div>
  );
}

export default function App() {

  // The scene tree won't rerender but it will listener to changes
  // from the redux store and draw three.js objects.
  const sceneTree = get_scene_tree();

  return (
    <div className="App">
      {/*
      Code that listens for websocket 'write' messages and updates the redux store.
      */}
      <SceneTreeWebSocketListener />
      {/* ----- */}
      <Banner />
      <div className="App-body">
        <div className="MySideBar">
          <BasicTabs sceneTree={sceneTree} />
        </div>
        <RenderControls />
        <ViewerWindow scene={sceneTree.object} />
      </div>
    </div>
  );
}
