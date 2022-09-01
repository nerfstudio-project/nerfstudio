import './App.css';
import 'react-pro-sidebar/dist/css/styles.css';

import { FaBeer, FaGem, FaHeart } from 'react-icons/fa';
import { BasicTabs } from './modules/Sidebar/Sidebar';
import Box from '@mui/material/Box';
import { Leva } from 'leva';
import LevaTheme from './modules/ConfigPanel/leva_theme.json';
import React from 'react';
import { RenderControls } from './modules/ConfigPanel/ConfigPanel';
import { Resizable } from 're-resizable';
import SetupScene from './modules/Scene/Scene';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Typography from '@mui/material/Typography';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';
import Button from '@mui/material/Button';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';


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
    <div className='banner'>
      <Button
        id="basic-button"
        aria-controls={open ? 'basic-menu' : undefined}
        aria-haspopup="true"
        aria-expanded={open ? 'true' : undefined}
        onClick={handleClick}
      >
        Dashboard
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
        <MenuItem onClick={handleClose}>Profile</MenuItem>
        <MenuItem onClick={handleClose}>My account</MenuItem>
        <MenuItem onClick={handleClose}>Logout</MenuItem>
      </Menu>
    </div>
  );
}

export default function App() {
  const sceneTree = SetupScene();

  return (
    <div className="App">
      <Banner></Banner>
      <BasicTabs sceneTree={sceneTree} />
      <RenderControls />
      <ViewerWindow scene={sceneTree.object} />
    </div>
  );
}
