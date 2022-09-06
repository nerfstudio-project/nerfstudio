import React from 'react';

import { CssBaseline, ThemeProvider } from '@mui/material';
import { appTheme } from './themes/theme.ts';
import { BasicTabs } from './modules/SidePanel/SidePanel';
import {
  SceneTreeWebSocketListener,
  get_scene_tree,
} from './modules/Scene/Scene';
import { RenderControls } from './modules/ConfigPanel/ConfigPanel';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';
import Banner from './modules/Banner';

export default function App() {
  // The scene tree won't rerender but it will listener to changes
  // from the redux store and draw three.js objects.
  const sceneTree = get_scene_tree();

  return (
    <ThemeProvider theme={appTheme}>
      <CssBaseline enableColorScheme />
      <div className="App">
        {/*
      Code that listens for websocket 'write' messages and updates the redux store.
      */}
        <SceneTreeWebSocketListener />
        {/* ----- */}
        <Banner />
        <div className="App-body">
          <div className="SidePanel">
            <BasicTabs sceneTree={sceneTree} />
          </div>
          <RenderControls />
          <ViewerWindow scene={sceneTree.object} />
        </div>
      </div>
    </ThemeProvider>
  );
}
