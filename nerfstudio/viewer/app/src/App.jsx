import { CssBaseline, ThemeProvider } from '@mui/material';
import React from 'react';
import { get_scene_tree } from './modules/Scene/Scene';

import Banner from './modules/Banner';
import { BasicTabs } from './modules/SidePanel/SidePanel';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';
import { appTheme } from './themes/theme';

export default function App() {
  // The scene tree won't rerender but it will listen to changes
  // from the redux store and draw three.js objects.
  // In particular, it listens to changes to 'sceneState' coming over the websocket.
  const sceneTree = get_scene_tree();

  return (
    <ThemeProvider theme={appTheme}>
      <CssBaseline enableColorScheme />
      <div className="App">
        {/* The banner at the top of the page. */}
        <Banner />
        <div className="App-body">
          {/* Order matters here. The viewer window must be rendered first. */}
          <ViewerWindow sceneTree={sceneTree} />
          <div className="SidePanel">
            <BasicTabs sceneTree={sceneTree} />
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}
