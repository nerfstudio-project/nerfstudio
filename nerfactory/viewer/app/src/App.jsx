import './App.css';

import React from 'react';
import { Leva } from 'leva';
import { RenderControls } from './modules/ConfigPanel/ConfigPanel';

import LevaTheme from './modules/ConfigPanel/leva_theme.json';
import SetupScene from './modules/Scene/Scene';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';

export default function App() {
  const scene = SetupScene();

  return (
    <div className="App">
      <Leva theme={LevaTheme} titleBar={false} />
      <RenderControls />
      <ViewerWindow scene={scene} />
    </div>
  );
}
