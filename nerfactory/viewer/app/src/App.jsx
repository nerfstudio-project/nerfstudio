import './App.css';

import React from 'react';
import { RenderControls } from './modules/ConfigPanel/ConfigPanel';
import { Leva } from 'leva';

import SetupScene from './modules/Scene/Scene';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';

export default function App() {
  const scene = SetupScene();
  const theme = require('./modules/ConfigPanel/leva_theme.json');

  return (
    <div className="App">
      <Leva theme={theme} titleBar={false} />
      <RenderControls />
      <ViewerWindow scene={scene} />
    </div>
  );
}
