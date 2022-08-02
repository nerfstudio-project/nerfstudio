import './App.css';

import React from 'react';
import { RenderControls } from './modules/ConfigPanel/ConfigPanel';

import SetupScene from './modules/Scene/Scene';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';

export default function App() {
  const scene = SetupScene();

  return (
    <div className="App">
      <RenderControls />
      <ViewerWindow scene={scene} />
    </div>
  );
}
