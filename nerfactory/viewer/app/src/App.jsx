import './App.css';

import {
  ConfigPanel,
  MyComponent,
  RenderControls,
} from './modules/ConfigPanel/ConfigPanel';
import React, { useEffect, useState } from 'react';

import Alert from './modules/Alert/Alert';
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
