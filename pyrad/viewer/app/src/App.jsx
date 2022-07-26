import { ConfigPanel, RenderControls } from './modules/ConfigPanel/ConfigPanel';

import Alert from './modules/Alert/Alert';
import React from 'react';
import SetupScene from './modules/Scene/Scene';
import ViewerWindow from './modules/ViewerWindow/ViewerWindow';

export default function App() {
  let [scene] = SetupScene();
  const [controls, setControls] = RenderControls();
  return (
    <div className="App">
      <ConfigPanel />
      <ViewerWindow scene={scene} />
    </div>
  );
}
