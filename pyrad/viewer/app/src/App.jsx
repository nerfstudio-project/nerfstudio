import { ConfigPanel, RenderControls } from './modules/ConfigPanel/ConfigPanel';

import React from 'react';
import ViewerState from './modules/Viewer/Viewer';
import Alert from "./modules/Alert/Alert";

export default function App() {
  const [controls, setControls] = RenderControls();
  console.log(controls);
  return (
    <div className="App">
      <ViewerState
        {...controls}
        // paused={paused}
        setControls={setControls}
        // setOutputOptions={setOutputOptions}
      />
      <button id="demo">Button 1</button>
      <ConfigPanel />
    </div>
  );
}
