import {
  ConfigPanel,
  RenderControls,
} from './components/ConfigPanel/ConfigPanel';

import React from 'react';
import { ViewerState } from './components/Viewer/Viewer';

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
