import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import * as serviceWorker from "./serviceWorker";
import { ViewerState } from "./Viewer";
import { PanelConfig, RenderControls } from "./controlPanel";

function App() {
  let [controls, paused, setControls, setOutputOptions] = RenderControls();

  console.log(controls);
  return (
    <div>
      <ViewerState
        {...controls}
        paused={paused}
        setControls={setControls}
        setOutputOptions={setOutputOptions}
      />
      <button id="demo">Button 1</button>
      <PanelConfig />
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
