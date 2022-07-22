import React from "react";
import { Leva, button, buttonGroup, useControls } from "leva";

export function PanelConfig() {
  let params = { titleBar: false };
  let panel = (
    <div style={{ position: "relative", width: 250, top: 60 }}>
      <Leva fill oneLineLabels {...params} />
    </div>
  );
  return panel;
}

export function RenderControls() {
  const [outputOptions, setOutputOptions] = React.useState([]);
  const [controls, setControls] = useControls(
    "Render Options",
    () => ({
      pause_training: {
        label: "Pause Training?",
        value: false,
      },
      output_options: {
        label: "Output Render",
        options: outputOptions,
      },
      min_resolution: {
        label: "Min Res.",
        value: 50,
        min: 10,
        max: 100,
        step: 1,
        suffix: "px",
      },
      " ": buttonGroup({
        "25px": () => setControls({ min_resolution: 25 }),
        "50px": () => setControls({ min_resolution: 50 }),
        "75px": () => setControls({ min_resolution: 75 }),
        "100px": () => setControls({ min_resolution: 100 }),
      }),
      max_resolution: {
        label: "Max Res.",
        min: 10,
        max: 2048,
        step: 1,
        value: 512,
        suffix: "px",
      },
      "  ": buttonGroup({
        "128px": () => setControls({ max_resolution: 128 }),
        "256px": () => setControls({ max_resolution: 256 }),
        "512px": () => setControls({ max_resolution: 512 }),
        "1024px": () => setControls({ max_resolution: 1024 }),
      }),
    }),
    [outputOptions]
  );

  return [controls, setControls, setOutputOptions];
}
