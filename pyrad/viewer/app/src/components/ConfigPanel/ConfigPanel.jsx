import { Leva, button, buttonGroup, useControls } from 'leva';
import { useDispatch, useSelector } from 'react-redux';
import React, { useState, useEffect } from 'react';

export function ConfigPanel() {
  const params = { titleBar: false };
  const panel = (
    <div style={{ position: 'relative', width: 250, top: 60 }}>
      <Leva fill oneLineLabels {...params} />
    </div>
  );
  return panel;
}

export function RenderControls() {
  const renderingState = useSelector((state) => state.renderingState);
  const [isTraining, setIsTraining] = React.useState(renderingState.isTraining);
  const [outputOptions, setOutputOptions] = React.useState(
    renderingState.output_options,
  );

  const toggleIsTraining = () => {
    setIsTraining((current) => !current);
  };

  const [controls, setControls] = useControls(
    'Render Options',
    () => ({
      // isTraining
      'Resume Training': button(toggleIsTraining, {
        disabled: !isTraining,
      }),
      'Pause Training': button(toggleIsTraining, {
        disabled: isTraining,
      }),
      // output_options
      output_options: {
        label: 'Output Render',
        options: outputOptions,
      },
      // resolution
      min_resolution: {
        label: 'Min Res.',
        value: 50,
        min: 10,
        max: 100,
        step: 1,
        suffix: 'px',
      },
      ' ': buttonGroup({
        '25px': () => setControls({ min_resolution: 25 }),
        '50px': () => setControls({ min_resolution: 50 }),
        '75px': () => setControls({ min_resolution: 75 }),
        '100px': () => setControls({ min_resolution: 100 }),
      }),
      max_resolution: {
        label: 'Max Res.',
        min: 10,
        max: 2048,
        step: 1,
        value: 512,
        suffix: 'px',
      },
      '  ': buttonGroup({
        '128px': () => setControls({ max_resolution: 128 }),
        '256px': () => setControls({ max_resolution: 256 }),
        '512px': () => setControls({ max_resolution: 512 }),
        '1024px': () => setControls({ max_resolution: 1024 }),
      }),
    }),
    [outputOptions, isTraining],
  );

  // Similar to componentDidMount and componentDidUpdate:
  useEffect(() => {
    setOutputOptions(['hello', 'world']);
    // setControls({ output_options: 1 });
  }, []);

  // setOutputOptions(arr => ["hello", "world"]);
  // setOutputOptions( arr => [...arr, `${arr.length}`]);
  return [controls, setControls];
}
