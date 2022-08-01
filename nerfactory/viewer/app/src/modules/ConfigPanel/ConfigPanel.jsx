import { Leva, button, buttonGroup, useControls } from 'leva';
import React, { useContext, useEffect, useState } from 'react';
import { ReactReduxContext, seDispatch, useSelector } from 'react-redux';

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
  // connection status indicators
  const isWebsocketConnected = useSelector(
    (state) => state.websocketState.isConnected,
  );
  const isWebrtcConnected = useSelector(
    (state) => state.webrtcState.isConnected,
  );

  const renderingState = useSelector((state) => state.renderingState);
  const [isTraining, setIsTraining] = React.useState(renderingState.isTraining);
  // const [outputOptions, setOutputOptions] = React.useState(
  //   renderingState.output_options,
  // );
  const outputOptions = useSelector((state) => state.renderingState.output_options);
  const outputChoice = useSelector((state) => state.renderingState.output_choice);

  const toggleIsTraining = () => {
    setIsTraining((current) => !current);
  };

  const { controls, setControls } = useControls(
    'Rendering State',
    () => ({
      // WebSocket isConnected
      // button does nothing except be an indicator
      'WebSocket Connected': button(() => {}, {
        disabled: !isWebsocketConnected,
      }),
      // webRTC isConnected
      // button does nothing except be an indicator
      'WebRTC Connected': button(() => {}, {
        disabled: !isWebrtcConnected,
      }),
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
        value: outputChoice,
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
    [isWebsocketConnected, isWebrtcConnected, isTraining, outputOptions, outputChoice],
  );

  // some listeners to update the state
  const { store } = useContext(ReactReduxContext);
  const selectOutputChoice = (state) => {
    return state.renderingState.output_choice;
  };
  let outputChoiceCurrent;
  const handleOutputOptions = () => {
    const outputChoicePrevious = outputChoiceCurrent;
    outputChoiceCurrent = selectOutputChoice(store.getState());
    if (outputChoicePrevious !== outputChoiceCurrent) {
      if (outputChoiceCurrent !== null) {
        console.log(outputChoiceCurrent);
      }
    }
  };
  store.subscribe(handleOutputOptions);

  return controls;
}
