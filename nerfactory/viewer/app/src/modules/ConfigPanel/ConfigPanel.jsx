import React, { useContext } from 'react';
import { ReactReduxContext, useSelector } from 'react-redux';
import { button, buttonGroup, useControls } from 'leva';

import { useDispatch } from 'react-redux';

export function RenderControls() {
  // connection status indicators
  const isWebsocketConnected = useSelector(
    (state) => state.websocketState.isConnected,
  );
  const isWebrtcConnected = useSelector(
    (state) => state.webrtcState.isConnected,
  );

  // const renderingState = useSelector((state) => state.renderingState);
  // const [isTraining, setIsTraining] = React.useState(renderingState.isTraining);
  const isTraining = useSelector((state) => state.renderingState.isTraining);

  // const [outputOptions, setOutputOptions] = React.useState(
  //   renderingState.output_options,
  // );
  const outputOptions = useSelector(
    (state) => state.renderingState.output_options,
  );
  const outputChoice = useSelector(
    (state) => state.renderingState.output_choice,
  );

  const field_of_view = useSelector(
    (state) => state.renderingState.field_of_view,
  );

  const dispatch = useDispatch();

  const toggleIsTraining = () => {
    console.log('toggle is training');
    // setIsTraining((current) => !current);
    dispatch({
      type: 'write',
      path: 'renderingState/isTraining',
      data: !isTraining,
    });
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
      'Pause Training': button(
        () => {
          console.log('toggle is training resume');
          console.log(isTraining);
          console.log(!isTraining);
          console.log(false);
          // setIsTraining((current) => !current);
          dispatch({
            type: 'write',
            path: 'renderingState/isTraining',
            data: false,
          });
        },
        {
          disabled: !isTraining,
        },
      ),
      'Resume Training': button(
        () => {
          console.log('toggle is training pause');
          console.log(isTraining);
          console.log(!isTraining);
          console.log(true);
          // setIsTraining((current) => !current);
          dispatch({
            type: 'write',
            path: 'renderingState/isTraining',
            data: true,
          });
        },
        {
          disabled: isTraining,
        },
      ),
      // output_options
      output_options: {
        label: 'Output Render',
        options: outputOptions,
        value: outputChoice,
      },
      // resolution
      min_resolution: {
        label: 'Min Res.',
        value: 1000,
        min: 10,
        max: 1000,
        step: 1,
        suffix: 'px',
      },
      ' ': buttonGroup({
        '25px': () => setControls({ min_resolution: 25 }),
        '50px': () => setControls({ min_resolution: 50 }),
        '75px': () => setControls({ min_resolution: 75 }),
        '100px': () => setControls({ min_resolution: 1000 }),
      }),
      max_resolution: {
        label: 'Max Res.',
        min: 1024,
        max: 2048,
        step: 1,
        value: 2048,
        suffix: 'px',
      },
      '  ': buttonGroup({
        '128px': () => setControls({ max_resolution: 128 }),
        '256px': () => setControls({ max_resolution: 256 }),
        '512px': () => setControls({ max_resolution: 512 }),
        '1024px': () => setControls({ max_resolution: 1024 }),
      }),
      'Camera FoV': field_of_view,
    }),
    [
      isWebsocketConnected,
      isWebrtcConnected,
      isTraining,
      outputOptions,
      outputChoice,
      field_of_view,
    ],
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
