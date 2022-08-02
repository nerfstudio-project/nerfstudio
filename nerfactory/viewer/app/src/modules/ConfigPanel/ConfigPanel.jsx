import React, { useContext } from 'react';
import { ReactReduxContext, useSelector } from 'react-redux';
import { button, buttonGroup, useControls } from 'leva';

import { WebSocketContext } from '../WebSocket/WebSocket';
import { subscribe_to_changes } from '../../subscriber';
import { useDispatch } from 'react-redux';

const msgpack = require('msgpack-lite');

export function RenderControls() {
  // connection status indicators
  const websocket = useContext(WebSocketContext).socket;
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

  const min_resolution = useSelector(
    (state) => state.renderingState.minResolution,
  );
  const max_resolution = useSelector(
    (state) => state.renderingState.maxResolution,
  );

  const field_of_view = useSelector(
    (state) => state.renderingState.field_of_view,
  );

  const dispatch = useDispatch();

  const set_min_resolution = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/minResolution',
      data: value,
    });
    const cmd = 'write';
    const path = 'renderingState/minResolution';
    const data = {
      type: cmd,
      path,
      data: value,
    };
    const message = msgpack.encode(data);
    websocket.send(message);
  };

  const set_max_resolution = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/maxResolution',
      data: value,
    });
    const cmd = 'write';
    const path = 'renderingState/maxResolution';
    const data = {
      type: cmd,
      path,
      data: value,
    };
    const message = msgpack.encode(data);
    websocket.send(message);
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
          // write locally to store
          dispatch({
            type: 'write',
            path: 'renderingState/isTraining',
            data: false,
          });
          // write to server
          const cmd = 'write';
          const path = 'renderingState/isTraining';
          const data = {
            type: cmd,
            path,
            data: false,
          };
          const message = msgpack.encode(data);
          websocket.send(message);
        },
        {
          disabled: !isTraining,
        },
      ),
      'Resume Training': button(
        () => {
          // write locally to store
          dispatch({
            type: 'write',
            path: 'renderingState/isTraining',
            data: true,
          });
          // write to server
          const cmd = 'write';
          const path = 'renderingState/isTraining';
          const data = {
            type: cmd,
            path,
            data: true,
          };
          const message = msgpack.encode(data);
          websocket.send(message);
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
        value: min_resolution,
        min: 10,
        max: 1000,
        step: 1,
        suffix: 'px',
        onChange: (v) => {
          // update the store on change...
          console.log(v);
        },
        transient: false,
      },
      ' ': buttonGroup({
        '25px': () => set_min_resolution(25),
        '50px': () => set_min_resolution(50),
        '75px': () => set_min_resolution(75),
        '100px': () => set_min_resolution(1000),
      }),
      max_resolution: {
        label: 'Max Res.',
        min: 10,
        max: 2048,
        step: 1,
        value: max_resolution,
        suffix: 'px',
      },
      '  ': buttonGroup({
        '128px': () => set_max_resolution(128),
        '256px': () => set_max_resolution(256),
        '512px': () => set_max_resolution(512),
        '1024px': () => set_max_resolution(1024),
      }),
      'Camera FoV': field_of_view,
    }),
    [
      isWebsocketConnected,
      isWebrtcConnected,
      isTraining,
      outputOptions,
      outputChoice,
      min_resolution,
      max_resolution,
      field_of_view,
    ],
  );

  return controls;
}
