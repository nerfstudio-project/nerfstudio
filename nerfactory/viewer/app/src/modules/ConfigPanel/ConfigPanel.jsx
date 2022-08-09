import { button, buttonGroup, useControls } from 'leva';
import { useContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { WebSocketContext } from '../WebSocket/WebSocket';

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
  const isTraining = useSelector((state) => state.renderingState.isTraining);
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
  let eval_fps = useSelector(
    (state) => state.renderingState.eval_fps,
  );
  let train_eta = useSelector(
    (state) => state.renderingState.train_eta,
  );

  const dispatch = useDispatch();

  const set_min_resolution = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
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
    }
  };

  const set_max_resolution = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
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
    }
  };

  const set_output_choice = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      dispatch({
        type: 'write',
        path: 'renderingState/output_choice',
        data: value,
      });
      const cmd = 'write';
      const path = 'renderingState/output_choice';
      const data = {
        type: cmd,
        path,
        data: value,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const set_fov = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      dispatch({
        type: 'write',
        path: 'renderingState/field_of_view',
        data: value,
      });
      const cmd = 'write';
      const path = 'renderingState/field_of_view';
      const data = {
        type: cmd,
        path,
        data: value,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const [, setControls] = useControls(
    'Rendering Controls',
    () => ({
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
        onChange: (v) => {
          set_output_choice(v);
        },
      },
      // resolution
      min_resolution: {
        label: 'Min Res.',
        value: min_resolution,
        min: 10,
        max: 100,
        step: 1,
        onChange: (v) => {
          set_min_resolution(v);
        },
      },
      ' ': buttonGroup({
        '25px': () => setControls({ min_resolution: 25 }),
        '50px': () => setControls({ min_resolution: 50 }),
        '75px': () => setControls({ min_resolution: 75 }),
        '100px': () => setControls({ min_resolution: 100 }),
      }),
      max_resolution: {
        label: 'Max Res.',
        value: max_resolution,
        min: 10,
        max: 2048,
        step: 1,
        onChange: (v) => {
          set_max_resolution(v);
        },
      },
      '  ': buttonGroup({
        '128px': () => setControls({ max_resolution: 128 }),
        '256px': () => setControls({ max_resolution: 256 }),
        '512px': () => setControls({ max_resolution: 512 }),
        '1024px': () => setControls({ max_resolution: 1024 }),
      }),
      'Camera FoV': {
        value: field_of_view,
        onChange: (v) => {
          set_fov(v);
        },
      },
    }),
    [
      isTraining,
      outputOptions,
      outputChoice,
      min_resolution,
      max_resolution,
      field_of_view,
    ],
  );

  const [, setState] = useControls(
    'Rendering Monitor',
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
      eval_fps: {
        label: 'Eval FPS',
        value: eval_fps,
        disabled: true,
      },
      train_eta: {
        label: 'Train ETA',
        value: train_eta,
        disabled: true,
      },
    }),
    [
      isWebsocketConnected,
      isWebrtcConnected,
      eval_fps,
      train_eta,
    ],
  );

  useEffect(() => {
    setControls({ min_resolution });
    setControls({ max_resolution });
    setControls({ output_options: outputChoice });
    setControls({ 'Camera FoV': field_of_view });
  }, [
    setControls,
    isTraining,
    outputOptions,
    outputChoice,
    min_resolution,
    max_resolution,
    field_of_view,
  ]);

  useEffect(() => {}, [
    setState,
    isWebsocketConnected,
    isWebrtcConnected,
  ]);

  useEffect(() => {
    websocket.addEventListener('message', (originalCmd) => {
      // set the remote description when the offer is received
      const cmd = msgpack.decode(new Uint8Array(originalCmd.data));
      if (cmd.path === '/renderingState/eval_fps') {
        eval_fps = cmd.data;
        console.log("setting state", eval_fps);
        setState({ eval_fps });
      }
      if (cmd.path === '/renderingState/train_eta') {
        train_eta = cmd.data;
        console.log("setting state", train_eta)
        setState({ train_eta });
      }
    });
  }, []);

  return null;
}
