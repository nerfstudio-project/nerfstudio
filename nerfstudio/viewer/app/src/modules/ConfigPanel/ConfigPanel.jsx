import { buttonGroup, useControls } from 'leva';
import { useContext, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { WebSocketContext } from '../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

export function RenderControls() {
  // connection status indicators
  const websocket = useContext(WebSocketContext).socket;
  const outputOptions = useSelector(
    (state) => state.renderingState.output_options,
  );
  const outputChoice = useSelector(
    (state) => state.renderingState.output_choice,
  );
  const colormapOptions = useSelector(
    (state) => state.renderingState.colormap_options,
  );
  const colormapChoice = useSelector(
    (state) => state.renderingState.colormap_choice,
  );
  const max_resolution = useSelector(
    (state) => state.renderingState.maxResolution,
  );
  const target_train_util = useSelector(
    (state) => state.renderingState.targetTrainUtil,
  );
  const render_time = useSelector(
    (state) => state.renderingState.renderTime,
  );

  const dispatch = useDispatch();

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

  const set_target_train_util = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      dispatch({
        type: 'write',
        path: 'renderingState/targetTrainUtil',
        data: value,
      });
      const cmd = 'write';
      const path = 'renderingState/targetTrainUtil';
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

  const set_colormap_choice = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      dispatch({
        type: 'write',
        path: 'renderingState/colormap_choice',
        data: value,
      });
      const cmd = 'write';
      const path = 'renderingState/colormap_choice';
      const data = {
        type: cmd,
        path,
        data: value,
      };
      const message = msgpack.encode(data);
      websocket.send(message);
    }
  };

  const [display_render_time, set_display_render_time] = useState(false);

  const receive_temporal_dist = e => {
      const msg = msgpack.decode(new Uint8Array(e.data));
      if (msg.path === "/model/has_temporal_distortion") {
        set_display_render_time(msg.data === "true");
        websocket.removeEventListener("message", receive_temporal_dist);
      }
  }
  websocket.addEventListener('message', receive_temporal_dist);

  const set_render_time = (value) => {
    if (websocket.readyState === WebSocket.OPEN) {
      const path = 'renderingState/render_time';
      dispatch({
        type: 'write',
        path,
        data: value,
      });
      const cmd = 'write';
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
    () => ({
      // training speed
      'Train Speed': buttonGroup({
        Fast: () =>
          setControls({ target_train_util: 0.9, max_resolution: 512 }),
        Balanced: () =>
          setControls({ target_train_util: 0.7, max_resolution: 1024 }),
        Slow: () =>
          setControls({ target_train_util: 0.1, max_resolution: 2048 }),
      }),
      // output_options
      output_options: {
        label: 'Output Render',
        options: [...new Set(outputOptions)],
        value: outputChoice,
        onChange: (v) => {
          set_output_choice(v);
        },
      },
      // colormap_options
      colormap_options: {
        label: 'Colormap',
        options: colormapOptions,
        value: colormapChoice,
        onChange: (v) => {
          set_colormap_choice(v);
        },
        disabled: colormapOptions.length === 1,
      },
      // Dynamic Resolution
      target_train_util: {
        label: 'Train Util.',
        value: target_train_util,
        min: 0,
        max: 1,
        step: 0.05,
        onChange: (v) => {
          set_target_train_util(v);
        },
      },
      // resolution
      max_resolution: {
        label: 'Max Res.',
        value: max_resolution,
        min: 256,
        max: 2048,
        step: 1,
        onChange: (v) => {
          set_max_resolution(v);
        },
      },
      '  ': buttonGroup({
        '256px': () => setControls({ max_resolution: 256 }),
        '512px': () => setControls({ max_resolution: 512 }),
        '1024px': () => setControls({ max_resolution: 1024 }),
        '2048px': () => setControls({ max_resolution: 2048 }),
      }),
      // Dynamic NeRF rendering time
      ...(display_render_time ? {render_time: {
        label: 'Render Timestep',
        value: render_time,
        min: 0,
        max: 1,
        step: 0.01,
        onChange: (v) => {
          set_render_time(v);
        },
      }} : {}),
    }),
    [
      outputOptions,
      outputChoice,
      colormapOptions,
      colormapChoice,
      max_resolution,
      target_train_util,
      render_time,
      display_render_time,
      websocket, // need to re-render when websocket changes to use the new websocket
    ],
  );

  useEffect(() => {
    setControls({ max_resolution });
    setControls({ output_options: outputChoice });
    setControls({ colormap_options: colormapChoice });
  }, [
    setControls,
    outputOptions,
    outputChoice,
    colormapOptions,
    colormapChoice,
    max_resolution,
    target_train_util,
    render_time,
    display_render_time,
  ]);

  return null;
}
