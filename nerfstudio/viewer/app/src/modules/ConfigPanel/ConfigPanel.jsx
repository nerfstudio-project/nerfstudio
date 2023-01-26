import { buttonGroup, useControls } from 'leva';
import { useContext, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { WebSocketContext } from '../WebSocket/WebSocket';

const msgpack = require('msgpack-lite');

function dispatch_and_send(websocket, dispatch, path, data) {
  dispatch({
    type: 'write',
    path,
    data,
  });
  if (websocket.readyState === WebSocket.OPEN) {
    const message = msgpack.encode({
      type: 'write',
      path,
      data,
    });
    websocket.send(message);
  }
}

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
  const render_time = useSelector((state) => state.renderingState.renderTime);
  const crop_enabled = useSelector(
    (state) => state.renderingState.crop_enabled,
  );
  console.log('crop_enabled_config panel', crop_enabled);

  const crop_bg_color = useSelector(
    (state) => state.renderingState.crop_bg_color,
  );

  const crop_scale = useSelector((state) => state.renderingState.crop_scale);

  const crop_center = useSelector((state) => state.renderingState.crop_center);

  const dispatch = useDispatch();

  const [display_render_time, set_display_render_time] = useState(false);

  const receive_temporal_dist = (e) => {
    const msg = msgpack.decode(new Uint8Array(e.data));
    if (msg.path === '/model/has_temporal_distortion') {
      set_display_render_time(msg.data === 'true');
      websocket.removeEventListener('message', receive_temporal_dist);
    }
  };
  websocket.addEventListener('message', receive_temporal_dist);

  const [, setControls] = useControls(
    () => ({
      // training speed
      SpeedButtonGroup: buttonGroup({
        label: `Train Speed`,
        hint: 'Select the training speed, affects viewer render quality, not final render quality',
        opts: {
          Fast: () =>
            setControls({ target_train_util: 0.9, max_resolution: 512 }),
          Balanced: () =>
            setControls({ target_train_util: 0.7, max_resolution: 1024 }),
          Slow: () =>
            setControls({ target_train_util: 0.1, max_resolution: 2048 }),
        },
      }),
      // output_options
      output_options: {
        label: 'Output Render',
        options: [...new Set(outputOptions)],
        value: outputChoice,
        hint: 'Select the output to render',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/output_choice',
            v,
          );
        },
      },
      // colormap_options
      colormap_options: {
        label: 'Colormap',
        options: colormapOptions,
        value: colormapChoice,
        hint: 'Select the colormap to use',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_choice',
            v,
          );
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
        hint: "Target training utilization, 0.0 is slow, 1.0 is fast, doesn't affect final render quality",
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/targetTrainUtil',
            v,
          );
        },
      },
      // resolution
      max_resolution: {
        label: 'Max Res.',
        value: max_resolution,
        min: 256,
        max: 2048,
        step: 1,
        hint: 'Maximum resolution to render in viewport',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/maxResolution',
            v,
          );
        },
      },
      '  ': buttonGroup({
        '256px': () => setControls({ max_resolution: 256 }),
        '512px': () => setControls({ max_resolution: 512 }),
        '1024px': () => setControls({ max_resolution: 1024 }),
        '2048px': () => setControls({ max_resolution: 2048 }),
      }),
      // Enable Crop
      crop_enabled: {
        label: 'Crop Viewport',
        value: crop_enabled,
        hint: 'Crop the viewport to the selected box',
        onChange: (value) => {
          console.log('dispact value', value);
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_enabled',
            value,
          );
        },
      },
      crop_bg_color: {
        label: '| Background Color',
        value: crop_bg_color,
        render: (get) => get('crop_enabled'),
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_bg_color',
            v,
          );
        },
      },
      crop_scale: {
        label: '|  Scale',
        value: crop_scale,
        min: 0,
        max: 10,
        step: 0.05,
        render: (get) => get('crop_enabled'),
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_scale',
            v,
          );
        },
      },
      crop_center: {
        label: '|  Center',
        value: crop_center,
        min: -10,
        max: 10,
        step: 0.05,
        render: (get) => get('crop_enabled'),
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/crop_center',
            v,
          );
        },
      },
      // Dynamic NeRF rendering time
      ...(display_render_time
        ? {
            render_time: {
              label: 'Render Timestep',
              value: render_time,
              min: 0,
              max: 1,
              step: 0.01,
              onChange: (v) => {
                dispatch_and_send(
                  websocket,
                  dispatch,
                  'renderingState/render_time',
                  v,
                );
              },
            },
          }
        : {}),
    }),
    [
      outputOptions,
      outputChoice,
      colormapOptions,
      colormapChoice,
      max_resolution,
      crop_enabled,
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
    setControls({ crop_enabled });
    setControls({ crop_bg_color });
    setControls({ crop_scale });
    setControls({ crop_center });
  }, [
    setControls,
    outputOptions,
    outputChoice,
    colormapOptions,
    colormapChoice,
    max_resolution,
    target_train_util,
    render_time,
    crop_enabled,
    crop_bg_color,
    crop_scale,
    crop_center,
    display_render_time,
  ]);

  return null;
}
