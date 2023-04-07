import * as React from 'react';
import {
  LevaPanel,
  useCreateStore,
  Leva,
  buttonGroup,
  useControls,
  folder,
  button,
} from 'leva';
import { useContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import LevaTheme from '../../themes/leva_theme.json';
import { GuiUpdateMessage } from '../WebSocket/ViserMessages';

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
function CustomLeva() {
  const customGui = useSelector((state) => state.custom_gui);
  const guiNames = customGui.guiNames;
  const guiConfigFromName = customGui.guiConfigFromName;
  // Add callbacks to guiConfigFromName.
  // const suppressOnChange = React.useRef<{ [key: string]: boolean }>({});
  const suppressOnChange = React.useRef({});

  // We're going to try and build an object that looks like:
  // {"folder name": {"input name": leva config}}
  const guiConfigTree: { [key: string]: any } = {};
  function getFolderContainer(folderLabels: string[]) {
    let guiConfigNode = guiConfigTree;
    folderLabels.forEach((label) => {
      if (guiConfigNode[label] === undefined) {
        guiConfigNode[label] = { _is_folder_marker: true };
      }
      guiConfigNode = guiConfigNode[label];
    });
    return guiConfigNode;
  }

  guiNames.forEach((key) => {
    const { levaConf, folderLabels } = guiConfigFromName[key];

    const leafFolder = getFolderContainer(folderLabels);

    // Hacky stuff that lives outside of TypeScript...
    if (levaConf['type'] === 'BUTTON') {
      // Add a button.
      leafFolder[key] = button((_get: any) => {
        const message: GuiUpdateMessage = {
          type: 'gui_update',
          name: key,
          value: true,
        };
      }, levaConf['settings']);
    } else {
      // Add any other kind of input.
      leafFolder[key] = {
        ...levaConf,
        onChange: (value: any, _propName: any, options: any) => {
          if (options.initial) return;
          if (suppressOnChange.current[key]) {
            delete suppressOnChange.current[key];
            return;
          }
          const message: GuiUpdateMessage = {
            type: 'gui_update',
            name: key,
            value: value,
          };
        },
      };
    }
  });

  // Recursively wrap folders in a GUI config tree with Leva's `folder()`.
  function wrapFoldersInGuiConfigTree(
    guiConfigNode: { [key: string]: any },
    root: boolean,
  ) {
    const { _is_folder_marker, ...rest } = guiConfigNode;
    guiConfigNode = rest;

    if (root || _is_folder_marker === true) {
      const out: { [key: string]: any } = {};
      for (const [k, v] of Object.entries(guiConfigNode)) {
        out[k] = wrapFoldersInGuiConfigTree(v, false);
      }
      return root ? out : folder(out);
    }
    return guiConfigNode;
  }

  // Make Leva controls.
  const levaStore = useCreateStore();
  const [, set] = useControls(
    () => wrapFoldersInGuiConfigTree(guiConfigTree, true),
    { store: levaStore },
    [guiConfigFromName],
  );

  // Logic for setting control inputs when items are put onto the guiSetQueue.
  // const guiSetQueue = props.useGui((state) => state.guiSetQueue);
  // const applyGuiSetQueue = props.useGui((state) => state.applyGuiSetQueue);
  // const timeouts = React.useRef<{ [key: string]: NodeJS.Timeout }>({});
  // React.useEffect(() => {
  //   if (Object.keys(guiSetQueue).length === 0) return;
  //   applyGuiSetQueue((name, value) => {
  //     suppressOnChange.current[name] = true;

  //     // Suppression timeout. Resolves some issues with onChange() not firing
  //     // after we call set... this is hacky and should be revisited.
  //     clearTimeout(timeouts.current[name]);
  //     timeouts.current[name] = setTimeout(() => {
  //       suppressOnChange.current[name] = false;
  //     }, 10);

  //     // Set Leva control.
  //     set({ [name]: value });
  //   });
  // }, [guiSetQueue, applyGuiSetQueue, set]);

  // Leva theming is a bit limited, so we hack at styles here...
  return (
    <LevaPanel
      fill
      flat
      titleBar={false}
      store={levaStore}
      theme={LevaTheme}
      hideCopyButton
    />
  );
}

function ControlsLeva() {
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
  const colormapInvert = useSelector(
    (state) => state.renderingState.colormap_invert,
  );
  const colormapNormalize = useSelector(
    (state) => state.renderingState.colormap_normalize,
  );
  const max_resolution = useSelector(
    (state) => state.renderingState.maxResolution,
  );
  const target_train_util = useSelector(
    (state) => state.renderingState.targetTrainUtil,
  );
  const crop_enabled = useSelector(
    (state) => state.renderingState.crop_enabled,
  );

  const crop_bg_color = useSelector(
    (state) => state.renderingState.crop_bg_color,
  );

  const crop_scale = useSelector((state) => state.renderingState.crop_scale);

  const crop_center = useSelector((state) => state.renderingState.crop_center);

  const dispatch = useDispatch();

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
      colormap_invert: {
        label: '| Invert',
        value: colormapInvert,
        hint: 'Invert the colormap',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_invert',
            v,
          );
        },
        render: (get) => get('colormap_options') !== 'default',
      },
      colormap_normalize: {
        label: '| Normalize',
        value: colormapNormalize,
        hint: 'Whether to normalize output between 0 and 1',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_normalize',
            v,
          );
        },
        render: (get) => get('colormap_options') !== 'default',
      },
      colormap_range: {
        label: '| Range',
        value: [0, 1],
        step: 0.01,
        min: -2,
        max: 5,
        hint: 'Min and max values of the colormap',
        onChange: (v) => {
          dispatch_and_send(
            websocket,
            dispatch,
            'renderingState/colormap_range',
            v,
          );
        },
        render: (get) => get('colormap_options') !== 'default',
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
    }),
    [
      outputOptions,
      outputChoice,
      colormapOptions,
      colormapChoice,
      max_resolution,
      crop_enabled,
      target_train_util,
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
    crop_enabled,
    crop_bg_color,
    crop_scale,
    crop_center,
  ]);

  return null;
}

export function RenderControls() {
  return (
    <div className="Leva-container">
      <ControlsLeva />
      <Leva
        className="Leva-panel"
        theme={LevaTheme}
        titleBar={false}
        fill
        flat
      />
      <CustomLeva />
    </div>
  );
}
