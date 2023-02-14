import { split_path } from './utils';

const initialState = {
  // the websocket connection state
  websocketState: {
    isConnected: false,
    websocket_url: 'ws://localhost:7007',
  },
  // the webrtc connection state
  webrtcState: {
    isConnected: false,
  },
  // for sending actual commands to the client
  camera_path_payload: null,
  populate_paths_payload: false,
  // the rendering state
  renderingState: {
    // cameras
    camera_choice: 'Main Camera', // the camera being used to render the scene

    // camera path information
    config_base_dir: 'config_base_dir', // the base directory of the config file
    render_height: 1080,
    render_width: 1920,
    field_of_view: 50,
    camera_type: 'perspective',

    data_base_dir: 'data_base_dir', // the base directory of the images for saving camera path with the data
    export_path: 'export_path', // export name for render and camera_path

    all_camera_paths: null, // object containing camera paths and names

    isTraining: true,

    // colormap options
    output_options: ['rgb'], // populated by the possible Graph outputs
    output_choice: 'rgb', // the selected output
    colormap_options: ['default'], // populated by the output choice
    colormap_choice: 'default', // the selected colormap
    colormap_invert: false, // whether to invert the colormap
    colormap_normalize: false, // whether to normalize the colormap
    colormap_range: [0.0, 1.0], // the range of the colormap

    maxResolution: 1024,
    targetTrainUtil: 0.9,
    eval_res: '?',
    train_eta: 'Paused',
    vis_train_ratio: 'Paused',
    log_errors: '',
    renderTime: 0.0,

    // export options
    clipping_enabled: true,
    clipping_center: [0.0, 0.0, 0.0],
    clipping_box_scale: [2.0, 2.0, 2.0],

    // Crop Box Options
    crop_enabled: false,
    crop_bg_color: { r: 38, g: 42, b: 55 },
    crop_scale: [2.0, 2.0, 2.0],
    crop_center: [0.0, 0.0, 0.0],
  },
  // the scene state
  sceneState: {
    sceneBox: null,
    cameras: null,
  },
  // ------------------------------------------------------------------------
  // which of the state keys to synchronize with the bridge server
  sync_keys: ['renderingState', 'sceneState'],
};

function setData(newState, state, path, data) {
  if (path === 'colormap_options') {
    newState.colormap_choice = 'default'; // eslint-disable-line no-param-reassign
  }
  if (path.length === 1) {
    newState[path[0]] = data; // eslint-disable-line no-param-reassign
  } else {
    newState[path[0]] = { ...state[path[0]] }; // eslint-disable-line no-param-reassign
    setData(newState[path[0]], state[path[0]], path.slice(1), data);
  }
}

// Use the initialState as a default value
// eslint-disable-next-line default-param-last
export default function rootReducer(state = initialState, action) {
  // The reducer normally looks at the action type field to decide what happens

  switch (action.type) {
    case 'write': {
      const path = split_path(action.path); // convert string with "/"s to a list
      const data = action.data;
      const newState = { ...state };
      setData(newState, state, path, data);
      return newState;
    }
    default:
      // If this reducer doesn't recognize the action type, or doesn't
      // care about this specific action, return the existing state unchanged
      return state;
  }
}
