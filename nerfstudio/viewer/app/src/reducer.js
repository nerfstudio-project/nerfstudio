import { split_path } from './utils';

const initialState = {
  // the websocket connection state
  websocketState: {
    isConnected: false,
    websocket_url: 'ws://localhost:7007',
  },

  // for sending actual commands to the client
  camera_path_payload: null,
  populate_paths_payload: false,

  render_img: null, // The rendered images

  show_export_box: false, // whether to show the export box

  custom_gui: {
    guiNames: [],
    guiConfigFromName: {},
    guiSetQueue: {},
  },

  file_path_info: {
    config_base_dir: 'config_base_dir', // the base directory of the config file
    data_base_dir: 'data_base_dir', // the base directory of the images for saving camera path with the data
    export_path_name: 'export_path_name', // export name for render and camera_path
  },

  all_camera_paths: null, // object containing camera paths and names

  // the rendering state
  renderingState: {
    // cameras
    camera_choice: 'Main Camera', // the camera being used to render the scene

    // camera path information
    render_height: 1080,
    render_width: 1920,
    field_of_view: 50,
    camera_type: 'perspective',

    training_state: 'training',

    step: 0,
    eval_res: '?',

    // export options
    clipping_enabled: true,
    clipping_center: [0.0, 0.0, 0.0],
    clipping_box_scale: [2.0, 2.0, 2.0],

    // Crop Box Options
    crop_enabled: false,
    crop_bg_color: { r: 38, g: 42, b: 55 },
    crop_scale: [2.0, 2.0, 2.0],
    crop_center: [0.0, 0.0, 0.0],

    // Time options
    use_time_conditioning: false,
  },
  // the scene state
  sceneState: {
    sceneBox: null,
    cameras: null,
  },
};

// Recursive function to update the state object with new data at a given path
function setData(state, path, data) {
  // If we've reached the final level of the path, update the property with the new data
  if (path.length === 1) {
    // Use the spread operator to create a shallow copy of the state object
    // and update the property with the new data
    return { ...state, [path[0]]: data };
  }
  // If we haven't reached the final level of the path, recursively update the nested object
  // by creating a shallow copy of the parent object and updating the relevant property
  return {
    ...state,
    [path[0]]: setData(state[path[0]], path.slice(1), data),
  };
}

// Reducer function that handles the state updates
// eslint-disable-next-line default-param-last
export default function rootReducer(state = initialState, action) {
  switch (action.type) {
    case 'write': {
      // Destructure the path and data values from the action object
      const { path, data } = action;
      // Split the path string into an array of path segments
      const pathSegments = split_path(path);
      // Call the setData function to update the state object with the new data
      const newState = setData(state, pathSegments, data);
      // Return the updated state object
      return newState;
    }
    // If the reducer doesn't recognize the action type, return the existing state unchanged
    default:
      return state;
  }
}
