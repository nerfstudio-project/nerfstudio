import { split_path } from './utils';

const initialState = {
  // the websocket connection state
  websocketState: {
    websocket_url: 'localhost:7007',
    isConnected: false,
  },
  // the webrtc connection state
  webrtcState: {
    isConnected: false,
  },
  // the rendering state
  renderingState: {
    camera: null, // the camera object used to render the three.js scene
    isTraining: true,
    output_options: ['rgb'], // populated by the possible Graph outputs
    output_choice: 'rgb', // the selected output
    colormap_options: ['default'], // populated by the output choice
    colormap_choice: 'default', // the selected colormap
    minResolution: 50,
    maxResolution: 512,
    field_of_view: 80,
    eval_fps: 'Paused',
    train_eta: 'Paused',
    vis_train_ratio: 'Paused',
  },

  // the scene state
  sceneState: {
    sceneBounds: null,
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
