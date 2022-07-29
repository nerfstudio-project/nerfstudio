import { split_path } from './utils';

const initialState = {
  // the websocket connection state
  websocketState: {
    isConnected: false,
  },
  // the webrtc connection state
  webrtcState: {
    isConnected: false,
  },
  // the rendering state
  renderingState: {
    camera: null,
    isTraining: false,
    output_options: ['temp', 'temp3'], // populated by the possible Graph outputs
    output_choice: null, // the selected output
    minResolution: 100,
    maxResolution: 500,
  },
  // the scene state
  sceneState: {
    sceneBounds: null,
    cameras: {},
  },
  // ------------------------------------------------------------------------
  // which of the state keys to synchronize with the bridge server
  sync_keys: ['rendering_state', 'scene_state'],
};

function setData(newState, state, path, data) {
  if (path.length === 0) {
  }
  else if (path.length === 1) {
    newState[path[0]] = data;
  }
  else {
    newState[path[0]] = {...state[path[0]]};
    setData(newState[path[0]], state[path[0]], path.slice(1), data);
  }
}

// Use the initialState as a default value
export default function rootReducer(state = initialState, action) {
  // The reducer normally looks at the action type field to decide what happens

  switch (action.type) {
    case 'write': {
      const path = split_path(action.path); // convert string with "/"s to a list
      const data = action.data;
      const newState = {...state};
      console.log("before");
      console.log(newState);
      setData(newState, state, path, data);
      console.log(newState);
      return newState;
    }
    default:
      // If this reducer doesn't recognize the action type, or doesn't
      // care about this specific action, return the existing state unchanged
      return state;
  }
}
