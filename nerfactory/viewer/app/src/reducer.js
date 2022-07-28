const initialState = {
  // todos: [
  //   { id: 0, text: 'Learn React', completed: true },
  //   { id: 1, text: 'Learn Redux', completed: false, color: 'purple' },
  //   { id: 2, text: 'Build something fun!', completed: false, color: 'blue' },
  // ],
  // filters: {
  //   status: 'All',
  //   colors: [],
  // },
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
    box: null,
    cameras: {},
  },
  // ------------------------------------------------------------------------
  // which of the state keys to synchronize with the bridge server
  sync_keys: ['rendering_state', 'scene_state'],
};

// Use the initialState as a default value
export default function rootReducer(state = initialState, action) {
  // The reducer normally looks at the action type field to decide what happens

  switch (action.type) {
    // Do something here based on the different types of actions
    case 'websocketState/setIsConnected':
      return {
        ...state,
        websocketState: {
          ...state.websocketState,
          isConnected: action.boolean,
        },
      };
    case 'webrtcState/setIsConnected':
      return {
        ...state,
        webrtcState: {
          ...state.websocketState,
          isConnected: action.boolean,
        },
      };
    default:
      // If this reducer doesn't recognize the action type, or doesn't
      // care about this specific action, return the existing state unchanged
      return state;
  }
}
