// Much of this code comes from or is inspired by:
// https://www.pluralsight.com/guides/using-web-sockets-in-your-reactredux-app

import React, { createContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import PropTypes from 'prop-types';
import { subscribe_to_changes } from '../../subscriber';

const WebSocketContext = createContext(null);

export { WebSocketContext };

function getParam(param_name) {
  // https://stackoverflow.com/questions/831030/how-to-get-get-request-parameters-in-javascript
  const params = new RegExp(
    `[?&]${encodeURIComponent(param_name)}=([^&]*)`,
  ).exec(window.location.href);
  if (params === null) {
    return undefined;
  }
  return decodeURIComponent(params[1]);
}

function getWebsocketEndpoint() {
  const endpoint = getParam('websocket_url');
  return endpoint;
}

export default function WebSocketContextFunction({ children }) {
  const dispatch = useDispatch();

  let socket;
  let ws;

  useEffect(() => {
    // should look like e.g., "ws://<localhost:port>"
    const websocket_url_from_argument = getWebsocketEndpoint();
    console.log(websocket_url_from_argument);
    if (websocket_url_from_argument !== undefined) {
      dispatch({
        type: 'write',
        path: 'websocketState/websocket_url',
        data: websocket_url_from_argument,
      });
    }
  }, []);

  console.log('here!!');

  const websocket_url = useSelector(
    (state) => state.websocketState.websocket_url,
  );

  const connect = () => {
    const url = `ws://${websocket_url}/`;
    console.log(url);
    socket = new WebSocket(url);
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => {
      dispatch({
        type: 'write',
        path: 'websocketState/isConnected',
        data: true,
      });
    };

    socket.onclose = () => {
      // when closed, the websocket will try to reconnect every second
      dispatch({
        type: 'write',
        path: 'websocketState/isConnected',
        data: false,
      });
    };

    socket.onerror = (err) => {
      console.error(
        'Socket encountered error: ',
        err.message,
        'Closing socket',
      );
      socket.close();
    };
  };

  // const selector_fn = (state) => {
  //   return state.websocketState.websocket_url;
  // };
  // const action_fn = (previous, current) => {
  //   if (socket) {
  //     socket.close();
  //   }
  //   connect();
  // };
  // subscribe_to_changes(selector_fn, action_fn);

  if (!socket) {
    connect();
    ws = {
      socket,
    };
  }

  return (
    <WebSocketContext.Provider value={ws}>{children}</WebSocketContext.Provider>
  );
}

WebSocketContextFunction.propTypes = {
  children: PropTypes.node.isRequired,
};
