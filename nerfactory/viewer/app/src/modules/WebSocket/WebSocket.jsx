// Much of this code comes from or is inspired by:
// https://www.pluralsight.com/guides/using-web-sockets-in-your-reactredux-app

import React, { createContext } from 'react';

import PropTypes from 'prop-types';
import { useDispatch } from 'react-redux';

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
  console.log("websocket url: ", endpoint);
  if (endpoint === undefined) {
    const message =
      'Please set the websocket endpoint. The format should look like "<viewer_url>?websocket_url=localhost:<port>". E.g., "https://viewer.nerfactory.com/branch/master/?websocket_url=localhost:7007".';
    window.alert(message);
    return '';
  }
  return endpoint;
}

export default function WebSocketContextFunction({ children }) {
  const dispatch = useDispatch();

  let socket;
  let ws;

  // should look like e.g., "ws://<localhost:port>"
  const websocketEndpoint = getWebsocketEndpoint();

  const connect = () => {
    socket = new WebSocket(`ws://${websocketEndpoint}/`);
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
      setTimeout(() => {
        connect();
      }, 1000);
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
