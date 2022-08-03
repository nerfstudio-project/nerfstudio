// Much of this code comes from or is inspired by:
// https://www.pluralsight.com/guides/using-web-sockets-in-your-reactredux-app

import React, { createContext } from 'react';

import PropTypes from 'prop-types';
import { useDispatch } from 'react-redux';

const WebSocketContext = createContext(null);

export { WebSocketContext };

function getWebsocketEndpointFromUrl(url: string) {
  const splitUrl = url.split('?');
  if (splitUrl.length !== 2) {
    window.alert("There should be exactly one '?' in the url.");
    return '';
  }
  const endpoint = splitUrl.pop();
  if (endpoint === '') {
    const message =
      'Please set the websocket endpoint. E.g., a correct URL may be: http://localhost:4000?localhost:8051';
    window.alert(message);
    return '';
  }
  return endpoint;
}

export default function WebSocketContextFunction({ children }) {
  const dispatch = useDispatch();

  let socket;
  let ws;

  // should look like ws://localhost:8051
  const websocketEndpoint = getWebsocketEndpointFromUrl(window.location.href);

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
