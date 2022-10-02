// Much of this code comes from or is inspired by:
// https://www.pluralsight.com/guides/using-web-sockets-in-your-reactredux-app

import React, { createContext, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import PropTypes from 'prop-types';

const WebSocketContext = createContext(null);

export { WebSocketContext };

export default function WebSocketContextFunction({ children }) {
  const dispatch = useDispatch();
  let ws = null;
  let socket = null;

  // this code will rerender anytime the webosocket changes now
  const websocket_port = useSelector(
    (state) => state.websocketState.websocket_port,
  );

  const connect = () => {
    const url = `ws://localhost:${websocket_port}/`;
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
    return socket;
  };

  useEffect(() => {
    // cleanup function to close the websocket on rerender
    return () => {
      socket.close();
    };
  }, [websocket_port]);

  connect();
  ws = {
    socket,
  };

  return (
    <WebSocketContext.Provider value={ws}>{children}</WebSocketContext.Provider>
  );
}

WebSocketContextFunction.propTypes = {
  children: PropTypes.node.isRequired,
};
