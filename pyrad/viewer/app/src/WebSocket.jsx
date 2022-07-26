import React, { createContext } from 'react';

import io from 'socket.io-client';
import { updateChatLog } from './actions';
import { useDispatch } from 'react-redux';

const WebSocketContext = createContext(null);

export { WebSocketContext };

function websocket_endpoint_from_url(url) {
  const endpoint = url.split('?').pop();
  if (endpoint == '') {
    const message =
      'Please set the websocket endpoint. E.g., a correct URL may be: http://localhost:4000?localhost:8051';
    window.alert(message);
    return null;
  }
  return endpoint;
}

export default function WebSocketContextFunction({ children }) {
  let socket;
  let ws;

  const WS_BASE = websocket_endpoint_from_url(window.location.href);
  console.log('WS_BASE');
  console.log(WS_BASE);

  const dispatch = useDispatch();

  const sendMessage = (roomId, message) => {
    const payload = {
      roomId: roomId,
      data: message,
    };
    socket.emit('event://send-message', JSON.stringify(payload));
    dispatch(updateChatLog(payload));
  };

  if (!socket) {
    socket = io.connect(WS_BASE);

    socket.on('event://get-message', (msg) => {
      const payload = JSON.parse(msg);
      // TODO(ethan): update the redux store with the new message information
      dispatch(updateChatLog(payload));
    });

    ws = {
      socket: socket,
      sendMessage,
    };
  }

  return (
    <WebSocketContext.Provider value={ws}>{children}</WebSocketContext.Provider>
  );
}

// https://www.pluralsight.com/guides/using-web-sockets-in-your-reactredux-app
