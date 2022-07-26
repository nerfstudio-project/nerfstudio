// Much of this code comes from or is inspired by:
// https://www.pluralsight.com/guides/using-web-sockets-in-your-reactredux-app

import React, { createContext, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';

// import io from 'socket.io-client';
import { updateChatLog } from './WebSocketSlice.js.js';

const WebSocketContext = createContext(null);

export { WebSocketContext };

function getWebsocketEndpointFromUrl(url: string) {
  console.log("i'm here");
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
  // const websocketState = useSelector((state) => state.websocketState);
  // const [isWebsocketConnected, setIsWebsocketConnected] = React.useState(
  //   websocketState.isConnected,
  // );

  const dispatch = useDispatch();

  let socket;
  let ws;

  // should look like ws://localhost:8051
  const websocketEndpoint = getWebsocketEndpointFromUrl(window.location.href);

  const sendMessage = (roomId, message) => {
    const payload = {
      roomId: roomId,
      data: message,
    };
    socket.emit('event://send-message', JSON.stringify(payload));
    dispatch(updateChatLog(payload));
  };

  // setIsWebsocketConnected(false);

  const connect = () => {
    socket = new WebSocket(`ws://${websocketEndpoint}/`);
    socket.onopen = () => {
      dispatch({
        type: 'websocketState/setIsConnected',
        boolean: true,
      });
      console.log('websocket connected');
    };

    socket.onmessage = (e) => {
      console.log('Message:', e.data);
    };

    socket.onclose = (e) => {
      console.log('on close');
      dispatch({
        type: 'websocketState/setIsConnected',
        boolean: false,
      });
      console.log(
        'Socket is closed. Reconnect will be attempted in 1 second.',
        e.reason,
      );
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
      socket: socket,
      sendMessage,
    };
  }

  // // Similar to componentDidMount and componentDidUpdate:
  // useEffect(() => {
  //   console.log('calling use effect');
  // }, []);

  return (
    <WebSocketContext.Provider value={ws}>{children}</WebSocketContext.Provider>
  );
}
