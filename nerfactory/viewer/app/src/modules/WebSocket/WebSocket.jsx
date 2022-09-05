// Much of this code comes from or is inspired by:
// https://www.pluralsight.com/guides/using-web-sockets-in-your-reactredux-app

import React, { createContext, useEffect, useRef } from 'react';
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
  console.log('WebSocketContextFunction');

  const dispatch = useDispatch();

  // let socket_ref = useRef(null);
  // let socket = socket_ref.current;
  let socket;
  let ws;

  // useEffect(() => {
  //   // should look like e.g., "ws://<localhost:port>"
  //   const websocket_url_from_argument = getWebsocketEndpoint();
  //   console.log(websocket_url_from_argument);
  //   if (websocket_url_from_argument !== undefined) {
  //     dispatch({
  //       type: 'write',
  //       path: 'websocketState/websocket_url',
  //       data: websocket_url_from_argument,
  //     });
  //   } else {
  //     // otherwise open up the getting started modal
  //   }
  // }, []);

  // console.log('here!!');

  // this code will rerender anytime the webosocket changes now
  const websocket_url = useSelector(
    (state) => state.websocketState.websocket_url,
  );

  const connect = () => {
    const url = `ws://${websocket_url}/`;
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
      console.log("closed");
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

  // const selector_fn = (state) => {
  //   return state.websocketState.websocket_url;
  // };
  // const action_fn = (previous, current) => {
  //   const websocket_url = current;
  //   console.log(socket);
  //   connect(websocket_url);
  // };
  // subscribe_to_changes(selector_fn, action_fn);

  useEffect(() => {

    console.log("component mounted");
    
    // cleanup function to close the websocket
    return () => {
      console.log("calling cleanup!!!");
      socket.close();
    };
  }, [websocket_url]);

  // if (!socket) {
  //   console.log("inside here");
  //   socket = connect();
  //   ws = {
  //     socket: socket,
  //   };
  // } else {
  //   console.log('else of socket.');
  // }

  console.log("INSIDE THE PROVIDER");

  connect();
  ws = {
    socket
  };

  return (
    <WebSocketContext.Provider value={ws}>{children}</WebSocketContext.Provider>
  );
}

WebSocketContextFunction.propTypes = {
  children: PropTypes.node.isRequired,
};
