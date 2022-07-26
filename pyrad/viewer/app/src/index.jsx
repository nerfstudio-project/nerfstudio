import './index.css';

import * as serviceWorker from './serviceWorker';

import Alert from './modules/Alert/Alert';
import App from './App';
import { Provider } from 'react-redux';
import React from 'react';
import ReactDOM from 'react-dom/client';
import WebRtcProvider from './modules/WebRtcWindow/WebRtcWindow';
import WebSocketProvider from './modules/WebSocket/WebSocket';
import store from './store';

// console.log(store.getState());
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <Provider store={store}>
    <WebSocketProvider>
      <Alert />
      <App />
    </WebSocketProvider>
  </Provider>,
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
