import * as serviceWorker from './serviceWorker';

import App from './App';
import { Provider } from 'react-redux';
import React from 'react';
import ReactDOM from 'react-dom/client';
import WebSocketProvider from './modules/WebSocket/WebSocket';
import store from './store';
import Alert from './modules/Alert/Alert';

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
