import './index.scss';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import App from './App';
import WebSocketProvider from './modules/WebSocket/WebSocket';
import store from './store';
import { ViserWebSocket } from './modules/WebSocket/ViserWebSocket';

const root = createRoot(document.getElementById('root'));
root.render(
  <Provider store={store}>
    <WebSocketProvider>
      <ViserWebSocket />
      <App />
    </WebSocketProvider>
  </Provider>,
);
