import * as React from 'react';
import { useSelector } from 'react-redux';

export default function RenderWindow() {
  const isWebsocketConnected = useSelector(
    (state) => state.websocketState.isConnected,
  );

  return (
    <div className="RenderWindow">
      <div id="not-connected-overlay" hidden={isWebsocketConnected}>
        <div id="not-connected-overlay-text">Renderer Disconnected</div>
      </div>
      <img
        id="background-image"
        width="100%"
        height="100%"
        alt="Render window"
        z-index="1"
        hidden={!isWebsocketConnected}
      />
    </div>
  );
}
