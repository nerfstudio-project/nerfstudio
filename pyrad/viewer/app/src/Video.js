import React, { Component } from "react";

export class WebRTCVideo extends Component {
  render() {
    return (
      <div>
        <h2> State </h2>
        <p>
          ICE gathering state: <span id="ice-gathering-state"> </span>
        </p>
        <p>
          ICE connection state: <span id="ice-connection-state"> </span>
        </p>
        <p>
          Signaling state: <span id="signaling-state"> </span>
        </p>
        <div id="media" style={{ display: "none" }}>
          <h2> Media </h2>
          <audio id="audio" autoPlay></audio>
          <video id="video" autoPlay playsInline></video>
        </div>
      </div>
    );
  }
}
