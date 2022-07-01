import React, { Component } from "react";

var pc = null;

function negotiate() {
  pc.addTransceiver("video", { direction: "recvonly" });
  return pc
    .createOffer()
    .then(function (offer) {
      return pc.setLocalDescription(offer);
    })
    .then(function () {
      // wait for ICE gathering to complete
      return new Promise(function (resolve) {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          function checkState() {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", checkState);
              resolve();
            }
          }
          pc.addEventListener("icegatheringstatechange", checkState);
        }
      });
    })
    .then(function () {
      var offer = pc.localDescription;
      return fetch("http://localhost:8052/offer", {
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    })
    .then(function (response) {
      return response.json();
    })
    .then(function (answer) {
      return pc.setRemoteDescription(answer);
    })
    .catch(function (e) {
      alert(e);
    });
}

function start() {
  var config = {
    sdpSemantics: "unified-plan",
  };
  config.iceServers = [{ urls: ["stun:stun.l.google.com:19302"] }];

  pc = new RTCPeerConnection(config);

  // connect audio / video
  pc.addEventListener("track", function (evt) {
    if (evt.track.kind == "video") {
      console.log("setting event steam");
      console.log(evt.streams[0]);
      document.getElementById("WebRTCVideo-video").srcObject = evt.streams[0];
    }
  });

  negotiate();
}

export class WebRTCVideo extends Component {
  componentDidMount() {
    start();
  }

  render() {
    return (
      <div className="WebRTCVideo">
        <video id="WebRTCVideo-video" autoPlay playsInline muted></video>
      </div>
    );
  }
}
