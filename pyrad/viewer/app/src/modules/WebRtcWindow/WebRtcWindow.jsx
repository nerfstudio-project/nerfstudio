import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { WebSocketContext } from '../WebSocket/WebSocket';

const WebRtcContext = createContext(null);
const msgpack = require('msgpack-lite');

export { WebRtcContext };

export default function WebRtcWindow() {
  const websocket = useContext(WebSocketContext).socket;
  const pcRef = useRef(null);
  const localVideoRef = useRef(null);

  const getRTCPeerConnection = () => {
    const pc = new RTCPeerConnection({
      iceServers: [
        {
          urls: 'stun:openrelay.metered.ca:80',
        },
        {
          urls: 'turn:openrelay.metered.ca:80',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
        {
          urls: 'turn:openrelay.metered.ca:443',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
        {
          urls: 'turn:openrelay.metered.ca:443?transport=tcp',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
      ],
    });
    // connect video
    pc.addEventListener('track', (evt) => {
      if (evt.track.kind === 'video') {
        [localVideoRef.current.srcObject] = evt.streams; // uses array destructuring
      }
    });
    pc.addTransceiver('video', { direction: 'recvonly' });
    return pc;
  };

  const sendOffer = () => {
    pcRef.current
      .createOffer()
      .then((offer) => {
        return pcRef.current.setLocalDescription(offer);
      })
      .then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
          if (pcRef.current.iceGatheringState === 'complete') {
            resolve();
          } else {
            const checkState = () => {
              if (pcRef.current.iceGatheringState === 'complete') {
                pcRef.current.removeEventListener(
                  'icegatheringstatechange',
                  checkState,
                );
                resolve();
              }
            };
            pcRef.current.addEventListener(
              'icegatheringstatechange',
              checkState,
            );
          }
        });
      })
      .then(() => {
        // send the offer
        const offer = pcRef.current.localDescription;
        const cmd = 'offer';
        const path = '';
        const data = {
          type: cmd,
          path,
          data: {
            sdp: offer.sdp,
            type: offer.type,
          },
        };
        const message = msgpack.encode(data);
        websocket.send(message);
      });
  };

  useEffect(() => {
    websocket.addEventListener('message', (originalCmd) => {
      // set the remote description when the offer is received
      const cmd = msgpack.decode(new Uint8Array(originalCmd.data));
      if (cmd.type === 'answer') {
        const answer = cmd.data;
        pcRef.current.setRemoteDescription(answer);
      }
    });

    websocket.addEventListener('open', () => {
      // connect webrtc when the websocket is connected
      pcRef.current = getRTCPeerConnection();
      sendOffer();
    });
  }, []); // empty dependency array means only run once

  return (
    <div className="WebRTCVideo">
      <video
        id="WebRTCVideo-video"
        autoPlay
        playsInline
        muted
        ref={localVideoRef}
      />
    </div>
  );
}
