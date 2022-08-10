import React, { createContext, useContext, useEffect, useRef } from 'react';

import { useDispatch } from 'react-redux';
import { WebSocketContext } from '../WebSocket/WebSocket';

const WebRtcContext = createContext(null);
const msgpack = require('msgpack-lite');

export { WebRtcContext };

export default function WebRtcWindow() {
  const websocket = useContext(WebSocketContext).socket;
  const pcRef = useRef(null);
  const localVideoRef = useRef(null);

  const dispatch = useDispatch();

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
      dispatch({
        type: 'write',
        path: 'webrtcState/isConnected',
        data: true,
      });
      if (evt.track.kind === 'video') {
        [localVideoRef.current.srcObject] = evt.streams; // uses array destructuring
      }
    });
    pc.addTransceiver('video', { direction: 'recvonly' });

    // for updating the status of the peer connection
    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'connected') {
        dispatch({
          type: 'write',
          path: 'webrtcState/isConnected',
          data: true,
        });
      } else if (pc.iceConnectionState === 'disconnected') {
        dispatch({
          type: 'write',
          path: 'webrtcState/isConnected',
          data: false,
        });
      }
    };
    return pc;
  };

  const sendOffer = () => {
    pcRef.current
      .createOffer()
      .then((offer) => {
        console.log('[webrtc] created offer');
        return pcRef.current.setLocalDescription(offer);
      })
      .then(() => {
        console.log('[webrtc] set local description');
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
          if (pcRef.current.iceGatheringState === 'complete') {
            console.log('[webrtc] ICE gathering complete');
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
        console.log('[webrtc] sending offer');
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
      console.log('sending offer');
      sendOffer();
    });
  }, []); // empty dependency array means only run once

  return (
    <div className="WebRTCVideo">
      <video
        className="WebRTCVideo-video"
        autoPlay
        playsInline
        muted
        ref={localVideoRef}
      />
    </div>
  );
}
