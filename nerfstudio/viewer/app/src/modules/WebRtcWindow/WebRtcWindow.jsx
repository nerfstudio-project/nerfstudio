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
        { urls: 'stun:stun.l.google.com:19302' },
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

    // for updating the status of the peer connection
    pc.oniceconnectionstatechange = () => {
      // https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/connectionState
      console.log(`[webrtc] connectionState: ${pc.connectionState}`);
      if (
        pc.connectionState === 'connecting' ||
        pc.connectionState === 'connected'
      ) {
        console.log('[webrtc] connected');
        dispatch({
          type: 'write',
          path: 'webrtcState/isConnected',
          data: true,
        });
      } else {
        dispatch({
          type: 'write',
          path: 'webrtcState/isConnected',
          data: false,
        });
      }
    };

    pc.onclose = () => {
      dispatch({
        type: 'write',
        path: 'webrtcState/isConnected',
        data: false,
      });
    };

    return pc;
  };

  const sendOffer = () => {
    pcRef.current
      .createOffer()
      .then((offer) => {
        console.log('[webrtc] created offer');
        console.log(offer);
        return pcRef.current.setLocalDescription(offer);
      })
      .then(() => {
        // wait for ICE gathering to complete
        console.log('[webrtc] set local description');
        return new Promise((resolve) => {
          if (pcRef.current.iceGatheringState === 'complete') {
            console.log('[webrtc] ICE gathering complete');
            resolve();
          } else {
            const checkState = () => {
              console.log(
                `[webrtc] iceGatheringState: ${pcRef.current.iceGatheringState}`,
              );
              if (pcRef.current.iceGatheringState === 'complete') {
                pcRef.current.removeEventListener(
                  'icegatheringstatechange',
                  checkState,
                );
                resolve();
              }
            };
            console.log(
              '[webrtc] adding listener for `icegatheringstatechange`',
            );
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
        const cmd = 'write';
        const path = 'webrtc/offer';
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
      if (cmd.path === '/webrtc/answer') {
        console.log('[webrtc] received answer');
        const answer = cmd.data;
        console.log(answer);
        if (answer !== null) {
          pcRef.current.setRemoteDescription(answer);
        }
      }
    });

    websocket.addEventListener('open', () => {
      // connect webrtc when the websocket is connected
      pcRef.current = getRTCPeerConnection();
      console.log('[webrtc] starting process');
      sendOffer();
    });

    // kill the webrtc connection on dismount
    return () => {
      if (pcRef.current !== null) {
        dispatch({
          type: 'write',
          path: 'webrtcState/isConnected',
          data: false,
        });
        pcRef.current.close();
      }
    };
  }, [websocket]); // dependency to call this whenever the websocket changes

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
