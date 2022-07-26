import React, { createContext, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
const WebRtcContext = createContext(null);

export { WebRtcContext };

export default function WebRtcContextFunction({ children }) {
  const dispatch = useDispatch();

  let context = {
    pc: 'my pc',
  };

//   const getRTCPeerConnection = () => {
//   }

  // Similar to componentDidMount and componentDidUpdate:
  useEffect(() => {
    console.log('calling use effect from webrtc context');
    console.log(document.getElementById('WebRTCVideo-video'));
  }, []);

  return (
    <WebRtcContext.Provider value={context}>{children}</WebRtcContext.Provider>
  );
}
