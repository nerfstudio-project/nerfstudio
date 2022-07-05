function createPeerConnection() {
  var config = {
    sdpSemantics: "unified-plan",
  };

  // if (document.getElementById('use-stun').checked) {
  //     config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
  // }

  let pc = new RTCPeerConnection(config);

  // register some listeners to help debugging
  pc.addEventListener(
    "icegatheringstatechange",
    function () {
      iceGatheringLog.textContent += " -> " + pc.iceGatheringState;
    },
    false
  );
  iceGatheringLog.textContent = pc.iceGatheringState;

  pc.addEventListener(
    "iceconnectionstatechange",
    function () {
      iceConnectionLog.textContent += " -> " + pc.iceConnectionState;
    },
    false
  );
  iceConnectionLog.textContent = pc.iceConnectionState;

  pc.addEventListener(
    "signalingstatechange",
    function () {
      signalingLog.textContent += " -> " + pc.signalingState;
    },
    false
  );
  signalingLog.textContent = pc.signalingState;

  // connect audio / video
  pc.addEventListener("track", function (evt) {
    if (evt.track.kind == "video")
      document.getElementById("video").srcObject = evt.streams[0];
    else document.getElementById("audio").srcObject = evt.streams[0];
  });

  return pc;
}
