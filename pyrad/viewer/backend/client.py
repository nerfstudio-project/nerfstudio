import argparse
import asyncio
import logging
import math
import traceback
import numpy
import cv2

import socketio
from av import VideoFrame

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp


class FlagVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self):
        super().__init__()  # don't forget this!
        self.counter = 0
        height, width = 480, 640

        # generate flag
        data_bgr = numpy.hstack(
            [
                self._create_rectangle(
                    width=213, height=480, color=(255, 0, 0)
                ),  # blue
                self._create_rectangle(
                    width=214, height=480, color=(255, 255, 255)
                ),  # white
                self._create_rectangle(width=213, height=480, color=(0, 0, 255)),  # red
            ]
        )

        # shrink and center it
        M = numpy.float32([[0.5, 0, width / 4], [0, 0.5, height / 4]])
        data_bgr = cv2.warpAffine(data_bgr, M, (width, height))

        # compute animation
        omega = 2 * math.pi / height
        id_x = numpy.tile(numpy.array(range(width), dtype=numpy.float32), (height, 1))
        id_y = numpy.tile(
            numpy.array(range(height), dtype=numpy.float32), (width, 1)
        ).transpose()

        self.frames = []
        for k in range(30):
            phase = 2 * k * math.pi / 30
            map_x = id_x + 10 * numpy.cos(omega * id_x + phase)
            map_y = id_y + 10 * numpy.sin(omega * id_x + phase)
            self.frames.append(
                VideoFrame.from_ndarray(
                    cv2.remap(data_bgr, map_x, map_y, cv2.INTER_LINEAR), format="bgr24"
                )
            )

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self.frames[self.counter % 30]
        frame.pts = pts
        frame.time_base = time_base
        self.counter += 1
        return frame

    def _create_rectangle(self, width, height, color):
        data_bgr = numpy.zeros((height, width, 3), numpy.uint8)
        data_bgr[:, :] = color
        return data_bgr


def object_from_json(message):
    print("got ", message)
    if message["type"] in ["answer", "offer"]:
        return RTCSessionDescription(**message)
    elif message["type"] == "candidate" and message["candidate"]:
        if len(message["candidate"]["candidate"]) != 0:
            candidate = candidate_from_sdp(message["candidate"]["candidate"])
            candidate.sdpMid = message["candidate"]["sdpMid"]
            candidate.sdpMLineIndex = message["candidate"]["sdpMLineIndex"]
            return candidate
        else:
            return None
    elif message["type"] == "bye":
        return BYE


def object_to_json(obj):
    if isinstance(obj, RTCSessionDescription):
        message = {"sdp": obj.sdp, "type": obj.type}
    elif isinstance(obj, RTCIceCandidate):
        message = {
            "type": "candidate",
            "candidate": {"candidate": candidate_to_sdp(obj), "sdpMid": obj.sdpMid, "sdpMLineIndex": obj.sdpMLineIndex},
        }
    else:
        assert obj is BYE
        message = {"type": "bye"}

    return message


class Ready:
    pass


class Signaling:
    def __init__(self):
        sio = socketio.AsyncClient()
        self.queue = asyncio.Queue()

        @sio.on("data")
        async def on_data(data):
            obj = object_from_json(data)
            if obj is not None:
                await self.queue.put(obj)

        @sio.on("ready")
        async def on_ready():
            await self.queue.put(Ready())

        self.sio = sio

    async def connect(self, host, port):
        await self.sio.connect("http://{}:{}".format(host, port))

    async def send(self, data):
        await self.sio.emit("data", object_to_json(data))

    async def receive(self):
        return await self.queue.get()

    async def close(self):
        pass


async def run(host, port, pc, video_track, recorder, signaling):
    await signaling.connect(host, port)

    def add_tracks():
        pc.addTrack(video_track)

    while True:
        obj = await signaling.receive()

        if isinstance(obj, Ready):
            print("ready")
            add_tracks()
            await pc.setLocalDescription(await pc.createOffer())
            await signaling.send(pc.localDescription)

        elif isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
            await recorder.start()

            if obj.type == "offer":
                print("received offer")
                # send answer
                add_tracks()
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await signaling.send(pc.localDescription)

        elif isinstance(obj, RTCIceCandidate):
            await pc.addIceCandidate(obj)

        elif obj is BYE:
            print("Exiting")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sighost", default="localhost")
    parser.add_argument("--sigport", default=8052)

    args = parser.parse_args()

    video_track = FlagVideoStreamTrack()

    recorder = MediaBlackhole()

    pc = RTCPeerConnection()
    signaling = Signaling()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run(args.sighost, args.sigport, pc, video_track, recorder, signaling))

    except KeyboardInterrupt:
        pass

    finally:
        print("stopping recorder")
        loop.run_until_complete(recorder.stop())
        print("closing signaling client")
        loop.run_until_complete(signaling.close())
        print("closing peer connection")
        loop.run_until_complete(pc.close())
        print("done")
