# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client class"""
import argparse
import asyncio
import json
import logging
import math
import os
import ssl

import aiohttp_cors
import cv2
import numpy
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame

ROOT = os.path.dirname(__file__)


class SingleFrameStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()  # don't forget this!

        self.last_frame = None
        self.frame_queue = asyncio.Queue(10)

    async def put_frame(self, frame):
        await self.frame_queue.put(VideoFrame.from_ndarray(frame))

    async def recv(self):
        if self.last_frame is None:
            frame = await self.frame_queue.get()

        else:
            try:
                frame = self.frame_queue.get_nowait()

            except asyncio.QueueEmpty:
                frame = self.last_frame

        pts, time_base = await self.next_timestamp()

        frame.pts = pts
        frame.time_base = time_base
        self.last_frame = frame
        return frame


def _create_rectangle(width, height, color):
    data_bgr = numpy.zeros((height, width, 3), numpy.uint8)
    data_bgr[:, :] = color
    return data_bgr


def _gen_test_image(k):
    height, width = 480, 640

    # generate flag
    data_bgr = numpy.hstack(
        [
            _create_rectangle(width=213, height=480, color=(255, 0, 0)),  # blue
            _create_rectangle(width=214, height=480, color=(255, 255, 255)),  # white
            _create_rectangle(width=213, height=480, color=(0, 0, 255)),  # red
        ]
    )

    # shrink and center it
    M = numpy.float32([[0.5, 0, width / 4], [0, 0.5, height / 4]])
    data_bgr = cv2.warpAffine(data_bgr, M, (width, height))

    # compute animation
    omega = 2 * math.pi / height
    id_x = numpy.tile(numpy.array(range(width), dtype=numpy.float32), (height, 1))
    id_y = numpy.tile(numpy.array(range(height), dtype=numpy.float32), (width, 1)).transpose()

    phase = 2 * k * math.pi / 30
    map_x = id_x + 10 * numpy.cos(omega * id_x + phase)
    map_y = id_y + 10 * numpy.sin(omega * id_x + phase)
    return cv2.remap(data_bgr, map_x, map_y, cv2.INTER_LINEAR)


async def gen_image_frames(video_track):
    k = 0
    while True:
        if k > 500:
            k = 0
        img = _gen_test_image(k)
        k += 1

        await video_track.put_frame(img)


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences([codec for codec in codecs if codec.mimeType == forced_codec])


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    video = SingleFrameStreamTrack()
    loop = asyncio.get_event_loop()
    try:
        asyncio.run_coroutine_threadsafe(gen_image_frames(video), loop)
    except KeyboardInterrupt:
        pass
    finally:
        pc.close()

    if video:
        video_sender = pc.addTrack(video)
        if args.video_codec:
            force_codec(pc, video_sender, args.video_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the video codec using --video-codec")

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8052, help="Port for HTTP server (default: 8080)")
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--audio-codec", help="Force a specific audio codec (e.g. audio/opus)")
    parser.add_argument("--video-codec", help="Force a specific video codec (e.g. video/H264)")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None  # pylint: disable=invalid-name

    app = web.Application()
    app.router.add_route("POST", "/offer", offer)
    app.on_shutdown.append(on_shutdown)
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        },
    )
    for route in list(app.router.routes()):
        cors.add(route)

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
