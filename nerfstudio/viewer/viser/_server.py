# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

""" Core Viser Server """
# pylint: disable=protected-access
# pylint: disable=too-many-statements

from __future__ import annotations

import asyncio
import dataclasses
import threading
from asyncio.events import AbstractEventLoop
from typing import Callable, Dict, List, NewType

import rich
import websockets.connection
import websockets.datastructures
import websockets.exceptions
import websockets.server
from typing_extensions import override
from websockets.legacy.server import WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._message_api import MessageApi
from ._messages import Message


@dataclasses.dataclass
class _ClientHandleState:
    # Internal state for ClientHandle objects.
    message_buffer: asyncio.Queue
    event_loop: AbstractEventLoop
    camera_cb: List[Callable[[ClientHandle], None]]


@dataclasses.dataclass
class ClientHandle(MessageApi):
    """Handle for interacting with a single connected client.

    We can use this to read the camera state or send client-specific messages."""

    client_id: ClientId
    _state: _ClientHandleState

    def __post_init__(self) -> None:
        super().__init__()

    @override
    def _queue(self, message: Message) -> None:
        """Implements message enqueue required by MessageApi.
        Pushes a message onto a client-specific queue.

        Args:
            message: Message to enqueue."""
        self._state.event_loop.call_soon_threadsafe(self._state.message_buffer.put_nowait, message)


ClientId = NewType("ClientId", int)


class ViserServer(MessageApi):
    """Core visualization server. Communicates asynchronously with client applications
    via websocket connections.

    By default, all messages (eg `server.add_frame()`) are broadcasted to all connected
    clients.

    To send messages to an individual client, we can grab a client ID -> handle mapping
    via `server.get_clients()`, and then call `client.add_frame()` on the handle.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        super().__init__()

        # Track connected clients.
        self._handle_from_client: Dict[ClientId, _ClientHandleState] = {}
        self._client_lock = threading.Lock()

        # Start server thread.
        ready_sem = threading.Semaphore(value=1)
        ready_sem.acquire()  # pylint: disable=consider-using-with
        threading.Thread(
            target=lambda: self._background_worker(host, port, ready_sem),
            daemon=True,
        ).start()

        # Wait for the thread to set self._event_loop and self._broadcast_buffer...
        ready_sem.acquire()  # pylint: disable=consider-using-with

        # Broadcast buffer should be populated by the background worker.
        assert isinstance(self._broadcast_buffer, AsyncMessageBuffer)

        # Reset the scene.
        self.reset_scene()

    def get_clients(self) -> Dict[ClientId, ClientHandle]:
        """Get a mapping from client IDs to client handles.

        We can use client handles to get camera information, send individual messages to
        clients, etc."""

        with self._client_lock:
            out = {k: ClientHandle(k, v) for k, v in self._handle_from_client.items()}
        return out

    @override
    def _queue(self, message: Message) -> None:
        """Implements message enqueue required by MessageApi.

        Pushes a message onto a broadcast queue."""
        self._broadcast_buffer.push(message)

    def _background_worker(
        self,
        host: str,
        port: int,
        ready_sem: threading.Semaphore,
    ) -> None:
        # Need to make a new event loop for notebook compatbility.
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        self._event_loop = event_loop
        self._broadcast_buffer = AsyncMessageBuffer(event_loop)
        ready_sem.release()

        count_lock = asyncio.Lock()
        connection_count = 0
        total_connections = 0

        async def serve(websocket: WebSocketServerProtocol) -> None:
            """Server loop, run once per connection."""

            async with count_lock:
                nonlocal connection_count
                client_id = ClientId(connection_count)  # pylint: disable=used-before-assignment
                connection_count += 1

                nonlocal total_connections
                total_connections += 1

            rich.print(
                f"[bold](viser)[/bold] Connection opened ({client_id},"
                f" {total_connections} total),"
                f" {len(self._broadcast_buffer.message_from_id)} persistent messages"
            )

            client_handle = _ClientHandleState(
                message_buffer=asyncio.Queue(),
                event_loop=event_loop,
                camera_cb=[],
            )
            with self._client_lock:
                self._handle_from_client[client_id] = client_handle

            def handle_incoming(message: Message) -> None:
                self._handle_incoming_message(client_id, message)
                ClientHandle(client_id, client_handle)._handle_incoming_message(client_id, message)

            try:
                # For each client: infinite loop over producers (which send messages)
                # and consumers (which receive messages).
                await asyncio.gather(
                    _single_connection_producer(websocket, client_handle.message_buffer),
                    _broadcast_producer(websocket, client_id, self._broadcast_buffer),
                    _consumer(websocket, client_id, handle_incoming),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                # Cleanup.
                rich.print(f"[bold](viser)[/bold] Connection closed ({client_id}," f" {total_connections} total)")
                with self._client_lock:
                    self._handle_from_client.pop(client_id)

        for _ in range(500):
            try:
                event_loop.run_until_complete(
                    websockets.server.serve(
                        serve,
                        host,
                        port,
                        compression=None,
                    )
                )
                break
            except OSError:  # Port not available.
                port += 1
                continue

        event_loop.run_forever()


async def _single_connection_producer(websocket: WebSocketServerProtocol, buffer: asyncio.Queue) -> None:
    """Infinite loop to send messages from the client buffer."""
    while True:
        message = await buffer.get()
        await websocket.send(message.serialize())


async def _broadcast_producer(
    websocket: WebSocketServerProtocol, client_id: ClientId, buffer: AsyncMessageBuffer
) -> None:
    """Infinite loop to send messages from the broadcast buffer."""
    async for message in buffer:
        if message.excluded_self_client == client_id:
            continue
        await websocket.send(message.serialize())


async def _consumer(
    websocket: WebSocketServerProtocol,
    client_id: ClientId,  # pylint: disable=unused-argument
    handle_message: Callable[[Message], None],
) -> None:
    """Infinite loop waiting for and then handling incoming messages."""
    while True:
        raw = await websocket.recv()
        assert isinstance(raw, bytes)
        message = Message.deserialize(raw)
        handle_message(message)
