from __future__ import annotations

import asyncio
import dataclasses
import http.server
import mimetypes
import threading
import time
from asyncio.events import AbstractEventLoop
from pathlib import Path
from typing import Callable, Dict, List, NewType, Optional, Tuple

import rich
import websockets.connection
import websockets.datastructures
import websockets.exceptions
import websockets.server
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from typing_extensions import override
from websockets.legacy.server import WebSocketServerProtocol

from ._async_message_buffer import AsyncMessageBuffer
from ._message_api import MessageApi
from ._messages import Message, ViewerCameraMessage


@dataclasses.dataclass(frozen=True)
class CameraState:
    """Information about a client's camera state."""

    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    fov: float
    aspect: float
    last_updated: float


@dataclasses.dataclass
class _ClientHandleState:
    # Internal state for ClientHandle objects.
    camera_info: Optional[CameraState]
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

        def handle_camera(client_id: ClientId, message: Message) -> None:
            """Handle camera messages."""
            if not isinstance(message, ViewerCameraMessage):
                return
            self._state.camera_info = CameraState(
                message.wxyz, message.position, message.fov, message.aspect, time.time()
            )
            for cb in self._state.camera_cb:
                cb(self)

        self._incoming_handlers.append(handle_camera)

    def get_camera(self) -> CameraState:
        while self._state.camera_info is None:
            time.sleep(0.01)
        return self._state.camera_info

    def on_camera_update(
        self, callback: Callable[[ClientHandle], None]
    ) -> Callable[[ClientHandle], None]:
        self._state.camera_cb.append(callback)
        return callback

    @override
    def _queue(self, message: Message) -> None:
        """Implements message enqueue required by MessageApi.

        Pushes a message onto a client-specific queue."""
        self._state.event_loop.call_soon_threadsafe(
            self._state.message_buffer.put_nowait, message
        )


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
        http_server: bool = True,
    ):
        super().__init__()

        # Track connected clients.
        self._handle_from_client: Dict[ClientId, _ClientHandleState] = {}
        self._client_lock = threading.Lock()

        # Start server thread.
        ready_sem = threading.Semaphore(value=1)
        ready_sem.acquire()
        threading.Thread(
            target=lambda: self._background_worker(host, port, ready_sem, http_server),
            daemon=True,
        ).start()

        # Wait for the thread to set self._event_loop and self._broadcast_buffer...
        ready_sem.acquire()

        # Broadcast buffer should be populated by the background worker.
        assert isinstance(self._broadcast_buffer, AsyncMessageBuffer)

        # Reset the scene.
        self.reset_scene()

    def get_clients(self) -> Dict[ClientId, ClientHandle]:
        """Get a mapping from client IDs to client handles.

        We can use client handles to get camera information, send individual messages to
        clients, etc."""

        self._client_lock.acquire()
        out = {k: ClientHandle(k, v) for k, v in self._handle_from_client.items()}
        self._client_lock.release()
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
        http_server: bool,
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
                client_id = ClientId(connection_count)
                connection_count += 1

                nonlocal total_connections
                total_connections += 1

            rich.print(
                f"[bold](viser)[/bold] Connection opened ({client_id},"
                f" {total_connections} total),"
                f" {len(self._broadcast_buffer.message_from_id)} persistent messages"
            )

            client_handle = _ClientHandleState(
                camera_info=None,
                message_buffer=asyncio.Queue(),
                event_loop=event_loop,
                camera_cb=[],
            )
            self._client_lock.acquire()
            self._handle_from_client[client_id] = client_handle
            self._client_lock.release()

            def handle_incoming(message: Message) -> None:
                self._handle_incoming_message(client_id, message)
                ClientHandle(client_id, client_handle)._handle_incoming_message(
                    client_id, message
                )

            try:
                # For each client: infinite loop over producers (which send messages)
                # and consumers (which receive messages).
                await asyncio.gather(
                    _single_connection_producer(
                        websocket, client_handle.message_buffer
                    ),
                    _broadcast_producer(websocket, client_id, self._broadcast_buffer),
                    _consumer(websocket, client_id, handle_incoming),
                )
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                # Cleanup.
                rich.print(
                    f"[bold](viser)[/bold] Connection closed ({client_id},"
                    f" {total_connections} total)"
                )
                total_connections -= 1
                self._client_lock.acquire()
                self._handle_from_client.pop(client_id)
                self._client_lock.release()

        # Host client on the same port as the websocket.
        async def viser_http_server(
            path: str, request_headers: websockets.datastructures.Headers
        ) -> Optional[
            Tuple[http.HTTPStatus, websockets.datastructures.HeadersLike, bytes]
        ]:
            # Ignore websocket packets.
            if request_headers.get("Upgrade") == "websocket":
                return None

            # Strip out search params, get relative path.
            path = path.partition("?")[0]
            relpath = str(Path(path).relative_to("/"))
            if relpath == ".":
                relpath = "index.html"
            source = Path(__file__).absolute().parent / "client" / "build" / relpath

            # Try to read + send over file.
            try:
                return (  # type: ignore
                    http.HTTPStatus.OK,
                    {
                        "content-type": mimetypes.MimeTypes().guess_type(relpath)[0],
                    },
                    source.read_bytes(),
                )
            except FileNotFoundError:
                return (http.HTTPStatus.NOT_FOUND, {}, b"404")  # type: ignore

        for _ in range(500):
            try:
                event_loop.run_until_complete(
                    websockets.server.serve(
                        serve,
                        host,
                        port,
                        compression=None,
                        process_request=viser_http_server if http_server else None,
                    )
                )
                break
            except OSError:  # Port not available.
                port += 1
                continue

        http_url = f"http://{host}:{port}"
        ws_url = f"ws://{host}:{port}"

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("HTTP", f"[link={http_url}]{http_url}[/link]")
        table.add_row("Websocket", f"[link={ws_url}]{ws_url}[/link]")

        rich.print(Panel(table, title="[bold]viser[/bold]", expand=False))

        event_loop.run_forever()


def httpserver(port: int):
    http.server.HTTPServer(
        ("", port), http.server.BaseHTTPRequestHandler
    ).serve_forever()


async def _single_connection_producer(
    websocket: WebSocketServerProtocol, buffer: asyncio.Queue
) -> None:
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
    client_id: ClientId,
    handle_message: Callable[[Message], None],
) -> None:
    """Infinite loop waiting for and then handling incoming messages."""
    while True:
        raw = await websocket.recv()
        assert isinstance(raw, bytes)
        message = Message.deserialize(raw)
        handle_message(message)
