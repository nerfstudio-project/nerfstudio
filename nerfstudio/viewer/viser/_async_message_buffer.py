import asyncio
import dataclasses
from asyncio.events import AbstractEventLoop
from typing import Dict

from ._messages import (
    BackgroundImageMessage,
    Message,
    RemoveSceneNodeMessage,
    ResetSceneMessage,
)


@dataclasses.dataclass
class AsyncMessageBuffer:
    """Async iterable for keeping a persistent buffer of messages.

    Uses heuristics on message names to automatically cull out redundant messages."""

    event_loop: AbstractEventLoop
    message_counter: int = 0
    message_from_id: Dict[int, Message] = dataclasses.field(default_factory=dict)
    id_from_name: Dict[str, int] = dataclasses.field(default_factory=dict)
    message_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    def push(self, message: Message) -> None:
        """Push a new message to our buffer, and remove old redundant ones."""

        # If we're resetting the scene, we don't need any of the prior messages.
        if isinstance(message, ResetSceneMessage):
            self.message_from_id.clear()
            self.id_from_name.clear()

        # Add message to buffer.
        new_message_id = self.message_counter
        self.message_from_id[new_message_id] = message
        self.message_counter += 1

        # All messages that modify scene nodes have a name field.
        node_name = getattr(message, "name", None)

        if isinstance(message, BackgroundImageMessage):
            node_name = "__viser_background_image__"

        if node_name is not None:
            # TODO: hack to prevent undesirable message culling. We should revisit
            # this.
            node_name = str(type(message)) + node_name

            # If an existing message with the same name already exists in our buffer, we
            # don't need the old one anymore. :-)
            if node_name is not None and node_name in self.id_from_name:
                old_message_id = self.id_from_name.pop(node_name)
                self.message_from_id.pop(old_message_id)

            # If we're removing a scene node, remove children as well.
            #
            # TODO: this currently does a linear pass over all existing messages. We
            # could easily optimize this.
            if isinstance(message, RemoveSceneNodeMessage) and node_name is not None:
                remove_list = []
                for name, id in self.id_from_name.items():
                    if name.startswith(node_name):
                        remove_list.append((name, id))
                for name, id in remove_list:
                    self.id_from_name.pop(name)
                    self.message_from_id.pop(id)
            self.id_from_name[node_name] = new_message_id

        # Notify consumers that a new message is available.
        self.event_loop.call_soon_threadsafe(self.message_event.set)

    async def __aiter__(self):
        """Async iterator over messages. Loops infinitely, and waits when no messages
        are available."""
        # Wait for a first message to arrive.
        if len(self.message_from_id) == 0:
            await self.message_event.wait()

        last_sent_id = -1
        while True:
            # Wait until there are new messages available.
            # TODO: there are potential race conditions here.
            most_recent_message_id = next(reversed(self.message_from_id))
            while last_sent_id >= most_recent_message_id:
                await self.message_event.wait()
                most_recent_message_id = next(reversed(self.message_from_id))

            # Try to yield the next message ID. Note that messages can be culled before
            # they're sent.
            last_sent_id += 1
            message = self.message_from_id.get(last_sent_id, None)
            if message is not None:
                yield message
                # TODO: it's likely OK for now, but feels sketchy to be sharing the same
                # message event across all consumers.
                self.event_loop.call_soon_threadsafe(self.message_event.clear)

                # Small sleep: this is needed when (a) messages are being queued faster than
                # we can send them and (b) when there are multiple clients.
                await asyncio.sleep(1e-4)
