import React, { useEffect, MutableRefObject } from 'react';

import AwaitLock from 'await-lock';
import { pack, unpack } from 'msgpackr';

import { Message } from './ViserMessages';

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  websocketRef: MutableRefObject<WebSocket | null>,
  throttleMilliseconds: number,
) {
  let readyToSend = true;
  let stale = false;
  let latestMessage: Message | null = null;

  function send(message: Message) {
    if (websocketRef.current === null) return;
    latestMessage = message;
    if (readyToSend) {
      websocketRef.current!.send(pack(message));
      stale = false;
      readyToSend = false;

      setTimeout(() => {
        readyToSend = true;
        if (!stale) return;
        send(latestMessage!);
      }, throttleMilliseconds);
    } else {
      stale = true;
    }
  }
  return send;
}
export function ViserWebSocket() {
  const server = 'ws://localhost:8080';
  useEffect(() => {
    // Lock for making sure messages are handled in order.
    const orderLock = new AwaitLock();

    let ws: null | WebSocket = null;
    let done = false;

    function tryConnect(): void {
      if (done) return;

      ws = new WebSocket(server);

      ws.onopen = () => {
        console.log('Viser connected!' + server);
      };

      ws.onclose = () => {
        console.log('Viser disconnected! ' + server);

        // Try to reconnect.
        timeout = setTimeout(tryConnect, 1000);
      };

      ws.onmessage = async (event) => {
        // Reduce websocket backpressure.
        const messagePromise = new Promise<Message>(async (resolve) => {
          resolve(
            unpack(new Uint8Array(await event.data.arrayBuffer())) as Message,
          );
        });

        // Try our best to handle messages in order. If this takes more than 1 second, we give up. :)
        await orderLock.acquireAsync({ timeout: 1000 }).catch(() => {
          console.log('Order lock timed.');
          orderLock.release();
        });
        try {
          handleMessage(await messagePromise);
        } finally {
          orderLock.acquired && orderLock.release();
        }
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      done = true;
      clearTimeout(timeout);
      ws && ws.close();
      clearTimeout(timeout);
    };
  }, [server]);

  return <></>;
}

function handleMessage(message: Message) {
  // TODO: we need to actually handle messages that are received.
  console.log('Handling viser message!');
  console.log(message);
  switch (message.type) {
    // Add a coordinate frame.
    case 'frame': {
      break;
    }
    // Add a point cloud.
    case 'point_cloud': {
      break;
    }
    // Add mesh
    case 'mesh': {
      break;
    }
    // Add a camera frustum.
    case 'camera_frustum': {
      break;
    }
    case 'transform_controls': {
      break;
    }
    case 'transform_controls_set': {
      break;
    }
    // Add a background image.
    case 'background_image': {
      break;
    }
    // Add an image.
    case 'image': {
      break;
    }
    // Remove a scene node by name.
    case 'remove_scene_node': {
      break;
    }
    // Set the visibility of a particular scene node.
    case 'set_scene_node_visibility': {
      break;
    }
    // Reset the entire scene, removing all scene nodes.
    case 'reset_scene': {
      break;
    }
    // Add a GUI input.
    case 'add_gui': {
      break;
    }
    // Set the value of a GUI input.
    case 'gui_set': {
      break;
    }
    // Add a GUI input.
    case 'gui_set_leva_conf': {
      break;
    }
    // Remove a GUI input.
    case 'remove_gui': {
      break;
    }
    default: {
      console.log('Received message did not match any known types:', message);
      break;
    }
  }
}
