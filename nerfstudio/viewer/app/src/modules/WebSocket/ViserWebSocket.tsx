import React, { useEffect, MutableRefObject, Dispatch } from 'react';

import AwaitLock from 'await-lock';
import { pack, unpack } from 'msgpackr';
import { useDispatch, useStore } from 'react-redux';
import { Store } from 'redux';
import { Message } from './ViserMessages';

const ViserWebSocketContext =
  React.createContext<React.RefObject<WebSocket> | null>(null);

export { ViserWebSocketContext };

/** Send message over websocket. */
export function sendWebsocketMessage(
  websocketRef: MutableRefObject<WebSocket | null>,
  message: Message,
) {
  if (websocketRef.current === null) return;
  websocketRef.current!.send(pack(message));
}

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  websocketRef: MutableRefObject<WebSocket | null>,
  throttleMilliseconds: number,
) {
  let readyToSend = true;
  let stale = false;
  let latestMessage: Message | null = null;

  function send(message: Message) {
    if (websocketRef.current == null) return;
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

function handleMessage(
  message: Message,
  dispatch: Dispatch<any>,
  store: Store,
) {
  // TODO: we need to actually handle messages that are received.
  // console.log('Handling viser message!');
  // console.log(message);
  switch (message.type) {
    // Add a background image.
    case 'background_image': {
      dispatch({
        type: 'write',
        path: 'render_img',
        data: `data:${message.media_type};base64,${message.base64_data}`,
      });
      break;
    }
    // Reset the entire scene, removing all scene nodes.
    case 'reset_scene': {
      break;
    }
    // Add a GUI input.
    case 'add_gui': {
      const curGuiNames = store.getState().custom_gui.guiNames;
      const curGuiConfigFromName =
        store.getState().custom_gui.guiConfigFromName;
      dispatch({
        type: 'write',
        path: 'custom_gui/guiNames',
        data: [...curGuiNames, message.name],
      });
      dispatch({
        type: 'write',
        path: 'custom_gui/guiConfigFromName',
        data: {
          ...curGuiConfigFromName,
          [message.name]: {
            folderLabels: message.folder_labels,
            levaConf: message.leva_conf,
            hidden: false,
          },
        },
      });
      break;
    }
    // Set the hidden state of a GUI input.
    case "gui_set_hidden": {
      const curGuiConfigFromName =
        store.getState().custom_gui.guiConfigFromName;
      const currentConf = curGuiConfigFromName[message.name];
      if (currentConf !== undefined) {
        dispatch({
          type: 'write',
          path: 'custom_gui/guiConfigFromName',
          data: {
            ...curGuiConfigFromName,
            [message.name]: {
              ...currentConf,
              hidden: message.hidden,
            },
          },
        });
      }
      break;
    }
    // Set the value of a GUI input.
    case 'gui_set': {
      break;
    }
    // Set leva conf of element.
    case 'gui_set_leva_conf': {
      const curGuiConfigFromName =
        store.getState().custom_gui.guiConfigFromName;
      const currentConf = curGuiConfigFromName[message.name];
      if (currentConf !== undefined) {
        dispatch({
          type: 'write',
          path: 'custom_gui/guiConfigFromName',
          data: {
            ...curGuiConfigFromName,
            [message.name]: {
              ...currentConf,
              levaConf: message.leva_conf,
            },
          },
        });
      }
      break;
    }
    // Remove a GUI input.
    case 'remove_gui': {
      break;
    }
    // Update scene box.
    case 'scene_box': {
      dispatch({
        type: 'write',
        path: 'sceneState/sceneBox',
        data: message,
      });
      break;
    }
    // Add dataset image.
    case 'dataset_image': {
      const dataset_path = `sceneState/cameras/${message.idx}`;
      dispatch({
        type: 'write',
        path: dataset_path,
        data: message.json,
      });
      break;
    }
    // Set training value.
    case 'is_training': {
      dispatch({
        type: 'write',
        path: 'renderingState/isTraining',
        data: message.is_training,
      });
      break;
    }
    // Populate camera paths.
    case 'camera_paths': {
      dispatch({
        type: 'write',
        path: 'renderingState/all_camera_paths',
        data: message.payload,
      });
      break;
    }
    // Set file path info.
    case 'path_info': {
      dispatch({
        type: 'write',
        path: 'file_path_info/config_base_dir',
        data: message.config_base_dir,
      });
      dispatch({
        type: 'write',
        path: 'file_path_info/data_base_dir',
        data: message.data_base_dir,
      });
      dispatch({
        type: 'write',
        path: 'file_path_info/export_path_name',
        data: message.export_path_name,
      });
      break;
    }
    default: {
      console.log('Received message did not match any known types:', message);
      break;
    }
  }
}

export function ViserWebSocket({ children }: { children: React.ReactNode }) {
  const dispatch = useDispatch();
  const server = 'ws://localhost:8080';
  const store = useStore();

  const ws = React.useRef<WebSocket>();

  useEffect(() => {
    // Lock for making sure messages are handled in order.
    const orderLock = new AwaitLock();

    let done = false;

    function tryConnect(): void {
      if (done) return;

      ws.current = new WebSocket(server);

      ws.current.onopen = () => {
        console.log(`Viser connected! ${server}`);
        dispatch({
          type: 'write',
          path: 'websocketState/isConnected',
          data: true,
        });
      };

      ws.current.onclose = () => {
        console.log(`Viser disconnected! ${server}`);
        dispatch({
          type: 'write',
          path: 'websocketState/isConnected',
          data: false,
        });

        // Try to reconnect.
        timeout = setTimeout(tryConnect, 1000);
      };

      ws.current.onmessage = async (event) => {
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
          handleMessage(await messagePromise, dispatch, store);
        } finally {
          orderLock.acquired && orderLock.release();
        }
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      done = true;
      clearTimeout(timeout);
      ws.current && ws.current.close();
      clearTimeout(timeout);
    };
  }, [server]);

  return (
    <ViserWebSocketContext.Provider value={ws}>
      {children}
    </ViserWebSocketContext.Provider>
  );
}
