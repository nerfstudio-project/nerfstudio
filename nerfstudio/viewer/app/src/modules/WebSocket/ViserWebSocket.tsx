import React, { useEffect, MutableRefObject, Dispatch } from 'react';

import AwaitLock from 'await-lock';
import { pack, unpack } from 'msgpackr';
import { useDispatch, useStore, useSelector } from 'react-redux';
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
  switch (message.type) {
    // Add a background image.
    case 'BackgroundImageMessage': {
      document
        .getElementById('background-image')!
        .setAttribute(
          'src',
          `data:${message.media_type};base64,${message.base64_data}`,
        );
      break;
    }
    // Add a GUI input.
    case 'GuiAddMessage': {
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
    case 'GuiSetHiddenMessage': {
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
    case 'GuiSetValueMessage': {
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
              value: message.value,
            },
          },
        });
        //  To propagate change to the leva element, need to add to the guiSetQueue
        const curSetQueue = store.getState().custom_gui.guiSetQueue;
        dispatch({
          type: 'write',
          path: 'custom_gui/guiSetQueue',
          data: {
            ...curSetQueue,
            [message.name]: message.value,
          },
        });
      }
      break;
    }
    // Set leva conf of element.
    case 'GuiSetLevaConfMessage': {
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
    case 'GuiRemoveMessage': {
      // TODO: not implemented.
      break;
    }
    // Update scene box.
    case 'SceneBoxMessage': {
      dispatch({
        type: 'write',
        path: 'sceneState/sceneBox',
        data: message,
      });
      break;
    }
    // Add dataset image.
    case 'DatasetImageMessage': {
      const dataset_path = `sceneState/cameras/${message.idx}`;
      dispatch({
        type: 'write',
        path: dataset_path,
        data: message.json,
      });
      break;
    }
    // Set training value.
    case 'TrainingStateMessage': {
      dispatch({
        type: 'write',
        path: 'renderingState/training_state',
        data: message.training_state,
      });
      break;
    }
    // Populate camera paths.
    case 'CameraPathsMessage': {
      dispatch({
        type: 'write',
        path: 'all_camera_paths',
        data: message.payload,
      });
      break;
    }
    // Set file path info.
    case 'FilePathInfoMessage': {
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
    // Set crop parameters
    case 'CropParamsMessage': {
      dispatch({
        type: 'write',
        path: 'renderingState/crop_enabled',
        data: message.crop_enabled,
      });
      dispatch({
        type: 'write',
        path: 'renderingState/crop_bg_color',
        data: message.crop_bg_color,
      });
      dispatch({
        type: 'write',
        path: 'renderingState/crop_scale',
        data: message.crop_scale,
      });
      dispatch({
        type: 'write',
        path: 'renderingState/crop_center',
        data: message.crop_center,
      });
      break;
    }
    // Handle status messages.
    case 'StatusMessage': {
      dispatch({
        type: 'write',
        path: 'renderingState/eval_res',
        data: message.eval_res,
      });
      dispatch({
        type: 'write',
        path: 'renderingState/step',
        data: message.step,
      });
      break;
    }
    // Handle time conditioning messages.
    case 'UseTimeConditioningMessage': {
      console.log('HERERERE');
      dispatch({
        type: 'write',
        path: 'renderingState/use_time_conditioning',
        data: true,
      });
      break;
    }
    case 'TimeConditionMessage': {
      dispatch({
        type: 'write',
        path: 'renderingState/time_condition',
        data: message.time,
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
  const store = useStore();

  const ws = React.useRef<WebSocket | null>(null);

  const websocket_url = useSelector(
    (state) => state.websocketState.websocket_url,
  );

  useEffect(() => {
    // Lock for making sure messages are handled in order.
    const orderLock = new AwaitLock();

    let done = false;

    function tryConnect(): void {
      if (done) return;

      ws.current = new WebSocket(websocket_url);

      const connecting_timeout = setTimeout(() => {
        if (
          ws.current != null &&
          ws.current.readyState === WebSocket.CONNECTING
        ) {
          ws.current.close();
          console.log('WebSocket connection timed out');
        }
      }, 5000); // timeout after 5 seconds

      ws.current.onopen = () => {
        console.log(`Viser connected! ${websocket_url}`);
        clearTimeout(connecting_timeout);
        dispatch({
          type: 'write',
          path: 'websocketState/isConnected',
          data: true,
        });
      };

      ws.current.onclose = () => {
        console.log(`Viser disconnected! ${websocket_url}`);
        dispatch({
          type: 'write',
          path: 'websocketState/isConnected',
          data: false,
        });

        // Try to reconnect.
        // eslint-disable-next-line no-use-before-define
        timeout = setTimeout(tryConnect, 1000);
      };

      ws.current.onmessage = async (event) => {
        // Reduce websocket backpressure.
        try {
          const message = (await unpack(
            new Uint8Array(await event.data.arrayBuffer()),
          )) as Message;
          await orderLock.acquireAsync({ timeout: 1000 });
          handleMessage(message, dispatch, store);
        } catch (error) {
          console.error(`Error handling message: ${error}`);
        } finally {
          if (orderLock.acquired) {
            orderLock.release();
          }
        }
      };

      // add websocket error handling
      ws.current.onerror = (err) => {
        console.log('Websocket error: ', err);
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      clearTimeout(timeout);
      if (ws.current) {
        done = true;
        ws.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [websocket_url]);

  return (
    <ViserWebSocketContext.Provider value={ws}>
      {children}
    </ViserWebSocketContext.Provider>
  );
}
