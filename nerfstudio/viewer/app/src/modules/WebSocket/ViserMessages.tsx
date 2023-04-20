// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.
interface BackgroundImageMessage {
  type: 'BackgroundImageMessage';
  media_type: 'image/jpeg' | 'image/png';
  base64_data: string;
}
interface GuiAddMessage {
  type: 'GuiAddMessage';
  name: string;
  folder_labels: [string];
  leva_conf: any;
}
interface GuiRemoveMessage {
  type: 'GuiRemoveMessage';
  name: string;
}
interface GuiUpdateMessage {
  type: 'GuiUpdateMessage';
  name: string;
  value: any;
}
interface GuiSetHiddenMessage {
  type: 'GuiSetHiddenMessage';
  name: string;
  hidden: boolean;
}
interface GuiSetValueMessage {
  type: 'GuiSetValueMessage';
  name: string;
  value: any;
}
interface GuiSetLevaConfMessage {
  type: 'GuiSetLevaConfMessage';
  name: string;
  leva_conf: any;
}
interface FilePathInfoMessage {
  type: 'FilePathInfoMessage';
  config_base_dir: string;
  data_base_dir: string;
  export_path_name: string;
}
interface CameraMessage {
  type: 'CameraMessage';
  aspect: number;
  render_aspect: number;
  fov: number;
  matrix: [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  camera_type: 'perspective' | 'fisheye' | 'equirectangular';
  is_moving: boolean;
  timestamp: number;
}
interface SceneBoxMessage {
  type: 'SceneBoxMessage';
  min: [number, number, number];
  max: [number, number, number];
}
interface DatasetImageMessage {
  type: 'DatasetImageMessage';
  idx: string;
  json: any;
}
interface TrainingStateMessage {
  type: 'TrainingStateMessage';
  training_state: 'training' | 'paused' | 'completed';
}
interface CameraPathPayloadMessage {
  type: 'CameraPathPayloadMessage';
  camera_path_filename: string;
  camera_path: any;
}
interface CameraPathOptionsRequest {
  type: 'CameraPathOptionsRequest';
}
interface CameraPathsMessage {
  type: 'CameraPathsMessage';
  payload: any;
}
interface CropParamsMessage {
  type: 'CropParamsMessage';
  crop_enabled: boolean;
  crop_bg_color: [number, number, number];
  crop_center: [number, number, number];
  crop_scale: [number, number, number];
}
interface StatusMessage {
  type: 'StatusMessage';
  eval_res: string;
  step: number;
}
interface SaveCheckpointMessage {
  type: 'SaveCheckpointMessage';
}
interface UseTimeConditioningMessage {
  type: 'UseTimeConditioningMessage';
}
interface TimeConditionMessage {
  type: 'TimeConditionMessage';
  time: number;
}

export type Message =
  | BackgroundImageMessage
  | GuiAddMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetHiddenMessage
  | GuiSetValueMessage
  | GuiSetLevaConfMessage
  | FilePathInfoMessage
  | CameraMessage
  | SceneBoxMessage
  | DatasetImageMessage
  | TrainingStateMessage
  | CameraPathPayloadMessage
  | CameraPathOptionsRequest
  | CameraPathsMessage
  | CropParamsMessage
  | StatusMessage
  | SaveCheckpointMessage
  | UseTimeConditioningMessage
  | TimeConditionMessage;
