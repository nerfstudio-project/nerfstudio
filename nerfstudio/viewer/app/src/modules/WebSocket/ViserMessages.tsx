// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.

// For numpy arrays, we directly serialize the underlying data buffer.
type ArrayBuffer = Uint8Array;

export interface BackgroundImageMessage {
  type: "background_image";
  media_type: 'image/jpeg' | 'image/png';
  base64_data: string;
}
export interface GuiAddMessage {
  type: "add_gui";
  name: string;
  folder_labels: [string];
  leva_conf: any;
}
export interface GuiRemoveMessage {
  type: "remove_gui";
  name: string;
}
export interface GuiUpdateMessage {
  type: "gui_update";
  name: string;
  value: any;
}
export interface GuiSetHiddenMessage {
  type: "gui_set_hidden";
  name: string;
  hidden: boolean;
}
export interface GuiSetValueMessage {
  type: "gui_set";
  name: string;
  value: any;
}
export interface GuiSetLevaConfMessage {
  type: "gui_set_leva_conf";
  name: string;
  leva_conf: any;
}
export interface ResetSceneMessage {
  type: "reset_scene";
}
export interface FilePathInfoMessage {
  type: "path_info";
  config_base_dir: string;
  data_base_dir: string;
  export_path_name: string;
}
export interface CameraMessage {
  type: "camera";
  aspect: number;
  render_aspect: number;
  fov: number;
  matrix: [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];
  camera_type: 'perspective' | 'fisheye' | 'equirectangular';
  is_moving: boolean;
  timestamp: number;
}
export interface SceneBoxMessage {
  type: "scene_box";
  min: [number, number, number];
  max: [number, number, number];
}
export interface DatasetImageMessage {
  type: "dataset_image";
  idx: string;
  json: any;
}
export interface IsTrainingMessage {
  type: "is_training";
  is_training: boolean;
}
export interface CameraPathPayloadMessage {
  type: "camera_path_payload";
  camera_path_filename: string;
  camera_path: any;
}
export interface CameraPathOptionsRequest {
  type: "camera_path_options";
}
export interface CameraPathsMessage {
  type: "camera_paths";
  payload: any;
}
export interface CropParamsMessage {
  type: "crop_params";
  crop_enabled: boolean;
  crop_bg_color: [number, number, number];
  crop_center: [number, number, number];
  crop_scale: [number, number, number];
}

export type Message = 
  | BackgroundImageMessage
  | GuiAddMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetHiddenMessage
  | GuiSetValueMessage
  | GuiSetLevaConfMessage
  | ResetSceneMessage
  | FilePathInfoMessage
  | CameraMessage
  | SceneBoxMessage
  | DatasetImageMessage
  | IsTrainingMessage
  | CameraPathPayloadMessage
  | CameraPathOptionsRequest
  | CameraPathsMessage
  | CropParamsMessage;
