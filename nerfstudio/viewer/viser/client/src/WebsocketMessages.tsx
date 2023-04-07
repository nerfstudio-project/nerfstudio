// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.

// For numpy arrays, we directly serialize the underlying data buffer.
type ArrayBuffer = Uint8Array;

export interface ViewerCameraMessage {
  type: "viewer_camera";
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
  aspect: number;
}
export interface CameraFrustumMessage {
  type: "camera_frustum";
  name: string;
  fov: number;
  aspect: number;
  scale: number;
  color: number;
}
export interface FrameMessage {
  type: "frame";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
  show_axes: boolean;
  axes_length: number;
  axes_radius: number;
}
export interface PointCloudMessage {
  type: "point_cloud";
  name: string;
  position: ArrayBuffer;
  color: ArrayBuffer;
  point_size: number;
}
export interface MeshMessage {
  type: "mesh";
  name: string;
  vertices: ArrayBuffer;
  faces: ArrayBuffer;
  color: number;
  wireframe: boolean;
}
export interface TransformControlsMessage {
  type: "transform_controls";
  name: string;
  scale: number;
  line_width: number;
  fixed: boolean;
  auto_transform: boolean;
  active_axes: [boolean, boolean, boolean];
  disable_axes: boolean;
  disable_sliders: boolean;
  disable_rotations: boolean;
  translation_limits: [[number, number], [number, number], [number, number]];
  rotation_limits: [[number, number], [number, number], [number, number]];
  depth_test: boolean;
  opacity: number;
}
export interface TransformControlsSetMessage {
  type: "transform_controls_set";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
}
export interface TransformControlsUpdateMessage {
  type: "transform_controls_update";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
}
export interface BackgroundImageMessage {
  type: "background_image";
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
}
export interface ImageMessage {
  type: "image";
  name: string;
  media_type: "image/jpeg" | "image/png";
  base64_data: string;
  render_width: number;
  render_height: number;
}
export interface RemoveSceneNodeMessage {
  type: "remove_scene_node";
  name: string;
}
export interface SetSceneNodeVisibilityMessage {
  type: "set_scene_node_visibility";
  name: string;
  visible: boolean;
}
export interface ResetSceneMessage {
  type: "reset_scene";
}
export interface GuiAddMessage {
  type: "add_gui";
  name: string;
  folder_labels: string[];
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

export type Message =
  | ViewerCameraMessage
  | CameraFrustumMessage
  | FrameMessage
  | PointCloudMessage
  | MeshMessage
  | TransformControlsMessage
  | TransformControlsSetMessage
  | TransformControlsUpdateMessage
  | BackgroundImageMessage
  | ImageMessage
  | RemoveSceneNodeMessage
  | SetSceneNodeVisibilityMessage
  | ResetSceneMessage
  | GuiAddMessage
  | GuiRemoveMessage
  | GuiUpdateMessage
  | GuiSetValueMessage
  | GuiSetLevaConfMessage;
