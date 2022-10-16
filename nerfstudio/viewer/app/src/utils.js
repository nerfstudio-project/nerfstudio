export function split_path(path_str) {
  return path_str.split('/').filter((x) => x.length > 0);
}
export const fov_to_focal = (sensorSize, val) =>
  Math.round(sensorSize / 2 / Math.tan((val * (Math.PI / 180)) / 2));
export const focal_to_fov = (sensorSize, val) =>
  Math.round((180 / Math.PI) * 2 * Math.atan(sensorSize / 2 / val));
