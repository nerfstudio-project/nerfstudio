export function split_path(path_str) {
  return path_str.split('/').filter((x) => x.length > 0);
}
