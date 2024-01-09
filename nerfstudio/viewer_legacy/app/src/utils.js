export function split_path(path_str) {
  return path_str.split('/').filter((x) => x.length > 0);
}

export function get_normal_outputs(output_options) {
  // Get a list of normal outputs from the Model
  let normal_options = [];
  if (output_options) {
    // check which outputs have normals
    for (let i = 0; i < output_options.length; i += 1) {
      const output_name = output_options[i];
      if (output_name.includes('normals')) {
        normal_options.push(output_name);
      }
    }
  }
  if (normal_options.length === 0) {
    normal_options = ['none'];
  }
  return normal_options;
}

export function get_normal_methods(output_options) {
  return new Set([...get_normal_outputs(output_options), 'open3d']);
}
