import * as React from 'react';
import { useControls, useStoreContext } from 'leva';
import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';

import { Button } from '@mui/material';

function get_normal_outputs(output_options) {
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

export default function MeshSubPanel(props) {
  const update_box_center = props.update_box_center;
  const update_box_scale = props.update_box_scale;
  const store = useStoreContext();

  const dispatch = useDispatch();

  // redux store state
  const config_base_dir = useSelector(
    (state) => state.renderingState.config_base_dir,
  );
  const clippingEnabled = useSelector(
    (state) => state.renderingState.clipping_enabled,
  );
  const clippingCenter = useSelector(
    (state) => state.renderingState.clipping_center,
  );
  const clippingScale = useSelector(
    (state) => state.renderingState.clipping_box_scale,
  );
  const output_options = useSelector(
    (state) => state.renderingState.output_options,
  );
  const normal_options = get_normal_outputs(output_options);
  const mesh_method_options = ['poisson', 'tsdf'];

  const config_filename = `${config_base_dir}/config.yml`;

  const controlValues = useControls(
    {
      mesh_method_options: {
        label: 'Mesh Method Options',
        options: {
          'Poisson (best)': 'poisson',
          TSDF: 'tsdf',
        },
        value: mesh_method_options[0],
      },
      normal_options: {
        label: 'Normal Options',
        options: [...new Set(normal_options)],
        value: normal_options[0],
        render: (get) => get('mesh_method_options') === 'poisson',
      },
      numFaces: { label: 'Number of Faces', value: 50000, min: 1 },
      textureResolution: { label: 'Texture Resolution', value: 2048, min: 1 },
      numPoints: {
        label: 'Number of Points',
        value: 1000000,
        min: 1,
        render: (get) => get('mesh_method_options') === 'poisson',
      },
      removeOutliers: {
        label: 'Remove Outliers',
        value: true,
        render: (get) => get('mesh_method_options') === 'poisson',
      },
      useBoundingBox: {
        label: 'Crop',
        value: clippingEnabled,
        onChange: (value) => {
          dispatch({
            type: 'write',
            path: 'renderingState/clipping_enabled',
            data: value,
          });
        },
      },
      center: {
        label: '|  Center',
        value: clippingCenter,
        render: (get) => get('useBoundingBox'),
        onChange: (value) => {
          update_box_center(value);
        },
        onEditEnd: (value) => {
          dispatch({
            type: 'write',
            path: 'renderingState/clipping_center',
            data: value,
          });
        },
      },
      scale: {
        label: '|  Scale',
        value: clippingScale,
        min: 0.01,
        render: (get) => get('useBoundingBox'),
        onChange: (value) => {
          update_box_scale(value);
        },
        onEditEnd: (value) => {
          dispatch({
            type: 'write',
            path: 'renderingState/clipping_box_scale',
            data: value,
          });
        },
      },
      outputDir: { label: 'Output Directory', value: 'exports/mesh/' },
    },
    { store },
  );

  const getPythonBool = (boolean) => {
    return boolean ? 'True' : 'False';
  };

  console.log(controlValues.outputDir);
  console.log(getPythonBool(controlValues.removeOutliers));

  const bbox_min = [
    clippingCenter[0] - clippingScale[0] / 2,
    clippingCenter[1] - clippingScale[1] / 2,
    clippingCenter[2] - clippingScale[2] / 2,
  ];
  const bbox_max = [
    clippingCenter[0] + clippingScale[0] / 2,
    clippingCenter[1] + clippingScale[1] / 2,
    clippingCenter[2] + clippingScale[2] / 2,
  ];

  const mesh_method_choice = controlValues.mesh_method_options;
  console.log(mesh_method_choice);
  let cmd = '';
  if (mesh_method_choice === 'tsdf') {
    cmd =
      `ns-export ${mesh_method_choice}` +
      ` --load-config ${config_filename}` +
      ` --output-dir ${controlValues.outputDir}` +
      ` --target-num-faces ${controlValues.numFaces}` +
      ` --num-pixels-per-side ${controlValues.textureResolution}` +
      ` --use-bounding-box ${getPythonBool(clippingEnabled)}` +
      ` --bounding-box-min ${bbox_min[0]} ${bbox_min[1]} ${bbox_min[2]}` +
      ` --bounding-box-max ${bbox_max[0]} ${bbox_max[1]} ${bbox_max[2]}`;
  } else if (mesh_method_choice === 'poisson') {
    cmd =
      `ns-export ${mesh_method_choice}` +
      ` --load-config ${config_filename}` +
      ` --output-dir ${controlValues.outputDir}` +
      ` --target-num-faces ${controlValues.numFaces}` +
      ` --num-pixels-per-side ${controlValues.textureResolution}` +
      ` --normal-output-name ${controlValues.normal_options}` +
      ` --num-points ${controlValues.numPoints}` +
      ` --remove-outliers ${getPythonBool(controlValues.removeOutliers)}` +
      ` --use-bounding-box ${getPythonBool(clippingEnabled)}` +
      ` --bounding-box-min ${bbox_min[0]} ${bbox_min[1]} ${bbox_min[2]}` +
      ` --bounding-box-max ${bbox_max[0]} ${bbox_max[1]} ${bbox_max[2]}`;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(cmd);
  };

  useEffect(() => {
    update_box_center(clippingCenter);
    update_box_scale(clippingScale);
  });

  let inner_html;
  if (
    mesh_method_choice === 'poisson' &&
    controlValues.normal_options === 'none'
  ) {
    // check if normals aren't available when using poisson meshing
    inner_html = (
      <div className="ExportModal-text">
        You need to train a model with normals to use poisson meshing. Try tsdf
        or retrain the model with normals. You can use nerfacto with <br/>
        <b>--pipeline.model.predict-normals True</b>
      </div>
    );
  } else {
    // if valid, provide the command needed for creating the mesh
    inner_html = (
      <>
        <div className="ExportModal-text">
          Run the following command in a terminal to export the mesh:
        </div>
        <div className="ExportModal-code">{cmd}</div>
        <div className="ExportModal-button" style={{ textAlign: 'center' }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<ContentCopyRoundedIcon />}
            onClick={handleCopy}
          >
            Copy Command
          </Button>
        </div>
      </>
    );
  }

  return <div className="MeshPanel">{inner_html}</div>;
}
