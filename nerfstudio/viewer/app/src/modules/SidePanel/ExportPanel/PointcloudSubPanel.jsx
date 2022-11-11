import * as React from 'react';
import * as THREE from 'three';
import { useControls, useStoreContext } from 'leva';
import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';

import { Button } from '@mui/material';

const PCD_CLIPPING_BOX_NAME = 'PCD Clipping Box';

export default function PointcloudSubPanel(props) {
  const sceneTree = props.sceneTree;
  const showExportBox = props.showExportBox;
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

  const config_filename = `${config_base_dir}/config.yml`;

  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({
    color: 0xffd369,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  });
  const cube = new THREE.Mesh(geometry, material);
  sceneTree.set_object_from_path([PCD_CLIPPING_BOX_NAME], cube);
  cube.visible = showExportBox && clippingEnabled;

  const update_box_center = (value) => {
    const box = sceneTree.find_object_no_create([PCD_CLIPPING_BOX_NAME]);
    box.position.set(value[0], value[1], value[2]);
  };

  const update_box_scale = (value) => {
    const box = sceneTree.find_object_no_create([PCD_CLIPPING_BOX_NAME]);
    box.scale.set(value[0], value[1], value[2]);
  };

  const controlValues = useControls(
    {
      numPoints: { label: 'Number of Points', value: 1000000, min: 1 },
      removeOutliers: { label: 'Remove Outliers', value: true },
      estimateNormals: { label: 'Estimate Normals', value: false },
      useBoundingBox: {
        label: 'Crop',
        value: clippingEnabled,
        onChange: (value) => {
          const box = sceneTree.find_object_no_create([PCD_CLIPPING_BOX_NAME]);
          box.visible = value;
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
      outputDir: { label: 'Output Directory', value: 'pcd/' },
    },
    { store },
  );

  const getPythonBool = (boolean) => {
    return boolean ? 'True' : 'False';
  };

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

  const cmd =
    `ns-export pointcloud` +
    ` --load-config ${config_filename}` +
    ` --output-dir ${controlValues.outputDir}` +
    ` --num-points ${controlValues.numPoints}` +
    ` --remove-outliers ${getPythonBool(controlValues.removeOutliers)}` +
    ` --estimate-normals ${getPythonBool(controlValues.estimateNormals)}` +
    ` --use-bounding-box ${getPythonBool(clippingEnabled)}` +
    ` --bounding-box-min ${bbox_min[0]} ${bbox_min[1]} ${bbox_min[2]}` +
    ` --bounding-box-max ${bbox_max[0]} ${bbox_max[1]} ${bbox_max[2]}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(cmd);
  };

  useEffect(() => {
    update_box_center(clippingCenter);
    update_box_scale(clippingScale);
  });

  return (
    <div className="PointcloudPanel">
      <div className="ExportModal-text">
        Run the following command in a terminal to export the pointcloud:
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
    </div>
  );
}
