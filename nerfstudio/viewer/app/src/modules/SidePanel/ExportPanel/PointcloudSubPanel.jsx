import * as React from 'react';
import { useControls, useStoreContext } from 'leva';
import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';

import { Button } from '@mui/material';

export default function PointcloudSubPanel(props) {
  const update_box_center = props.update_box_center;
  const update_box_scale = props.update_box_scale;
  const store = useStoreContext();

  const dispatch = useDispatch();

  // redux store state
  const config_base_dir = useSelector(
    (state) => state.file_path_info.config_base_dir,
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

  const controlValues = useControls(
    {
      numPoints: { label: 'Number of Points', value: 1000000, min: 1 },
      removeOutliers: { label: 'Remove Outliers', value: true },
      estimateNormals: { label: 'Estimate Normals', value: false },
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
      outputDir: { label: 'Output Directory', value: 'exports/pcd/' },
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
