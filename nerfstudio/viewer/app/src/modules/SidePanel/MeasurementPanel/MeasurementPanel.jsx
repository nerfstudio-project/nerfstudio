import {
  LevaPanel,
  LevaStoreProvider,
  useControls,
  useCreateStore,
  useStoreContext,
} from 'leva';
import * as React from 'react';
import { useDispatch, useSelector, useStore } from 'react-redux';
import * as THREE from 'three';
import LevaTheme from '../../../themes/leva_theme.json';
import { Button, TextField } from '@mui/material';
import SceneNode from '../../../SceneNode';

interface MeasurementPanelProps {
  sceneTree: SceneNode;
}

function MeasurementPropsPanel(props) {
  const store = useStoreContext();
  const dispatch = useDispatch();

  // const scaleFactorValue = useSelector((state) => state.measState.scaleFactor);
  const fontSizeValue = useSelector((state) => state.measState.fontSize);
  const colorValue = useSelector((state) => state.measState.color);
  const markerRadiusValue = useSelector(
    (state) => state.measState.markerRadius,
  );
  const lineWidthValue = useSelector((state) => state.measState.lineWidth);

  // const setScaleFactor = (value) => {
  //   dispatch({
  //     type: 'write',
  //     path: 'measState/scaleFactor',
  //     data: value,
  //   });
  // };

  const setFontSize = (value) => {
    dispatch({
      type: 'write',
      path: 'measState/fontSize',
      data: value,
    });
  };

  const setColor = (value) => {
    dispatch({
      type: 'write',
      path: 'measState/color',
      data: value,
    });
  };

  const setMarkerRadius = (value) => {
    dispatch({
      type: 'write',
      path: 'measState/markerRadius',
      data: value,
    });
  };

  const setLineWidth = (value) => {
    dispatch({
      type: 'write',
      path: 'measState/lineWidth',
      data: value,
    });
  };

  const [, setControls] = useControls(
    () => ({
      // scaleFactor: {
      //   label: 'Scale',
      //   value: scaleFactorValue,
      //   onChange: (v) => {
      //     setScaleFactor(v);
      //   },
      // },
      fontSize: {
        label: 'Font Size',
        value: fontSizeValue,
        onChange: (v) => {
          setFontSize(v);
        },
      },
      color: {
        label: 'Color',
        value: colorValue,
        onChange: (v) => {
          setColor(v);
        },
      },
      markerRadius: {
        label: 'Marker Radius',
        value: markerRadiusValue,
        onChange: (v) => {
          setMarkerRadius(v);
        },
      },
      lineWidth: {
        label: 'Line Width',
        value: lineWidthValue,
        onChange: (v) => {
          setLineWidth(v);
        },
      },
    }),
    { store },
  );

  // setControls({ scaleFactor: scaleFactorValue });
  setControls({ fontSize: fontSizeValue });
  setControls({ color: colorValue });
  setControls({ markerRadius: markerRadiusValue });
  setControls({ lineWidth: lineWidthValue });

  return null;
}

export default function MeasurementPanel(props: MeasurementPanelProps) {
  const sceneTree = props.sceneTree;
  const dispatch = useDispatch();

  const measPropStore = useCreateStore();

  const [isMeasuring, setIsMeasuring] = React.useState(false);

  const handleToggleMeas = React.useCallback(() => {
    const enabled = !isMeasuring;
    setIsMeasuring(enabled);

    dispatch({
      type: 'write',
      path: 'measState/enabled',
      data: enabled,
    });
  }, [isMeasuring]);

  const handleClear = React.useCallback(() => {
    sceneTree.delete(['Measurement']);
  }, [sceneTree]);

  return (
    <div className="MeasPanel">
      <LevaPanel
        store={measPropStore}
        className="Leva-panel"
        theme={LevaTheme}
        titleBar={false}
        fill
        flat
      />
      <div className="MeasPanel-props">
        <LevaStoreProvider store={measPropStore}>
          <MeasurementPropsPanel />
        </LevaStoreProvider>
      </div>
      <div className="MeasPanel-controls">
        <Button
          sx={{}}
          variant="outlined"
          size="medium"
          onClick={handleToggleMeas}
        >
          {isMeasuring ? 'Stop' : 'Start'}
        </Button>
        <Button sx={{}} variant="outlined" size="medium" onClick={handleClear}>
          Clear
        </Button>
      </div>
    </div>
  );
}
