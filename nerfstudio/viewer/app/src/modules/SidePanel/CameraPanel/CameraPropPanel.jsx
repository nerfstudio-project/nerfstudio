import { useControls, useStoreContext } from 'leva';
import { useDispatch, useSelector } from 'react-redux';

export default function CameraPropPanel(props) {
  const seconds = props.seconds;
  const set_seconds = props.set_seconds;
  const fps = props.fps;
  const set_fps = props.set_fps;

  // redux store state
  const store = useStoreContext();

  const dispatch = useDispatch();

  // redux store state
  const render_height = useSelector(
    (state) => state.renderingState.render_height,
  );
  const render_width = useSelector(
    (state) => state.renderingState.render_width,
  );
  const camera_type = useSelector((state) => state.renderingState.camera_type);

  const export_path = useSelector((state) => state.renderingState.export_path);

  const setExportPath = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/export_path',
      data: value,
    });
  };

  const setResolution = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/render_width',
      data: value.width,
    });
    dispatch({
      type: 'write',
      path: 'renderingState/render_height',
      data: value.height,
    });
  };

  const setCameraType = (value) => {
    dispatch({
      type: 'write',
      path: 'renderingState/camera_type',
      data: value,
    });
  };

  const [, setControls] = useControls(
    () => ({
      path: {
        label: 'Export Name',
        value: export_path,
        onChange: (v) => {
          const valid_filename_reg = /^([a-z]|[A-Z]|[0-9]|-|_)+$/g;
          if(!valid_filename_reg.test(v)){
            alert("Please only use letters, numbers, and hyphens");
          }
          else {
            setExportPath(v);
          }
        },
        
      },
      camera_resolution: {
        label: 'Resolution',
        value: { width: render_width, height: render_height },
        joystick: false,
        onChange: (v) => {
          setResolution(v);
        },
      },
      video_duration: {
        label: 'Duration (Sec)',
        value: seconds,
        min: 0.1,
        step: 0.1,
        onChange: (v) => {
          set_seconds(v);
        },
      },
      video_fps: {
        label: 'Framerate (FPS)',
        value: fps,
        min: 0.1,
        onChange: (v) => {
          set_fps(v);
        },
      },
      camera_type_selector: {
        label: 'Camera Type',
        value: camera_type,
        options: {
          Perspective: 'perspective',
          Fisheye: 'fisheye',
          Equirectangular: 'equirectangular',
        },
        onChange: (v) => {
          setCameraType(v);
        },
      },
    }),
    { store },
  );

  setControls({path: export_path});
  setControls({ video_fps: fps });
  setControls({ video_duration: seconds });
  setControls({
    camera_resolution: { width: render_width, height: render_height },
  });
  setControls({ camera_type_selector: camera_type });

  return null;
}
