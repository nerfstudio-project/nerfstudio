import styled from "@emotion/styled";
import { OrbitControls, Environment } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";

import React, { MutableRefObject, RefObject, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { OrbitControls as OrbitControls_ } from "three-stdlib";

import ControlPanel from "./ControlPanel/ControlPanel";
import LabelRenderer from "./LabelRenderer";
import { SceneNodeThreeObject, useSceneTreeState } from "./SceneTree";

import "./index.css";

import Box from "@mui/material/Box";
import { Euler, PerspectiveCamera, Quaternion } from "three";
import { ViewerCameraMessage } from "./WebsocketMessages";
import {
  FormControlLabel,
  IconButton,
  useMediaQuery,
  Grid,
  Switch,
} from "@mui/material";
import { RemoveCircleRounded, AddCircleRounded } from "@mui/icons-material";
import WebsocketInterface, {
  makeThrottledMessageSender,
} from "./WebsocketInterface";
import { useGuiState, UseGui } from "./ControlPanel/GuiState";
import {
  getServersFromSearchParams,
  truncateSearchParamServers,
} from "./SearchParamsUtils";

interface SynchronizedOrbitControlsProps {
  globalCameras: MutableRefObject<CameraPrimitives>;
  useGui: UseGui;
  websocketRef: MutableRefObject<WebSocket | null>;
}

/** OrbitControls, but synchronized with the server and other panels. */
function SynchronizedOrbitControls(props: SynchronizedOrbitControlsProps) {
  const camera = useThree((state) => state.camera as PerspectiveCamera);
  const orbitRef = React.useRef<OrbitControls_>(null);

  const sendCameraThrottled = makeThrottledMessageSender(
    props.websocketRef,
    20
  );

  // Callback for sending cameras.
  const sendCamera = React.useCallback(() => {
    const three_camera = camera;

    // We put Z up to match the scene tree, and convert threejs camera convention
    // to the OpenCV one.
    const R_threecam_cam = new Quaternion();
    const R_worldfix_world = new Quaternion();
    R_threecam_cam.setFromEuler(new Euler(Math.PI, 0.0, 0.0));
    R_worldfix_world.setFromEuler(new Euler(Math.PI / 2.0, 0.0, 0.0));
    const R_world_camera = R_worldfix_world.clone()
      .multiply(three_camera.quaternion)
      .multiply(R_threecam_cam);

    const message: ViewerCameraMessage = {
      type: "viewer_camera",
      wxyz: [
        R_world_camera.w,
        R_world_camera.x,
        R_world_camera.y,
        R_world_camera.z,
      ],
      position: three_camera.position
        .clone()
        .applyQuaternion(R_worldfix_world)
        .toArray(),
      aspect: three_camera.aspect,
      fov: (three_camera.fov * Math.PI) / 180.0,
    };
    sendCameraThrottled(message);
  }, [camera, sendCameraThrottled]);

  // What do we need to when the camera moves?
  const cameraChangedCallback = React.useCallback(() => {
    const globalCameras = props.globalCameras.current;
    // Match all cameras.
    if (globalCameras.synchronize) {
      props.globalCameras.current!.cameras.forEach((other) => {
        if (camera === other) return;
        other.copy(camera);
      });
      props.globalCameras.current!.orbitRefs.forEach((other) => {
        if (orbitRef === other) return;
        other.current!.target.copy(orbitRef.current!.target);
      });
    }

    // If desired, send our camera via websocket.
    sendCamera();
  }, [props.globalCameras, camera, sendCamera]);

  // Send camera for new connections.
  // We add a small delay to give the server time to add a callback.
  const connected = props.useGui((state) => state.websocketConnected);
  React.useEffect(() => {
    if (!connected) return;
    setTimeout(() => cameraChangedCallback(), 50);
  }, [connected, cameraChangedCallback]);

  React.useEffect(() => {
    const globalCameras = props.globalCameras.current;

    if (globalCameras.synchronize && globalCameras.cameras.length > 0) {
      camera.copy(globalCameras.cameras[0]);
      orbitRef.current!.target.copy(globalCameras.orbitRefs[0].current!.target);
    }

    globalCameras.cameras.push(camera);
    globalCameras.orbitRefs.push(orbitRef);

    window.addEventListener("resize", cameraChangedCallback);

    return () => {
      window.removeEventListener("resize", cameraChangedCallback);

      // Remove ourself from camera list. Since we always add/remove panels
      // from the end, a pop() would actually work as well here in constant
      // time.
      globalCameras.cameras.splice(globalCameras.cameras.indexOf(camera), 1);
      globalCameras.orbitRefs.splice(
        globalCameras.orbitRefs.indexOf(orbitRef),
        1
      );
    };
  }, [cameraChangedCallback, camera, props.globalCameras]);

  return (
    <OrbitControls
      ref={orbitRef}
      minDistance={0.5}
      maxDistance={200.0}
      enableDamping={false}
      onChange={cameraChangedCallback}
      makeDefault
    />
  );
}

interface SingleViewerProps {
  panelKey: number;
  globalCameras: MutableRefObject<CameraPrimitives>;
}

const SingleViewer = React.memo((props: SingleViewerProps) => {
  // Layout and styles.
  const Wrapper = styled(Box)`
    width: 100%;
    height: 100%;
    position: relative;
  `;

  const Viewport = styled(Canvas)`
    position: relative;
    z-index: 0;

    width: 100%;
    height: 100%;
  `;

  // Our 2D label renderer needs access to the div used for rendering.
  const wrapperRef = React.useRef<HTMLDivElement>(null);
  const websocketRef = React.useRef<WebSocket | null>(null);

  // ...
  const servers = new URLSearchParams(window.location.search).getAll("server");
  const initialServer =
    props.panelKey < servers.length
      ? servers[props.panelKey]
      : window.location.href.replace("http://", "ws://");

  // Declare the scene tree state. This returns a zustand store/hook, which we
  // can pass to any children that need state access.
  const useSceneTree = useSceneTreeState();
  const useGui = useGuiState(initialServer);

  // <Stats showPanel={0} className="stats" />
  // <gridHelper args={[10.0, 10]} />
  return (
    <Wrapper ref={wrapperRef}>
      <WebsocketInterface
        panelKey={props.panelKey}
        useSceneTree={useSceneTree}
        useGui={useGui}
        websocketRef={websocketRef}
        wrapperRef={wrapperRef}
      />
      <ControlPanel
        useSceneTree={useSceneTree}
        useGui={useGui}
        websocketRef={websocketRef}
        wrapperRef={wrapperRef}
      />
      <Viewport camera={{ position: [3.0, 3.0, -3.0] }}>
        <LabelRenderer wrapperRef={wrapperRef} />
        <SynchronizedOrbitControls
          websocketRef={websocketRef}
          useGui={useGui}
          globalCameras={props.globalCameras}
        />
        <SceneNodeThreeObject name="" useSceneTree={useSceneTree} />
        <Environment preset="city" blur={1} />
      </Viewport>
    </Wrapper>
  );
});

interface CameraPrimitives {
  synchronize: boolean;
  cameras: PerspectiveCamera[];
  orbitRefs: RefObject<OrbitControls_>[];
}

function Root() {
  const globalCameras = useRef<CameraPrimitives>({
    synchronize: false,
    cameras: [],
    orbitRefs: [],
  });
  const [panelCount, setPanelCount] = useState(
    Math.max(1, getServersFromSearchParams().length)
  );
  const isPortrait = useMediaQuery("(orientation: portrait)");

  return (
    <Box
      component="div"
      sx={{
        width: "100%",
        height: "100%",
        position: "relative",
        boxSizing: "border-box",
        paddingBottom: "2.5em",
      }}
    >
      <PanelController
        panelCount={panelCount}
        setPanelCount={setPanelCount}
        globalCameras={globalCameras}
      />
      {Array.from({ length: panelCount }, (_, i) => {
        return (
          <Box
            component="div"
            key={"box-" + i.toString()}
            sx={{
              ...(isPortrait
                ? {
                    width: "100%",
                    height: (100.0 / panelCount).toString() + "%",
                  }
                : {
                    height: "100%",
                    float: "left",
                    width: (100.0 / panelCount).toString() + "%",
                  }),
              boxSizing: "border-box;",
              "&:not(:last-child)": {
                borderRight: isPortrait ? null : "1px solid",
                borderBottom: isPortrait ? "1px solid" : null,
                borderColor: "divider",
              },
            }}
          >
            <SingleViewer panelKey={i} globalCameras={globalCameras} />
          </Box>
        );
      })}
    </Box>
  );
}

interface PanelControllerProps {
  panelCount: number;
  setPanelCount: React.Dispatch<React.SetStateAction<number>>;
  globalCameras: MutableRefObject<CameraPrimitives>;
}

function PanelController(props: PanelControllerProps) {
  return (
    <Box
      component="div"
      sx={{
        position: "fixed",
        bottom: "0",
        width: "100%",
        height: "2.5em",
        zIndex: "1000",
        backgroundColor: "rgba(255, 255, 255, 0.85)",
        borderTop: "1px solid",
        borderTopColor: "divider",
      }}
    >
      <Grid sx={{ float: "right" }}>
        <IconButton
          onClick={() => {
            props.setPanelCount(props.panelCount + 1);
          }}
        >
          <AddCircleRounded />
        </IconButton>
        <IconButton
          disabled={props.panelCount === 1}
          onClick={() => {
            if (props.panelCount === 1) return;
            truncateSearchParamServers(props.panelCount - 1);
            props.setPanelCount(props.panelCount - 1);
          }}
        >
          <RemoveCircleRounded />
        </IconButton>
        <FormControlLabel
          control={<Switch />}
          label="Sync Cameras"
          defaultChecked={props.globalCameras.current.synchronize}
          onChange={(_event, checked) => {
            props.globalCameras.current.synchronize = checked;
          }}
          sx={{ pl: 1 }}
          disabled={props.panelCount === 1}
        />
      </Grid>
    </Box>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Root />);
