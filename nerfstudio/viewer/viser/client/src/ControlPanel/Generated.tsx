import { GuiUpdateMessage } from "../WebsocketMessages";
import { button, folder, LevaPanel, useControls, useCreateStore } from "leva";
import { LevaCustomTheme } from "leva/dist/declarations/src/styles";
import { UseGui } from "./GuiState";
import React, { MutableRefObject } from "react";
import Box from "@mui/material/Box";
import { pack } from "msgpackr";
import { makeThrottledMessageSender } from "../WebsocketInterface";

export const levaTheme: LevaCustomTheme = {
  colors: {
    elevation1: "#e5e5e5",
    elevation2: "#ffffff",
    elevation3: "#f5f5f5",
    accent1: "#0066dc",
    accent2: "#1976d2",
    accent3: "#3c93ff",
    folderWidgetColor: "#777",
    highlight1: "#000000",
    highlight2: "#1d1d1d",
    highlight3: "#000000",
    vivid1: "#ffcc00",
  },
  radii: {
    xs: "2px",
    sm: "3px",
    lg: "10px",
  },
  space: {
    sm: "6px",
    md: "12px",
    rowGap: "8px",
    colGap: "8px",
  },
  fontSizes: {
    root: "0.9em",
  },
  fonts: {
    mono: "",
    sans: "",
  },
  sizes: {
    rootWidth: "350px",
    controlWidth: "170px",
    scrubberWidth: "10px",
    scrubberHeight: "14px",
    rowHeight: "24px",
    numberInputMinWidth: "60px",
    folderTitleHeight: "24px",
    checkboxSize: "16px",
    joystickWidth: "100px",
    joystickHeight: "100px",
    colorPickerWidth: "160px",
    colorPickerHeight: "100px",
    monitorHeight: "60px",
    titleBarHeight: "39px",
  },
  borderWidths: {
    root: "0px",
    input: "1px",
    focus: "1px",
    hover: "1px",
    active: "1px",
    folder: "1px",
  },
  fontWeights: {
    label: "normal",
    folder: "normal",
    button: "normal",
  },
};

interface GeneratedControlsProps {
  useGui: UseGui;
  websocketRef: MutableRefObject<WebSocket | null>;
}

/** One tab in the control panel. */
export default function GeneratedControls(props: GeneratedControlsProps) {
  const guiNames = props.useGui((state) => state.guiNames);
  const guiConfigFromName = props.useGui((state) => state.guiConfigFromName);

  // Add callbacks to guiConfigFromName.
  const suppressOnChange = React.useRef<{ [key: string]: boolean }>({});

  // We're going to try and build an object that looks like:
  // {"folder name": {"input name": leva config}}
  const guiConfigWithCallbacks: { [key: string]: any } = {};
  for (const key of guiNames) {
    const { levaConf, folderName } = guiConfigFromName[key];

    // Create object for folder if it doesn't exist yet.
    if (guiConfigWithCallbacks[folderName] === undefined)
      guiConfigWithCallbacks[folderName] = {};

    // Hacky stuff that lives outside of TypeScript...
    if (levaConf["type"] === "BUTTON") {
      // Add a button.
      guiConfigWithCallbacks[folderName][key] = button((_get: any) => {
        const message: GuiUpdateMessage = {
          type: "gui_update",
          name: key,
          value: true,
        };
        props.websocketRef.current!.send(pack(message));
      }, levaConf["settings"]);
    } else {
      // Add any other kind of input.
      const sendUpdate = makeThrottledMessageSender(props.websocketRef, 50);
      guiConfigWithCallbacks[folderName][key] = {
        ...levaConf,
        onChange: (value: any, _propName: any, options: any) => {
          if (options.initial) return;
          if (suppressOnChange.current[key]) {
            delete suppressOnChange.current[key];
            return;
          }
          const message: GuiUpdateMessage = {
            type: "gui_update",
            name: key,
            value: value,
          };
          sendUpdate(message);
        },
      };
    }
  }
  for (const folderName of Object.keys(guiConfigWithCallbacks)) {
    // Apply Leva folder() wrapper...
    guiConfigWithCallbacks[folderName] = folder(
      guiConfigWithCallbacks[folderName]
    );
  }

  // Make Leva controls.
  const levaStore = useCreateStore();
  const [, set] = useControls(
    () => guiConfigWithCallbacks,
    { store: levaStore },
    [guiConfigFromName]
  );

  // Logic for setting control inputs when items are put onto the guiSetQueue.
  const guiSetQueue = props.useGui((state) => state.guiSetQueue);
  const applyGuiSetQueue = props.useGui((state) => state.applyGuiSetQueue);
  const timeouts = React.useRef<{ [key: string]: NodeJS.Timeout }>({});
  React.useEffect(() => {
    if (Object.keys(guiSetQueue).length === 0) return;
    applyGuiSetQueue((name, value) => {
      suppressOnChange.current[name] = true;

      // Suppression timeout. Resolves some issues with onChange() not firing
      // after we call set... this is hacky and should be revisited.
      clearTimeout(timeouts.current[name]);
      timeouts.current[name] = setTimeout(() => {
        suppressOnChange.current[name] = false;
      }, 10);

      // Set Leva control.
      set({ [name]: value });
    });
  }, [guiSetQueue, applyGuiSetQueue, set]);

  // Leva theming is a bit limited, so we hack at styles here...
  return (
    <Box
      component="div"
      sx={{
        "& label": { color: "#777" },
        "& input[type='checkbox']~label svg path": {
          stroke: "#fff !important",
        },
        "& button": { color: "#fff !important", height: "2em" },
      }}
    >
      <LevaPanel
        fill
        flat
        titleBar={false}
        theme={levaTheme}
        store={levaStore}
        hideCopyButton
      />
    </Box>
  );
}
