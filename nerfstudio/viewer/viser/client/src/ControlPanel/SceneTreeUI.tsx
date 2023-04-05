import { TreeItem } from "@mui/lab";
import { Box, IconButton } from "@mui/material";
import { VisibilityOffRounded, VisibilityRounded } from "@mui/icons-material";
import React from "react";
import { CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer";
import { UseSceneTree } from "../SceneTree";

interface SceneNodeUIChildrenProp {
  name: string;
  useSceneTree: UseSceneTree;
}

/** Control panel component for listing children of a scene node. */
function SceneNodeUIChildren(props: SceneNodeUIChildrenProp) {
  const children = props.useSceneTree(
    (state) => state.nodeFromName[props.name].children
  );
  return (
    <>
      {children.map((child_id) => {
        return (
          <SceneNodeUI
            name={child_id}
            useSceneTree={props.useSceneTree}
            key={child_id}
          />
        );
      })}
    </>
  );
}

interface SceneNodeUIProp {
  name: string;
  useSceneTree: UseSceneTree;
}

/** Control panel component for showing a particular scene node. */
export function SceneNodeUI(props: SceneNodeUIProp) {
  const sceneNode = props.useSceneTree(
    (state) => state.nodeFromName[props.name]
  );
  const threeObj = props.useSceneTree((state) => state.objFromName[props.name]);

  const visible = props.useSceneTree(
    (state) => state.visibilityFromName[props.name]
  );
  const setVisibility = props.useSceneTree((state) => state.setVisibility);
  const ToggleVisibilityIcon = visible
    ? VisibilityRounded
    : VisibilityOffRounded;

  const itemRef = React.useRef<HTMLElement>(null);

  const labelRef = React.useRef<CSS2DObject>();

  React.useEffect(() => {
    if (threeObj === undefined) return;
    if (threeObj === null) return;

    const labelDiv = document.createElement("div");
    labelDiv.style.cssText = `
      font-size: 0.7em;
      background-color: rgba(255, 255, 255, 0.9);
      padding: 0.5em;
      border-radius: 0.5em;
      color: #333;
    `;
    labelDiv.textContent = sceneNode.name;
    const label = new CSS2DObject(labelDiv);
    labelRef.current = label;

    if (itemRef.current!.matches(":hover")) {
      threeObj.add(label);
    }

    threeObj.visible = visible;
    return () => {
      threeObj.remove(label);
    };
  }, [threeObj, sceneNode.name, visible]);

  // Flag for indicating when we're dragging across hide/show icons. Makes it
  // easier to toggle visibility for many scene nodes at once.
  const suppressMouseLeave = React.useRef(false);

  const mouseEnter = (event: React.MouseEvent) => {
    // On hover, add an object label to the scene.
    threeObj.add(labelRef.current!);
    event.stopPropagation();
    if (event.buttons !== 0) {
      suppressMouseLeave.current = true;
      setVisibility(props.name, !visible);
    }
  };
  const mouseLeave = (event: React.MouseEvent) => {
    // Remove the object label.
    threeObj.remove(labelRef.current!);
    if (suppressMouseLeave.current) {
      suppressMouseLeave.current = false;
      return;
    }
    if (event.buttons !== 0) {
      setVisibility(props.name, !visible);
    }
  };

  const hideShowIcon = (
    <IconButton
      onClick={(event) => {
        event.stopPropagation();
        setVisibility(props.name, !visible);
      }}
      onMouseEnter={mouseEnter}
      onMouseLeave={mouseLeave}
    >
      <ToggleVisibilityIcon />
    </IconButton>
  );
  const label = (
    <Box component="div" onMouseEnter={mouseEnter} onMouseLeave={mouseLeave}>
      {sceneNode.name === "" ? "/" : sceneNode.name}
    </Box>
  );

  return (
    <TreeItem
      nodeId={"node_" + props.name.toString()}
      sx={{
        opacity: visible ? 1.0 : 0.5,
      }}
      ref={itemRef}
      icon={hideShowIcon}
      label={label}
    >
      <SceneNodeUIChildren
        name={props.name}
        useSceneTree={props.useSceneTree}
      />
    </TreeItem>
  );
}
