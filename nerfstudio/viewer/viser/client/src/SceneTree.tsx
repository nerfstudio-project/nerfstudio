import { createPortal } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { CoordinateFrame } from "./ThreeAssets";

import { immerable } from "immer";
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

// The covariance/contravariance rules are too complicated here, so we just
// type the reference with any.
export type MakeObject = (ref: React.RefObject<any>) => React.ReactNode;

/** Scenes will consist of nodes, which form a tree. */
export class SceneNode {
  [immerable] = true;

  public children: string[];

  constructor(public name: string, public make_object: MakeObject) {
    this.children = [];
  }
}

interface SceneTreeState {
  nodeFromName: { [key: string]: SceneNode };
  visibilityFromName: { [key: string]: boolean };
  objFromName: { [key: string]: THREE.Object3D };
}
export interface SceneTreeActions extends SceneTreeState {
  setObj(name: string, obj: THREE.Object3D): void;
  setVisibility(name: string, visible: boolean): void;
  clearObj(name: string): void;
  addSceneNode(nodes: SceneNode): void;
  removeSceneNode(name: string): void;
  resetScene(): void;
}

// Create default scene tree state.
// By default, the y-axis is up. Let's rotate everything so Z is up instead.
const rootFrameTemplate: MakeObject = (ref) => (
  <CoordinateFrame
    ref={ref}
    show_axes={false}
    quaternion={new THREE.Quaternion().setFromEuler(
      new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0)
    )}
  />
);
const rootAxesTemplate: MakeObject = (ref) => <CoordinateFrame ref={ref} />;

const rootNodeTemplate = new SceneNode("", rootFrameTemplate);
const rootAxesNode = new SceneNode("/WorldAxes", rootAxesTemplate);
rootNodeTemplate.children.push("/WorldAxes");

const cleanSceneTreeState = {
  nodeFromName: { "": rootNodeTemplate, "/WorldAxes": rootAxesNode },
  visibilityFromName: { "": true, "/WorldAxes": true },
  objFromName: {},
} as SceneTreeState;

/** Declare a scene state, and return a hook for accessing it. Note that we put
effort into avoiding a global state! */
export function useSceneTreeState() {
  return React.useState(() =>
    create(
      immer<SceneTreeState & SceneTreeActions>((set) => ({
        ...cleanSceneTreeState,
        setObj: (name, obj) =>
          set((state) => {
            state.objFromName[name] = obj;
          }),
        setVisibility: (name, visible) =>
          set((state) => {
            state.visibilityFromName[name] = visible;
          }),
        clearObj: (name) =>
          set((state) => {
            delete state.objFromName[name];
          }),
        addSceneNode: (node) =>
          set((state) => {
            if (node.name in state.nodeFromName) {
              state.nodeFromName[node.name] = {
                ...node,
                children: state.nodeFromName[node.name].children,
              };
            } else {
              const parent_name = node.name.split("/").slice(0, -1).join("/");
              state.nodeFromName[node.name] = node;
              state.nodeFromName[parent_name].children.push(node.name);
              if (!(node.name in state.visibilityFromName))
                state.visibilityFromName[node.name] = true;
            }
          }),
        removeSceneNode: (name) =>
          set((state) => {
            // Remove node from parent's children list.
            const parent_name = name.split("/").slice(0, -1).join("/");

            state.nodeFromName[parent_name].children = state.nodeFromName[
              parent_name
            ].children.filter((child_name) => child_name !== name);

            delete state.visibilityFromName[name];

            // If we want to remove "/tree", we should remove all of "/tree", "/tree/trunk", "/tree/branch", etc.
            const remove_names = Object.keys(state.nodeFromName).filter((n) =>
              n.startsWith(name)
            );
            remove_names.forEach((remove_name) => {
              delete state.nodeFromName[remove_name];
            });
          }),
        resetScene: () =>
          set((state) => {
            Object.assign(state, cleanSceneTreeState);
          }),
      }))
    )
  )[0];
}

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

interface SceneNodeThreeChildrenProps {
  name: string;
  useSceneTree: UseSceneTree;
}
function SceneNodeThreeChildren(props: SceneNodeThreeChildrenProps) {
  const children = props.useSceneTree(
    (state) => state.nodeFromName[props.name].children
  );
  const parentObj = props.useSceneTree(
    (state) => state.objFromName[props.name]
  );

  // Create a group of children inside of the parent object.
  return (
    parentObj &&
    createPortal(
      <group>
        {children.map((child_id) => {
          return (
            <SceneNodeThreeObject
              key={child_id}
              name={child_id}
              useSceneTree={props.useSceneTree}
            />
          );
        })}
      </group>,
      parentObj
    )
  );
}

interface SceneNodeThreeObjectProps {
  name: string;
  useSceneTree: UseSceneTree;
}

/** Component containing the three.js object and children for a particular scene node. */
export const SceneNodeThreeObject = React.memo(
  // This memo is very important for big scenes!!
  (props: SceneNodeThreeObjectProps) => {
    const sceneNode = props.useSceneTree(
      (state) => state.nodeFromName[props.name]
    );
    const setObj = props.useSceneTree((state) => state.setObj);
    const clearObj = props.useSceneTree((state) => state.clearObj);
    const ref = React.useRef<THREE.Object3D>(null);

    React.useEffect(() => {
      setObj(props.name, ref.current!);
      return () => clearObj(props.name);
    });

    return (
      <>
        {sceneNode.make_object(ref)}
        <SceneNodeUpdater
          name={props.name}
          objRef={ref}
          useSceneTree={props.useSceneTree}
        />
        <SceneNodeThreeChildren
          name={props.name}
          useSceneTree={props.useSceneTree}
        />
      </>
    );
  }
);

interface SceneNodeUpdaterProps {
  name: string;
  objRef: React.RefObject<THREE.Object3D>;
  useSceneTree: UseSceneTree;
}

/** Shove visibility updates into a separate components so the main object
 * component doesn't need to be repeatedly re-rendered.*/
function SceneNodeUpdater(props: SceneNodeUpdaterProps) {
  const visible = props.useSceneTree(
    (state) => state.visibilityFromName[props.name]
  );
  React.useEffect(() => {
    props.objRef.current!.visible = visible;
  }, [props, visible]);
  return <></>;
}
