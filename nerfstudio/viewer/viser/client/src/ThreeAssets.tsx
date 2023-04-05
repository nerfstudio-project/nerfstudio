import { extend } from "@react-three/fiber";
import {
  MeshLineGeometry as MeshLine,
  MeshLineMaterial,
  raycast as MeshLineRaycast,
} from "meshline";
import React from "react";
import * as THREE from "three";

extend({ MeshLine, MeshLineMaterial });

const axis_geom = new THREE.CylinderGeometry(1.0, 1.0, 1.0, 16, 1);
const x_material = new THREE.MeshBasicMaterial({ color: 0xcc0000 });
const y_material = new THREE.MeshBasicMaterial({ color: 0x00cc00 });
const z_material = new THREE.MeshBasicMaterial({ color: 0x0000cc });

const origin_geom = new THREE.SphereGeometry(1.0);
const origin_material = new THREE.MeshBasicMaterial({ color: 0xecec00 });

interface CoordinateFrameProps {
  quaternion?: THREE.Quaternion;
  position?: THREE.Vector3;
  show_axes?: boolean;
  axes_length?: number;
  axes_radius?: number;
}

/** Helper for adding coordinate frames as scene nodes. */
export const CoordinateFrame = React.forwardRef<
  THREE.Group,
  CoordinateFrameProps
>(
  (
    {
      quaternion = undefined,
      position = undefined,
      show_axes = true,
      axes_length = 0.5,
      axes_radius = 0.0125,
    }: CoordinateFrameProps,
    ref
  ) => {
    return (
      <group ref={ref} quaternion={quaternion} position={position}>
        {show_axes && (
          <>
            <mesh
              geometry={origin_geom}
              material={origin_material}
              scale={
                new THREE.Vector3(
                  axes_radius * 2.5,
                  axes_radius * 2.5,
                  axes_radius * 2.5
                )
              }
            />
            <mesh
              geometry={axis_geom}
              rotation={new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0)}
              position={[0.5 * axes_length, 0.0, 0.0]}
              scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
              material={x_material}
            />
            <mesh
              geometry={axis_geom}
              position={[0.0, 0.5 * axes_length, 0.0]}
              scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
              material={y_material}
            />
            <mesh
              geometry={axis_geom}
              rotation={new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)}
              position={[0.0, 0.0, 0.5 * axes_length]}
              scale={new THREE.Vector3(axes_radius, axes_length, axes_radius)}
              material={z_material}
            />
          </>
        )}
      </group>
    );
  }
);

// Camera frustum helper. We jitter to prevent z-fighting for overlapping lines.
const jitter = () => Math.random() * 1e-5;
const frustum_points: number[] = [];
frustum_points.push(0, 0, 0);
frustum_points.push(-1, -1, 1);
frustum_points.push(1, -1, 1);
frustum_points.push(0, 0, 0);
frustum_points.push(-1, 1, 1);
frustum_points.push(1, 1, 1);
frustum_points.push(0, 0, 0);
frustum_points.push(-1 + jitter(), 1 + jitter(), 1 + jitter());
frustum_points.push(-1, -1, 1);
frustum_points.push(1 + jitter(), -1 + jitter(), 1 + jitter());
frustum_points.push(1, 1, 1);

interface CameraFrustumProps {
  fov: number;
  aspect: number;
  scale: number;
  color: number;
}

/** Helper for visualizing camera frustums.

Note that:
 - This is currently just a pyramid, note a frustum. :-)
 - We currently draw two redundant/overlapping lines. This could be optimized. */
export const CameraFrustum = React.forwardRef<THREE.Group, CameraFrustumProps>(
  (props, ref) => {
    const y = Math.tan(props.fov / 2.0);
    const x = y * props.aspect;
    return (
      <group ref={ref}>
        <mesh
          raycast={MeshLineRaycast}
          scale={
            new THREE.Vector3(props.scale * x, props.scale * y, props.scale)
          }
        >
          {/* @ts-ignore */}
          <meshLine attach="geometry" points={frustum_points} />
          {/* @ts-ignore */}
          <meshLineMaterial
            attach="material"
            transparent
            lineWidth={0.02}
            color={props.color}
          />
        </mesh>
      </group>
    );
  }
);
