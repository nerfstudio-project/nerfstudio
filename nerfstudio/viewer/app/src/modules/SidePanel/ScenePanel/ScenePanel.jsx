import * as React from 'react';
import * as THREE from 'three';

import { FaTractor } from 'react-icons/fa';

import { Button, Collapse, IconButton } from '@mui/material';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import ArrowCircleUpIcon from '@mui/icons-material/ArrowCircleUp';
import {
  ExpandLess,
  ExpandMore,
  Videocam,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material/';
import SceneNode from '../../../SceneNode';
import LevaTheme from '../../../themes/leva_theme.json';

export const snap_to_camera = (sceneTree, camera, matrix) => {
  const mat = new THREE.Matrix4();
  mat.fromArray(matrix.elements);
  mat.decompose(camera.position, camera.quaternion, camera.scale);
  const unit = new THREE.Vector3(0, 0, -1);
  const viewDirection = unit.applyMatrix4(mat);
  sceneTree.metadata.camera_controls.setLookAt(
    camera.position.x,
    camera.position.y,
    camera.position.z,
    viewDirection.x,
    viewDirection.y,
    viewDirection.z,
    {enableTransition:true}
  );
};

interface ListItemProps {
  name: String;
  sceneTree: SceneNode;
  scene_node: SceneNode;
  level: Number;
  groupVisible: Boolean;
  canSnap: Boolean;
}

function MenuItems(props: ListItemProps) {
  const name = props.name;
  const sceneTree = props.sceneTree;
  const scene_node = props.scene_node;
  const level = props.level;
  const groupVisible = props.groupVisible;
  const canSnap = props.canSnap;

  // TODO: sort the keys by string
  const terminal = Object.keys(scene_node.children).includes('<object>');

  const getCamera = (node) => {
    return node.object.children[0];
  };

  const num_children = Object.keys(scene_node.children).length;
  if (num_children === 0) {
    return null;
  }

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [open, setOpen] = React.useState(true);
  const toggleOpen = () => {
    setOpen(!open);
  };

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [visible, setVisible] = React.useState(groupVisible);
  if (terminal) {
    scene_node.object.visible = visible;
  }

  const toggleVisible = (e) => {
    e.stopPropagation();
    setVisible(!visible);
    scene_node.object.traverse((obj) => {
      if (obj.name === 'CAMERA_LABEL') {
        // eslint-disable-next-line no-param-reassign
        obj.visible = !visible;
      }
    });
  };

  // eslint-disable-next-line react-hooks/rules-of-hooks
  React.useEffect(() => {
    setVisible(groupVisible);
  }, [groupVisible]);

  return (
    <>
      <ListItemButton
        onClick={terminal ? null : toggleOpen}
        dense
        sx={
          terminal
            ? {
                pl: 2 + level * 2,
                color: visible
                  ? LevaTheme.colors.accent2
                  : LevaTheme.colors.disabled,
              }
            : {
                pl: 2 + level * 2,
                bgcolor: open
                  ? LevaTheme.colors.elevation3
                  : LevaTheme.colors.elevation1,
                color: visible
                  ? LevaTheme.colors.accent2
                  : LevaTheme.colors.disabled,
              }
        }
      >
        <ListItemIcon
          sx={{
            color: visible
              ? LevaTheme.colors.accent2
              : LevaTheme.colors.disabled,
          }}
        >
          <FaTractor />
        </ListItemIcon>
        <ListItemText primary={name} />

        {canSnap && (
          <IconButton
            aria-label="visibility"
            onClick={() =>
              snap_to_camera(
                sceneTree,
                sceneTree.metadata.camera,
                getCamera(scene_node).matrix,
              )
            }
            sx={{ mr: 1 }}
          >
            <Videocam />
          </IconButton>
        )}

        <IconButton aria-label="visibility" onClick={toggleVisible}>
          {visible ? <Visibility /> : <VisibilityOff />}
        </IconButton>

        {terminal
          ? null
          : (() => {
              if (open) {
                return <ExpandLess />;
              }
              return <ExpandMore />;
            })()}
      </ListItemButton>

      {terminal ? null : (
        <Collapse in={open} timeout="auto">
          <List>
            {Object.keys(scene_node.children)
              .filter((key) => {
                if (key === 'Camera') {
                  return false; // skip
                }
                return true;
              })
              .map((key) => (
                <MenuItems
                  name={key}
                  key={key}
                  sceneTree={sceneTree}
                  scene_node={scene_node.children[key]}
                  level={level + 1}
                  groupVisible={visible}
                  canSnap={name === 'Training Cameras'}
                />
              ))}
          </List>
        </Collapse>
      )}
    </>
  );
}

interface ClickableListProps {
  sceneTree: object;
}

function ClickableList(props: ClickableListProps) {
  const sceneTree = props.sceneTree;

  return (
    <List sx={{ color: LevaTheme.colors.accent2 }}>
      <MenuItems
        name="Scene"
        sceneTree={sceneTree}
        scene_node={sceneTree}
        level={0}
        groupVisible
      />
    </List>
  );
}

export default function ScenePanel(props) {
  const sceneTree = props.sceneTree;
  const camera_main = sceneTree.find_object_no_create([
    'Cameras',
    'Main Camera',
  ]);
  const setUp = () => {
    const rot = camera_main.rotation;
    const unitY = new THREE.Vector3(0, 1, 0);
    const upVec = unitY.applyEuler(rot);

    const grid = sceneTree.find_object_no_create(['Grid']);
    grid.setRotationFromEuler(rot);

    const pos = new THREE.Vector3();
    camera_main.getWorldPosition(pos);
    camera_main.up.set(upVec.x, upVec.y, upVec.z);
    sceneTree.metadata.camera_controls.updateCameraUp();
    sceneTree.metadata.camera_controls.setLookAt(pos.x, pos.y, pos.z, 0, 0, 0);
    const points = [new THREE.Vector3(0, 0, 0), upVec.multiplyScalar(2)];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xaa46fc,
      linewidth: 1,
    });
    const line = new THREE.LineSegments(geometry, material);
    sceneTree.set_object_from_path(['Viewer Up Vector'], line);
  };
  return (
    <div className="ScenePanel">
      <div className="CameraPanel-top-button">
        <Button
          size="small"
          variant="outlined"
          startIcon={<ArrowCircleUpIcon />}
          onClick={setUp}
        >
          Reset Up Direction
        </Button>
      </div>
      <ClickableList sceneTree={sceneTree} />
    </div>
  );
}
