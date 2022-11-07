/* eslint-disable no-underscore-dangle */
/* eslint-disable no-restricted-syntax */
import * as THREE from 'three';

function dispose(object) {
  if (!object) {
    return;
  }
  if (object.geometry) {
    object.geometry.dispose();
  }
  if (object.material) {
    if (Array.isArray(object.material)) {
      for (const material of object.material) {
        if (material.map) {
          material.map.dispose();
        }
        material.dispose();
      }
    } else {
      if (object.material.map) {
        object.material.map.dispose();
      }
      object.material.dispose();
    }
  }
}

export default class SceneNode {
  constructor(object, scene_state) {
    this.metadata = {}; // relevant metadata for the object
    this.object = object; // three.js objects
    this.children = {};
    this.scene_state = scene_state;
    for (const c of this.object.children) {
      this.add_child(c);
    }
  }

  get_scene_state() {
    return this.scene_state;
  }

  add_child(object) {
    const node = new SceneNode(object);
    this.children[object.name] = node;
    return node;
  }

  create_child(name) {
    const obj = new THREE.Group();
    obj.name = name;
    this.object.add(obj);
    return this.add_child(obj);
  }

  find(path) {
    if (path.length === 0) {
      return this;
    }
    const name = path[0];
    let child = this.children[name];
    if (child === undefined) {
      child = this.create_child(name);
    }
    return child.find(path.slice(1));
  }

  find_object(path) {
    return this.find(path.concat(['<object>'])).object;
  }

  find_no_create(path) {
    if (path.length === 0) {
      return this;
    }
    const name = path[0];
    const child = this.children[name];
    if (child === undefined) {
      return null;
    }
    return child.find(path.slice(1));
  }

  find_object_no_create(path) {
    return this.find_no_create(path.concat(['<object>'])).object;
  }

  set_property(property, value) {
    if (property === 'position') {
      this.object.position.set(value[0], value[1], value[2]);
    } else if (property === 'quaternion') {
      this.object.quaternion.set(value[0], value[1], value[2], value[3]);
    } else if (property === 'scale') {
      this.object.scale.set(value[0], value[1], value[2]);
    } else {
      this.object[property] = value;
    }
    this.vis_controller.updateDisplay();
  }

  set_transform(matrix) {
    const mat = new THREE.Matrix4();
    mat.fromArray(matrix);
    mat.decompose(
      this.object.position,
      this.object.quaternion,
      this.object.scale,
    );
  }

  set_object_from_path(path, object) {
    this.find(path.concat(['<object>'])).set_object(object);
  }

  set_object(object) {
    const parent = this.object.parent;
    this.dispose_recursive();
    this.object.parent.remove(this.object);
    this.object = object;
    parent.add(object);
  }

  dispose_recursive() {
    for (const name of Object.keys(this.children)) {
      this.children[name].dispose_recursive();
    }
    dispose(this.object);
  }

  delete(path) {
    if (path.length === 0) {
      console.error("Can't delete an empty path.");
    } else {
      const parent = this.find(path.slice(0, path.length - 1));
      const name = path[path.length - 1];
      const child = parent.children[name];
      if (child !== undefined) {
        child.dispose_recursive();
        parent.object.remove(child.object);
        delete parent.children[name];
      }
    }
  }
}
