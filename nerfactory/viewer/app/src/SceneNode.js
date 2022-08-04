/* eslint-disable no-underscore-dangle */
/* eslint-disable no-restricted-syntax */
import * as THREE from 'three';

const dat = require('dat.gui');

function remove_folders(gui) {
  for (const name of Object.keys(gui.__folders)) {
    const folder = gui.__folders[name];
    remove_folders(folder);
    dat.dom.dom.unbind(window, 'resize', folder.__resizeHandler);
    gui.removeFolder(folder);
  }
}

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
  constructor(object, folder) {
    this.object = object;
    this.folder = folder;
    this.children = {};
    this.create_controls();
    for (const c of this.object.children) {
      this.add_child(c);
    }
  }

  add_child(object) {
    const f = this.folder.addFolder(object.name);
    const node = new SceneNode(object, f);
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

  create_controls() {
    if (this.vis_controller !== undefined) {
      this.folder.domElement.removeChild(this.vis_controller.domElement);
    }
    this.vis_controller = new dat.controllers.BooleanController(
      this.object,
      'visible',
    );
    this.folder.domElement.prepend(this.vis_controller.domElement);
    this.vis_controller.domElement.style.height = '0';
    this.vis_controller.domElement.style.float = 'right';
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

  set_object(object) {
    const parent = this.object.parent;
    this.dispose_recursive();
    this.object.parent.remove(this.object);
    this.object = object;
    parent.add(object);
    this.create_controls();
  }

  dispose_recursive() {
    for (const name of Object.keys(this.children)) {
      this.children[name].dispose_recursive();
    }
    dispose(this.object);
  }

  delete(path) {
    if (path.length === 0) {
      console.error("Can't delete an empty path");
    } else {
      const parent = this.find(path.slice(0, path.length - 1));
      const name = path[path.length - 1];
      const child = parent.children[name];
      if (child !== undefined) {
        child.dispose_recursive();
        parent.object.remove(child.object);
        remove_folders(child.folder);
        parent.folder.removeFolder(child.folder);
        delete parent.children[name];
      }
    }
  }
}
