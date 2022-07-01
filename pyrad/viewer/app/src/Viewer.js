import React, { Component } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls";
import Stats from "three/examples/jsm/libs/stats.module.js";
import { GUI } from "dat.gui";
import { split_path } from "./utils";
import { SceneNode } from "./SceneNode";
import { ExtensibleObjectLoader } from "./ExtensibleObjectLoader";
import { WebRTCVideo } from "./Video";
import { BackgroundDiv } from "./Background";

function websocket_endpoint_from_url(url) {
  let endpoint = url.split("/").pop();
  if (endpoint == "") {
    let message =
      "Please set the websocket endpoint. E.g., a correct URL may be: http://localhost:4000/localhost:8051";
    window.alert(message);
    return null;
  }
  return endpoint;
}

function removeAllChildNodes(parent) {
  // https://www.javascripttutorial.net/dom/manipulating/remove-all-child-nodes/
  while (parent.firstChild) {
    parent.removeChild(parent.firstChild);
  }
}

function findCameraObjectUnderObject3D(object) {
  // seach the tree for the camera
  if (object instanceof THREE.Camera) {
    return object;
  } else if (object instanceof THREE.CameraHelper) {
    return object.camera;
  }
  for (let name of Object.keys(object.children)) {
    return findCameraObjectUnderObject3D(object.children[name]);
  }
  return null;
}

var msgpack = require("msgpack-lite");

export class Viewer extends Component {
  constructor(props) {
    super(props);
    this.state = {
      scene: null,
      controls: null,
      renderer: null,
      gui: null,
      scene_tree: null,
      viewport_width: null,
      viewport_height: null,
      images: [],
      image_index: 0,
    };
    this.update = this.update.bind(this);
    this.set_object = this.set_object.bind(this);
    this.handle_command = this.handle_command.bind(this);
    this.threejs_ref = React.createRef();
  }

  handleResize() {
    this.state.viewport_width = this.getViewportWidth();
    this.state.viewport_height = this.getViewportHeight();
    this.state.camera_main.aspect =
      this.state.viewport_width / this.state.viewport_height;
    this.state.camera_main.updateProjectionMatrix();
    this.state.renderer_main.setSize(
      this.state.viewport_width,
      this.state.viewport_height
    );
  }

  update() {
    requestAnimationFrame(this.update);
    this.handleResize();
    let camera_main = findCameraObjectUnderObject3D(
      this.state.scene_tree.find(["Cameras", "Main Camera"]).object
    );
    camera_main.updateProjectionMatrix();
    if (this.state.controls_main != null) {
      this.state.controls_main.update();
    }
    this.state.renderer_main.render(this.state.scene, camera_main);
  }

  save_image(image) {
    let url = "https://friends.ethanweber.me/save_image";
    let data = {
      index: this.state.image_index,
      image: image,
    };
    fetch(url, {
      method: "POST", // The method
      mode: "no-cors", // It can be no-cors, cors, same-origin
      headers: {
        "Content-Type": "application/json", // Your headers
      },
      body: JSON.stringify(data),
    }).then((returned) => {
      console.log(returned);
    });
    this.state.image_index += 1;
  }

  set_object(path, object) {
    this.state.scene_tree.find(path.concat(["<object>"])).set_object(object);
  }

  set_object_from_json(path, object_json) {
    let loader = new ExtensibleObjectLoader();
    loader.onTextureLoad = () => {
      this.set_dirty();
    };
    loader.parse(object_json, (obj) => {
      // console.log("ethan.");
      // console.log(obj.type);
      if (
        obj.geometry !== undefined &&
        obj.geometry.type === "BufferGeometry"
      ) {
        if (
          obj.geometry.attributes.normal === undefined ||
          obj.geometry.attributes.normal.count === 0
        ) {
          obj.geometry.computeVertexNormals();
        }
      } else if (obj.type.includes("Camera")) {
        // console.log("uses camera !!");
        // this.set_camera(obj);
        // this.set_3d_pane_size();
      }
      obj.castShadow = true;
      obj.receiveShadow = true;
      this.set_object(path, obj);
    });
  }

  set_transform(path, matrix) {
    this.state.scene_tree.find(path).set_transform(matrix);
  }

  delete_path(path) {
    if (path.length === 0) {
      console.error("Deleting the entire scene is not implemented.");
    } else {
      this.state.scene_tree.delete(path);
    }
  }

  set_property(path, property, value) {
    this.state.scene_tree.find(path).set_property(property, value);
    // TODO(ethan): handle this issue
    if (path[0] === "Background") {
      // The background is not an Object3d, so needs a little help.
      this.state.scene_tree.find(path).on_update();
    }
  }

  handle_command(cmd) {
    // console.log(cmd);

    // convert binary serialization format back to JSON
    cmd = msgpack.decode(new Uint8Array(cmd));
    // console.log(cmd);

    // TODO(ethan): ignore these or remove status. maybe incorporate into a clean view
    if (cmd.type === "status") {
      return;
    }

    if (cmd.type === "set_object") {
      let path = split_path(cmd.path);
      this.set_object_from_json(path, cmd.object);
    } else if (cmd.type === "set_transform") {
      let path = split_path(cmd.path);
      this.set_transform(path, cmd.matrix);
    } else if (cmd.type === "delete") {
      let path = split_path(cmd.path);
      this.delete_path(path);
    } else if (cmd.type === "set_property") {
      let path = split_path(cmd.path);
      this.set_property(path, cmd.property, cmd.value);
    } else if (cmd.type === "set_animation") {
      console.error("Not implemented.");
    } else if (cmd.type === "set_image") {
      console.error("Not implemented.");
    }
  }

  save_scene() {}

  load_scene() {}

  add_camera_helper(path, camera) {
    let camera_copy = camera.clone();
    camera_copy.far = 1; // so it doesn't show so far away
    this.set_object(path, new THREE.CameraHelper(camera_copy));
  }

  getViewportWidth() {
    return window.innerWidth - (window.innerWidth % 2);
  }

  getViewportHeight() {
    return window.innerHeight;
  }

  componentDidMount() {
    // Open the websocket
    let endpoint = websocket_endpoint_from_url(window.location.href);
    this.state.websocket_is_opened = false;
    this.state.websocket = new WebSocket("ws://" + endpoint + "/");
    this.state.websocket.onopen = () => {
      console.log("connected");
      this.state.websocket_is_opened = true;
    };
    this.state.websocket.binaryType = "arraybuffer";
    // 'command' for updates to Viewer
    this.state.websocket.onmessage = (cmd) => this.handle_command(cmd.data);
    this.state.websocket.onclose = function (evt) {
      console.log("onclose:", evt);
    };

    // Get size of screen
    this.state.viewport_width = this.getViewportWidth();
    this.state.viewport_height = this.getViewportHeight();

    // Scene
    this.state.scene = new THREE.Scene();

    // GUI
    this.state.gui = new GUI();
    let scene_folder = this.state.gui.addFolder("Scene");
    scene_folder.open();
    this.state.scene_tree = new SceneNode(this.state.scene, scene_folder);

    // Renderer main
    this.state.renderer_main = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    this.state.renderer_main.setPixelRatio(window.devicePixelRatio);
    this.state.renderer_main.setSize(
      this.state.viewport_width,
      this.state.viewport_height
    );
    this.state.renderer_main.domElement.style.border = "1px solid black";

    let domElement = document.getElementById("canvas-container-main");
    domElement.append(this.state.renderer_main.domElement);

    // Camera main
    this.state.camera_main = new THREE.PerspectiveCamera(
      120,
      this.state.viewport_width / this.state.viewport_height,
      0.01,
      100
    );
    this.state.camera_main.position.x = 5;
    this.state.camera_main.position.y = -5;
    this.state.camera_main.position.z = -5;
    this.state.camera_main.up = new THREE.Vector3(0, -1, 0);
    // Controls left
    this.state.controls_main = new TrackballControls(
      this.state.camera_main,
      this.state.renderer_main.domElement
    );
    this.state.controls_main.rotateSpeed = 2.0;
    this.state.controls_main.zoomSpeed = 0.3;
    this.state.controls_main.panSpeed = 0.2;
    this.state.controls_main.staticMoving = false; // false is default
    this.state.controls_main.target.set(0, 0, 0); // focus point of the controls
    this.state.controls_main.autoRotate = false;
    this.state.controls_main.update();

    this.set_object(["Cameras", "Main Camera"], this.state.camera_main);
    let camera_axes_main = new THREE.AxesHelper(1);
    this.set_object(["Cameras", "Main Camera", "Axes"], camera_axes_main);

    // Axes display
    let axes = new THREE.AxesHelper(5);
    this.set_object(["Axes"], axes);

    // Grid
    let grid = new THREE.GridHelper(20, 20);
    this.set_object(["Grid"], grid);

    // Lights
    let color = 0xffffff;
    let intensity = 1;
    let light = new THREE.AmbientLight(color, intensity);
    this.set_object(["Light"], light);

    // Stats
    var stats = new Stats();
    let domElement_stats = document.getElementById("canvas-container-main");
    domElement_stats.append(stats.dom);

    this.update();
  }

  render() {
    return (
      <div>
        <WebRTCVideo> </WebRTCVideo>
        {/* <BackgroundDiv> </BackgroundDiv> */}
        <div id="canvas-container-main"> </div>
        <div id="stats-container"> </div>
      </div>
      
    );
  }
}
