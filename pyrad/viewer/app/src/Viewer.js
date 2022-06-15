import React, { Component } from "react";
import "./App.css";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls";
import Stats from "three/examples/jsm/libs/stats.module.js";
import { GUI } from "dat.gui";
import { split_path } from "./utils";
import { SceneNode } from "./SceneNode";
import { ExtensibleObjectLoader } from "./ExtensibleObjectLoader";

function websocket_endpoint_from_url(url) {
  let endpoint = url.split("/").pop();
  if (endpoint == "") {
    let message =
      "Please set a websocket endpoint. For example, a correct URL may look like the following: http://localhost:3000/visiongpu10.csail.mit.edu:8051";
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
      needs_render: true,
      viewport_width: null,
      viewport_height: null,
      images: [],
      image_index: 0,
    };
    this.animate = this.animate.bind(this);
    this.set_object = this.set_object.bind(this);
    this.handle_command = this.handle_command.bind(this);

    // window.addEventListener('resize', this.handleResize.bind(this));
  }

  set_dirty() {
    this.setState({ needs_render: true });
  }

  handleResize() {
    this.state.viewport_width = this.getViewportWidth();
    this.state.viewport_height = this.getViewportHeight();
    this.state.camera_l.aspect =
      this.state.viewport_width / this.state.viewport_height;
    this.state.camera_l.updateProjectionMatrix();
    this.state.camera_r.updateProjectionMatrix();
    this.state.camera_orth.updateProjectionMatrix();
    this.state.renderer_l.setSize(
      this.state.viewport_width,
      (this.state.viewport_height * 2) / 3
    );
    this.state.renderer_r.setSize(
      this.state.viewport_width / 2,
      (this.state.viewport_height * 1) / 3
    );
    this.state.renderer_orth.setSize(
      this.state.viewport_width / 2,
      (this.state.viewport_height * 1) / 3
    );
  }

  animate() {
    requestAnimationFrame(this.animate);
    if (this.state.needs_render) {
      this.handleResize();
      let camera_l = findCameraObjectUnderObject3D(
        this.state.scene_tree.find(["Cameras", "Main Camera L"]).object
      );
      let camera_r = findCameraObjectUnderObject3D(
        this.state.scene_tree.find(["Cameras", "Main Camera R"]).object
      );
      let camera_orth = findCameraObjectUnderObject3D(
        this.state.scene_tree.find(["Cameras", "Main Camera Orth"]).object
      );
      camera_l.updateProjectionMatrix();
      camera_r.updateProjectionMatrix();
      camera_orth.updateProjectionMatrix();
      if (this.state.controls_l != null) {
        this.state.controls_l.update();
      }
      if (this.state.controls_r != null) {
        this.state.controls_r.update();
      }
      if (this.state.controls_orth != null) {
        this.state.controls_orth.update();
      }
      this.state.renderer_l.render(this.state.scene, camera_l);
      this.state.renderer_r.render(this.state.scene, camera_r);

      // NOTE(ethan): trying to add background image

      // console.log("ethan is here");
      if (this.state.websocket_is_opened) {
        this.state.websocket.send("TODO: put camera pose here");
      }

      this.state.renderer_orth.render(
        this.backgroundScene,
        this.backgroundCamera
      );
      // this.state.renderer_orth.render(this.state.scene, camera_orth);
      this.state.needs_render = false;
      this.set_dirty(); // TODO(ethan): remove this but make sure to include it everywhere
    }
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
        console.log("uses camera !!");
        // this.set_camera(obj);
        // this.set_3d_pane_size();
      }
      obj.castShadow = true;
      obj.receiveShadow = true;
      this.set_object(path, obj);
      this.set_dirty();
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
    console.log(cmd);

    // convert binary serialization format back to JSON
    cmd = msgpack.decode(new Uint8Array(cmd));
    console.log(cmd);

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
      // TODO(ethan): implement animations
      console.error("Animations not implemented yet.");
    } else if (cmd.type === "set_image") {
      console.log("ethan: setting camera");
      // var url = URL.createObjectURL(
      //   new Blob([cmd["image"].buffer], { type: "image/png" } /* (1) */)
      // );
      // console.log(url);
      // document.getElementById("my-img").src = url;
      var height = 300;
      var width = 200;
      //   console.log(cmd["image"]);
      var texture = new THREE.DataTexture(
        cmd["image"],
        width,
        height,
        THREE.RGBFormat
      );
      //   var texture = new THREE.TextureLoader().load("texture2.jpeg");
      // var texture = new THREE.TextureLoader().load(url);
      // console.log(texture);
      // console.log(this.backgroundMesh);
      this.backgroundMesh.material.map = texture;
      this.backgroundMesh.material.map.needsUpdate = true;
      this.backgroundMesh.material.needsUpdate = true;
    }
    this.set_dirty();

    if (cmd.type === "set_transform" && cmd.path === "/Cameras/Main Camera R") {
      let camera_r = findCameraObjectUnderObject3D(
        this.state.scene_tree.find(["Cameras", "Main Camera R"]).object
      );
      camera_r.updateProjectionMatrix();
      this.state.renderer_save.render(this.state.scene, camera_r);
      let imgData = this.state.renderer_save.domElement.toDataURL();
      this.state.images.push(imgData.toString());
      // this.save_image(imgData);
    }
  }

  save_scene() {}

  load_scene() {}

  // save_image() {}

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
    this.state.viewport_width = this.getViewportWidth(); // notice that this is half width
    this.state.viewport_height = this.getViewportHeight();

    // Scene
    this.state.scene = new THREE.Scene();
    this.state.scene.background = new THREE.Color(0xffffff);

    // GUI
    this.state.gui = new GUI();
    let scene_folder = this.state.gui.addFolder("Scene");
    scene_folder.open();
    this.state.scene_tree = new SceneNode(this.state.scene, scene_folder, () =>
      this.set_dirty()
    );
    // let save_folder = this.state.gui.addFolder("Save / Load / Capture");
    // save_folder.add(this, 'save_scene');
    // save_folder.add(this, 'load_scene');
    // save_folder.add(this, 'save_image');
    // this.state.gui.open();

    // console.log(this.mount);

    // Renderer Left
    this.state.renderer_l = new THREE.WebGLRenderer({ antialias: true });
    this.state.renderer_l.setPixelRatio(window.devicePixelRatio);
    this.state.renderer_l.setSize(
      this.state.viewport_width,
      (this.state.viewport_height * 2) / 3
    );
    this.state.renderer_l.domElement.style.border = "1px solid black";
    this.mount.appendChild(this.state.renderer_l.domElement);

    // Renderer Right
    let div = document.createElement("div");
    this.state.renderer_r = new THREE.WebGLRenderer({ antialias: true });
    this.state.renderer_r.setPixelRatio(window.devicePixelRatio);
    this.state.renderer_r.setSize(
      this.state.viewport_width / 2,
      (this.state.viewport_height * 1) / 3
    );
    this.state.renderer_r.domElement.style.border = "1px solid black";
    this.state.renderer_r.domElement.style = "inline-block";
    div.appendChild(this.state.renderer_r.domElement);

    this.state.renderer_orth = new THREE.WebGLRenderer({ antialias: true });
    this.state.renderer_orth.setPixelRatio(window.devicePixelRatio);
    this.state.renderer_orth.setSize(
      this.state.viewport_width / 2,
      (this.state.viewport_height * 1) / 3
    );
    this.state.renderer_orth.domElement.style.border = "1px solid black";
    this.state.renderer_orth.domElement.style = "inline-block";
    div.appendChild(this.state.renderer_orth.domElement);
    // div.display.style = "relative";

    // NOTE(ethan): not sure if this is needed anymore?
    this.state.renderer_save = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.state.renderer_save.setPixelRatio(1.0);
    this.state.renderer_save.setSize(960, 540);
    this.state.renderer_save.domElement.style.border = "1px solid black";
    this.state.renderer_save.domElement.style.display = "none";
    // div.appendChild(this.state.renderer_save.domElement);

    this.mount.appendChild(div);

    // console.log(this.mount);

    this.mount.style.display = "block";

    // Camera settings at https://stackoverflow.com/questions/46182845/field-of-view-aspect-ratio-view-matrix-from-projection-matrix-hmd-ost-calib/46195462
    // Camera left
    this.state.camera_l = new THREE.PerspectiveCamera(
      120,
      this.state.viewport_width / this.state.viewport_height,
      0.01,
      100
    );
    this.state.camera_l.position.x = 5;
    this.state.camera_l.position.y = -5;
    this.state.camera_l.position.z = -5;
    this.state.camera_l.up = new THREE.Vector3(0, -1, 0);
    // Controls left
    this.state.controls_l = new TrackballControls(
      this.state.camera_l,
      this.state.renderer_l.domElement
    );
    this.state.controls_l.rotateSpeed = 2.0;
    this.state.controls_l.zoomSpeed = 0.3;
    this.state.controls_l.panSpeed = 0.2;
    this.state.controls_l.staticMoving = false; // false is default
    this.state.controls_l.target.set(0, 0, 0); // focus point of the controls
    this.state.controls_l.autoRotate = false;
    this.state.controls_l.update();

    // Camera right
    this.state.camera_r = new THREE.PerspectiveCamera(
      120,
      this.state.viewport_width / (this.state.viewport_height / 2),
      0.01,
      100
    );
    this.state.camera_orth = new THREE.OrthographicCamera(
      -2,
      2,
      -2,
      2,
      0.01,
      100
    );
    // this.state.camera_r.position.x = 0;
    // this.state.camera_r.position.y = 0;
    // this.state.camera_r.position.z = 0;
    // this.state.camera_r.up = new THREE.Vector3(0, -1, 0);

    var texture = new THREE.TextureLoader().load("texture.jpeg");
    this.backgroundMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.MeshBasicMaterial({
        map: texture,
      })
      // material
    );
    // const geometry = new THREE.PlaneGeometry(1, 1);
    // const material = new THREE.MeshBasicMaterial({
    //   color: 0xffff00,
    //   side: THREE.DoubleSide,
    // });
    // const plane = new THREE.Mesh(geometry, material);
    this.backgroundMesh.material.depthTest = false;
    this.backgroundMesh.material.depthWrite = false;
    // Create your background scene
    this.backgroundScene = new THREE.Scene();
    this.backgroundCamera = new THREE.Camera();
    this.backgroundScene.add(this.backgroundCamera);
    this.backgroundScene.add(this.backgroundMesh);

    this.set_object(["Cameras", "Main Camera L"], this.state.camera_l);
    this.set_object(["Cameras", "Main Camera R"], this.state.camera_r);
    this.set_object(["Cameras", "Main Camera Orth"], this.state.camera_orth);

    // rotate the cameras to match the COLMAP format
    let rotY = new THREE.Matrix4();
    rotY.makeRotationY(Math.PI);
    let rotZ = new THREE.Matrix4();
    rotZ.makeRotationZ(Math.PI);
    let rot = rotY.multiply(rotZ);
    this.set_transform(["Cameras", "Main Camera R", "<object>"], rot.toArray());

    // this.state.scene_tree.find(["Cameras", "Main Camera L"]).set_property("position", [5, -5, -5]);
    this.state.scene_tree
      .find(["Cameras", "Main Camera R"])
      .set_property("position", [-2, -2, -2]);

    // Controls right
    // this.state.controls_r = new TrackballControls(this.state.scene_tree.find(["Cameras", "Main Camera R"]).object, this.state.renderer_r.domElement);
    // this.state.controls_r.rotateSpeed = 2.0;
    // this.state.controls_r.zoomSpeed = 0.3;
    // this.state.controls_r.panSpeed = 0.2;
    // this.state.controls_r.staticMoving = false; // false is default
    // this.state.controls_r.target.set(0, 0, 0); // focus point of the controls
    // this.state.controls_r.autoRotate = false;
    // this.state.controls_r.update();

    this.set_transform(
      ["Cameras", "Main Camera Orth", "<object>"],
      rot.toArray()
    );
    this.state.scene_tree
      .find(["Cameras", "Main Camera Orth"])
      .set_property("position", [-1, -1, -1]);

    let camera_axes_l = new THREE.AxesHelper(1);
    let camera_axes_r = new THREE.AxesHelper(1);
    let camera_axes_orth = new THREE.AxesHelper(1);
    this.set_object(["Cameras", "Main Camera L", "Axes"], camera_axes_l);
    this.set_object(["Cameras", "Main Camera R", "Axes"], camera_axes_r);
    this.set_object(["Cameras", "Main Camera Orth", "Axes"], camera_axes_orth);

    // Axes display
    let axes = new THREE.AxesHelper(5);
    this.set_object(["Axes"], axes);

    // var arrowPos = new THREE.Vector3(0, 0, 0);
    // var arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), arrowPos, 60, 0xFF0000, 20, 10)
    // this.set_object(["arrow"], arrow);

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
    this.mount.appendChild(stats.dom);

    // let camera = new THREE.PerspectiveCamera(75, this.state.width / this.state.height, 0.1, 1000);
    // let helper = new THREE.CameraHelper(camera);
    // // scene.add( helper );
    // // console.log(helper);
    // console.log(helper);
    // // this.set_object(["Helper"], helper);
    // // console.log(this.state.scene);

    // Cube
    // var geometry = new THREE.BoxGeometry(1, 1, 1);
    // var material = new THREE.MeshBasicMaterial({color: 0x00ff00});
    // var cube = new THREE.Mesh(geometry, material);
    // this.state.scene.add(cube);

    this.animate();
  }

  componentWillUnmount() {
    // this.mount.removeChild(this.state.renderer_l.domElement);
    // this.mount.removeChild(this.state.renderer_r.domElement);
    removeAllChildNodes(this.mount);
  }

  render() {
    return <div ref={(ref) => (this.mount = ref)} />;
  }
}
