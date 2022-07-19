import React, { Component } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import Stats from "three/examples/jsm/libs/stats.module.js";
import { GUI } from "dat.gui";
import { split_path } from "./utils";
import { SceneNode } from "./SceneNode";
import { ExtensibleObjectLoader } from "./ExtensibleObjectLoader";

function websocket_endpoint_from_url(url) {
  let endpoint = url.split("?").pop();
  if (endpoint == "") {
    let message =
      "Please set the websocket endpoint. E.g., a correct URL may be: http://localhost:4000?localhost:8051";
    window.alert(message);
    return null;
  }
  return endpoint;
}

function findCameraObjectUnderObject3D(object) {
  // seach the tree for the camera
  if (object instanceof THREE.Camera) {
    return object;
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
    };
    this.created_controls = false;
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
    this.send_camera_over_websocket();
  }

  send_camera_over_websocket() {
    /* update the camera information in the python server
                            if the websocket is connected */
    // console.log("send_camera_over_websocket");
    if (this.state.websocket.readyState === WebSocket.OPEN) {
      // update the camera information in the python server
      let camera_main = findCameraObjectUnderObject3D(
        this.state.scene_tree.find(["Cameras", "Main Camera"]).object
      );
      // console.log(camera_main.toJSON());
      let cmd = "set_object";
      let path = "Cameras/Main Camera";
      let data = {
        type: cmd,
        path: path,
        object: camera_main.toJSON(),
      };
      let message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  send_training_state_over_websocket(value) {
    if (this.state.websocket.readyState === WebSocket.OPEN) {
      let cmd = "set_training_state";
      let path = "Training State";
      let data = {
        type: cmd,
        path: path,
        training_state: value,
      };
      let message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  send_output_type_over_websocket(value) {
    /* update the output option in the python server
                            if the user changes selection */
    if (this.state.websocket.readyState === WebSocket.OPEN) {
      let cmd = "set_output_type";
      let path = "Output Type";
      let data = {
        type: cmd,
        path: path,
        output_type: value,
      };
      let message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  set_object(path, object) {
    if (!(object instanceof THREE.Camera)) {
      this.state.scene_tree.find(path.concat(["<object>"])).set_object(object);
    }
  }

  set_object_from_json(path, object_json) {
    let loader = new ExtensibleObjectLoader();
    loader.parse(object_json, (obj) => {
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

  set_output_options(object) {
    if (this.created_controls === false) {
      let output_options_control = new (function () {
        this.output_options = "default";
      })();
      let params = {
        switch: false
      };
      // add controls
      this.state.gui
        .add(output_options_control, "output_options", object)
        .name("Output Options")
        .listen()
        .onChange((value) => {
          this.send_output_type_over_websocket(value);
        });
      this.state.gui.add(params, "switch")
        .name("Pause Training?")
        .listen()
        .onChange((value)=>{
          this.send_training_state_over_websocket(value);
      });
    }
    this.created_controls = true;
  }

  handle_command(cmd) {
    // convert binary serialization format back to JSON
    cmd = msgpack.decode(new Uint8Array(cmd));
    // console.log(cmd);

    // three js scene commands
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
    } else if (cmd.type == "set_output_options") {
      this.set_output_options(cmd.output_options);
    }
    // web rtc commands
    else if (cmd.type === "answer") {
      let answer = cmd.data;
      this.state.pc.setRemoteDescription(answer);
    }

    // possibly update the camera information in the python server
    // this.send_camera_over_websocket();
  }

  getViewportWidth() {
    return window.innerWidth - (window.innerWidth % 2);
  }

  getViewportHeight() {
    return window.innerHeight;
  }

  setupWebRTC() {
    console.log("setting up WebRTC");
    // The iceServers config comes from the following URL:
    // https://www.metered.ca/tools/openrelay/
    this.state.pc = new RTCPeerConnection({
      iceServers: [
        {
          urls: "stun:openrelay.metered.ca:80",
        },
        {
          urls: "turn:openrelay.metered.ca:80",
          username: "openrelayproject",
          credential: "openrelayproject",
        },
        {
          urls: "turn:openrelay.metered.ca:443",
          username: "openrelayproject",
          credential: "openrelayproject",
        },
        {
          urls: "turn:openrelay.metered.ca:443?transport=tcp",
          username: "openrelayproject",
          credential: "openrelayproject",
        },
      ],
    });

    // connect video
    this.state.pc.addEventListener("track", function (evt) {
      if (evt.track.kind == "video") {
        console.log("setting event stream");
        document.getElementById("WebRTCVideo-video").srcObject = evt.streams[0];
      }
    });
    this.state.pc.addTransceiver("video", { direction: "recvonly" });

    this.state.pc
      .createOffer()
      .then((offer) => {
        return this.state.pc.setLocalDescription(offer);
      })
      .then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
          if (this.state.pc.iceGatheringState === "complete") {
            resolve();
          } else {
            var checkState = () => {
              if (this.state.pc.iceGatheringState === "complete") {
                this.state.pc.removeEventListener(
                  "icegatheringstatechange",
                  checkState
                );
                resolve();
              }
            };
            this.state.pc.addEventListener(
              "icegatheringstatechange",
              checkState
            );
          }
        });
      })
      .then(() => {
        var offer = this.state.pc.localDescription;
        console.log("sending the offer");
        let cmd = "offer";
        let path = "";
        let data = {
          type: cmd,
          path: path,
          data: {
            sdp: offer.sdp,
            type: offer.type,
          },
        };
        let message = msgpack.encode(data);
        this.state.websocket.send(message);
      });
  }

  componentDidMount() {
    // Open the websocket
    let endpoint = websocket_endpoint_from_url(window.location.href);
    this.state.websocket = new WebSocket("ws://" + endpoint + "/");
    this.state.websocket.onopen = () => {
      console.log("websocket connected");
      this.setupWebRTC();
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
    this.state.camera_main.position.z = 5;
    this.state.camera_main.up = new THREE.Vector3(0, 0, 1);

    this.state.controls_main = new OrbitControls(
      this.state.camera_main,
      this.state.renderer_main.domElement
    );
    this.state.controls_main.rotateSpeed = 2.0;
    this.state.controls_main.zoomSpeed = 0.3;
    this.state.controls_main.panSpeed = 0.2;
    this.state.controls_main.target.set(0, 0, 0); // focus point of the controls
    this.state.controls_main.autoRotate = false;
    this.state.controls_main.enableDamping = true;
    this.state.controls_main.dampingFactor = 1.0;
    this.state.controls_main.update();

    let path = ["Cameras", "Main Camera"];
    this.state.scene_tree
      .find(path.concat(["<object>"]))
      .set_object(this.state.camera_main);

    // Axes display
    let axes = new THREE.AxesHelper(5);
    this.set_object(["Axes"], axes);

    // Grid
    let grid = new THREE.GridHelper(20, 20);
    grid.rotateX(Math.PI / 2); // now on xy plane
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
        <div className="WebRTCVideo">
          <video id="WebRTCVideo-video" autoPlay playsInline muted></video>
        </div>
        <div id="canvas-container-main"> </div>
        <div id="stats-container"> </div>
      </div>
    );
  }
}
