import * as THREE from 'three';

import React, { Component } from 'react';

import { ExtensibleObjectLoader } from '../../ExtensibleObjectLoader';
import { GUI } from 'dat.gui';
import { SceneNode } from '../../SceneNode';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls';
import { split_path } from '../../utils';

import { Leva, button, buttonGroup, useControls } from 'leva';
import { useDispatch, useSelector } from 'react-redux';

function websocket_endpoint_from_url(url) {
  const endpoint = url.split('?').pop();
  if (endpoint == '') {
    const message =
      'Please set the websocket endpoint. E.g., a correct URL may be: http://localhost:4000?localhost:8051';
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
  for (const name of Object.keys(object.children)) {
    return findCameraObjectUnderObject3D(object.children[name]);
  }
  return null;
}

const msgpack = require('msgpack-lite');

export class ViewerState extends Component {
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
    this.update = this.update.bind(this);
    this.set_object = this.set_object.bind(this);
    this.handle_command = this.handle_command.bind(this);
    this.threejs_ref = React.createRef();
    this.props = props;

    // const renderingState = useSelector((state) => state.renderingState);
    // const [isTraining, setIsTraining] = React.useState(
    //   renderingState.isTraining,
    // );
    // const [outputOptions, setOutputOptions] = React.useState(
    //   renderingState.output_options,
    // );
    // this.setOutputOptions = setOutputOptions;
  }

  handleResize() {
    this.state.viewport_width = this.getViewportWidth();
    this.state.viewport_height = this.getViewportHeight();
    this.state.camera_main.aspect =
      this.state.viewport_width / this.state.viewport_height;
    this.state.camera_main.updateProjectionMatrix();
    this.state.renderer_main.setSize(
      this.state.viewport_width,
      this.state.viewport_height,
    );
  }

  update() {
    requestAnimationFrame(this.update);
    this.handleResize();
    const camera_main = findCameraObjectUnderObject3D(
      this.state.scene_tree.find(['Cameras', 'Main Camera']).object,
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
      const camera_main = findCameraObjectUnderObject3D(
        this.state.scene_tree.find(['Cameras', 'Main Camera']).object,
      );
      // console.log(camera_main.toJSON());
      const cmd = 'set_object';
      const path = 'Cameras/Main Camera';
      const data = {
        type: cmd,
        path,
        object: camera_main.toJSON(),
      };
      const message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  send_training_state_over_websocket(value) {
    if (this.state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_training_state';
      const path = 'Training State';
      const data = {
        type: cmd,
        path,
        training_state: value,
      };
      const message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  send_output_type_over_websocket(value) {
    /* update the output option in the python server
                            if the user changes selection */
    if (this.state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_output_type';
      const path = 'Output Type';
      const data = {
        type: cmd,
        path,
        output_type: value,
      };
      const message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  send_min_resolution_over_websocket(value) {
    if (this.state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_min_resolution';
      const path = 'Min Resolution';
      const data = {
        type: cmd,
        path,
        min_resolution: value,
      };
      const message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  send_max_resolution_over_websocket(value) {
    if (this.state.websocket.readyState === WebSocket.OPEN) {
      const cmd = 'set_max_resolution';
      const path = 'Max Resolution';
      const data = {
        type: cmd,
        path,
        max_resolution: value,
      };
      const message = msgpack.encode(data);
      this.state.websocket.send(message);
    }
  }

  set_object(path, object) {
    if (!(object instanceof THREE.Camera)) {
      this.state.scene_tree.find(path.concat(['<object>'])).set_object(object);
    }
  }

  set_object_from_json(path, object_json) {
    const loader = new ExtensibleObjectLoader();
    loader.parse(object_json, (obj) => {
      if (
        obj.geometry !== undefined &&
        obj.geometry.type === 'BufferGeometry'
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
      console.error('Deleting the entire scene is not implemented.');
    } else {
      this.state.scene_tree.delete(path);
    }
  }

  set_property(path, property, value) {
    this.state.scene_tree.find(path).set_property(property, value);
    // TODO(ethan): handle this issue
    if (path[0] === 'Background') {
      // The background is not an Object3d, so needs a little help.
      this.state.scene_tree.find(path).on_update();
    }
  }

  handle_command(cmd) {
    // convert binary serialization format back to JSON
    cmd = msgpack.decode(new Uint8Array(cmd));
    // console.log(cmd);

    // three js scene commands
    if (cmd.type === 'set_object') {
      const path = split_path(cmd.path);
      this.set_object_from_json(path, cmd.object);
    } else if (cmd.type === 'set_transform') {
      const path = split_path(cmd.path);
      this.set_transform(path, cmd.matrix);
    } else if (cmd.type === 'delete') {
      const path = split_path(cmd.path);
      this.delete_path(path);
    } else if (cmd.type === 'set_property') {
      const path = split_path(cmd.path);
      this.set_property(path, cmd.property, cmd.value);
    } else if (cmd.type == 'set_output_options') {
      // this.setOutputOptions(cmd.output_options);
      // this.props.setControls({ output_options: 1 });
    }
    // web rtc commands
    else if (cmd.type === 'answer') {
      const answer = cmd.data;
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
    console.log('setting up WebRTC');
    // The iceServers config comes from the following URL:
    // https://www.metered.ca/tools/openrelay/
    this.state.pc = new RTCPeerConnection({
      iceServers: [
        {
          urls: 'stun:openrelay.metered.ca:80',
        },
        {
          urls: 'turn:openrelay.metered.ca:80',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
        {
          urls: 'turn:openrelay.metered.ca:443',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
        {
          urls: 'turn:openrelay.metered.ca:443?transport=tcp',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
      ],
    });

    // connect video
    this.state.pc.addEventListener('track', (evt) => {
      if (evt.track.kind == 'video') {
        console.log('setting event stream');
        document.getElementById('WebRTCVideo-video').srcObject = evt.streams[0];
      }
    });
    this.state.pc.addTransceiver('video', { direction: 'recvonly' });

    this.state.pc
      .createOffer()
      .then((offer) => this.state.pc.setLocalDescription(offer))
      .then(
        () =>
          // wait for ICE gathering to complete
          new Promise((resolve) => {
            if (this.state.pc.iceGatheringState === 'complete') {
              resolve();
            } else {
              var checkState = () => {
                if (this.state.pc.iceGatheringState === 'complete') {
                  this.state.pc.removeEventListener(
                    'icegatheringstatechange',
                    checkState,
                  );
                  resolve();
                }
              };
              this.state.pc.addEventListener(
                'icegatheringstatechange',
                checkState,
              );
            }
          }),
      )
      .then(() => {
        const offer = this.state.pc.localDescription;
        console.log('sending the offer');
        const cmd = 'offer';
        const path = '';
        const data = {
          type: cmd,
          path,
          data: {
            sdp: offer.sdp,
            type: offer.type,
          },
        };
        const message = msgpack.encode(data);
        this.state.websocket.send(message);
      });
  }

  componentDidMount() {
    // Open the websocket
    const endpoint = websocket_endpoint_from_url(window.location.href);
    this.state.websocket = new WebSocket(`ws://${endpoint}/`);
    this.state.websocket.onopen = () => {
      console.log('websocket connected');
      this.setupWebRTC();
    };
    this.state.websocket.binaryType = 'arraybuffer';
    // 'command' for updates to Viewer
    this.state.websocket.onmessage = (cmd) => this.handle_command(cmd.data);
    this.state.websocket.onclose = function (evt) {
      console.log('onclose:', evt);
    };

    // Get size of screen
    this.state.viewport_width = this.getViewportWidth();
    this.state.viewport_height = this.getViewportHeight();

    // Scene
    this.state.scene = new THREE.Scene();

    // GUI
    this.state.gui = new GUI();
    const scene_folder = this.state.gui.addFolder('Scene');
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
      this.state.viewport_height,
    );
    this.state.renderer_main.domElement.style.border = '1px solid black';

    const domElement = document.getElementById('canvas-container-main');
    domElement.append(this.state.renderer_main.domElement);

    // Camera main
    this.state.camera_main = new THREE.PerspectiveCamera(
      120,
      this.state.viewport_width / this.state.viewport_height,
      0.01,
      100,
    );
    this.state.camera_main.position.x = 5;
    this.state.camera_main.position.y = -5;
    this.state.camera_main.position.z = 5;
    this.state.camera_main.up = new THREE.Vector3(0, 0, 1);

    this.state.controls_main = new TrackballControls(
      this.state.camera_main,
      this.state.renderer_main.domElement,
    );
    this.state.controls_main.rotateSpeed = 2.0;
    this.state.controls_main.zoomSpeed = 0.3;
    this.state.controls_main.panSpeed = 0.2;
    this.state.controls_main.target.set(0, 0, 0); // focus point of the controls
    this.state.controls_main.autoRotate = false;
    this.state.controls_main.enableDamping = true;
    this.state.controls_main.dampingFactor = 1.0;
    this.state.controls_main.update();

    const path = ['Cameras', 'Main Camera'];
    this.state.scene_tree
      .find(path.concat(['<object>']))
      .set_object(this.state.camera_main);

    // Axes display
    const axes = new THREE.AxesHelper(5);
    this.set_object(['Axes'], axes);

    // Grid
    const grid = new THREE.GridHelper(20, 20);
    grid.rotateX(Math.PI / 2); // now on xy plane
    this.set_object(['Grid'], grid);

    // Lights
    const color = 0xffffff;
    const intensity = 1;
    const light = new THREE.AmbientLight(color, intensity);
    this.set_object(['Light'], light);

    // Stats
    const stats = new Stats();
    const domElement_stats = document.getElementById('canvas-container-main');
    domElement_stats.append(stats.dom);

    this.update();
  }

  componentDidUpdate(prevProps, prevState) {
    // Pause training
    if (prevProps.paused !== this.props.paused) {
      this.send_training_state_over_websocket(this.props.paused);
    }

    // Choose render type
    if (prevProps.output_options !== this.props.output_options) {
      console.log(this.props.output_options);
      this.send_output_type_over_websocket(this.props.output_options);
    }

    // Set minimum render resolution
    if (prevProps.min_resolution !== this.props.min_resolution) {
      this.send_min_resolution_over_websocket(
        this.props.min_resolution.slice(0, -2),
      );
    }

    // Set maximum render resolution
    if (prevProps.max_resolution !== this.props.max_resolution) {
      this.send_max_resolution_over_websocket(
        this.props.max_resolution.slice(0, -2),
      );
    }
  }

  render() {
    return (
      <div>
        <div className="WebRTCVideo">
          <video id="WebRTCVideo-video" autoPlay playsInline muted />
        </div>
        <div id="canvas-container-main"> </div>
        <div id="stats-container"> </div>
      </div>
    );
  }
}
