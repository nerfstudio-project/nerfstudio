import React, { Component } from "react";
import "./App.css";
import { Viewer } from "./Viewer";

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

class Image extends Component {
  constructor(props) {
    super(props);
    this.state = {};
  }
}

class ColorGUIHelper {
  constructor(object, prop) {
    this.object = object;
    this.prop = prop;
  }

  get value() {
    return `#${this.object[this.prop].getHexString()}`;
  }

  set value(hexString) {
    this.object[this.prop].set(hexString);
  }
}

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      client: null,
    };
    this.viewer_ref = React.createRef();
  }

  componentDidMount() {
    let endpoint = websocket_endpoint_from_url(window.location.href);
    this.state.client = new WebSocket("ws://" + endpoint + "/");
    this.state.client.binaryType = "arraybuffer";
    // 'command' for updates to Viewer
    this.state.client.onmessage = (cmd) =>
      this.viewer_ref.current.handle_command(cmd.data);
    this.state.client.onclose = function (evt) {
      console.log("onclose:", evt);
    };
  }

  render() {
    return (
      <div className="App">
        <Viewer ref={this.viewer_ref} />
      </div>
    );
  }
}

export default App;
