import React, { Component } from "react";
import "./App.css";
import { Viewer } from "./Viewer";

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

  render() {
    return (
      <div className="App">
        <Viewer ref={this.viewer_ref} />
      </div>
    );
  }
}

export default App;
