import React, { Component } from "react";

export class BackgroundDiv extends Component {
  componentDidMount() {
    document.getElementById("BackgroundDiv-video").play();
  }
  render() {
    return (
      <div className="BackgroundDiv">
        <video id="BackgroundDiv-video" autoPlay muted loop>
          <source
            src="https://www.w3schools.com/html/mov_bbb.mp4"
            type="video/mp4"
          ></source>
        </video>{" "}
      </div>
    );
  }
}
