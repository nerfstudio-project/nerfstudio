import * as React from 'react';
import { useSelector } from 'react-redux';

export default function RenderWindow() {
  const render_img = useSelector((state) => state.render_img);

  return (
    <div className="RenderWindow">
      <img src={render_img} width="100%" height="100%" alt="Render window" />
    </div>
  );
}
