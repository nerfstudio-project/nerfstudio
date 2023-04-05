import React from "react";
import { CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer";
import { useFrame } from "@react-three/fiber";

interface LabelRendererProps {
  wrapperRef: React.RefObject<HTMLDivElement>;
}

/** Component for rendering text labels on scene nodes. */
export default function LabelRenderer(props: LabelRendererProps) {
  const labelRenderer = new CSS2DRenderer();

  React.useEffect(() => {
    const wrapper = props.wrapperRef.current!;
    labelRenderer.domElement.style.overflow = "hidden";
    labelRenderer.domElement.style.position = "absolute";
    labelRenderer.domElement.style.pointerEvents = "none";
    labelRenderer.domElement.style.top = "0px";
    wrapper.appendChild(labelRenderer.domElement);

    function updateDimensions() {
      labelRenderer.setSize(wrapper.offsetWidth, wrapper.offsetHeight);
    }
    updateDimensions();

    window.addEventListener("resize", updateDimensions);
    return () => {
      window.removeEventListener("resize", updateDimensions);
    };
  });

  useFrame(({ scene, camera }) => {
    labelRenderer.render(scene, camera);
  });
  return <></>;
}
