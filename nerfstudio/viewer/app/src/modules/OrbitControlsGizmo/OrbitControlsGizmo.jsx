/**
 * Copyright https://github.com/fennec-hub/ThreeOrbitControlsGizmo
 */
import {
  Vector2,
  Vector3,
  Matrix4,
} from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';

class OrbitControlsGizmo {
  constructor(orbitControls, options) {
    options = Object.assign(
      {
        size: 90,
        padding: 8,
        bubbleSizePrimary: 8,
        bubbleSizeSecondary: 6,
        lineWidth: 2,
        fontSize: '12px',
        fontFamily: 'arial',
        fontWeight: 'bold',
        fontColor: '#222222',
        className: 'obit-controls-gizmo',
        colors: {
          x: ['#f73c3c', '#942424'],
          y: ['#6ccb26', '#417a17'],
          z: ['#178cf0', '#0e5490'],
        },
      },
      options,
    );

    this.lock = false;
    this.lockX = false;
    this.lockY = false;

    this.update = () => {
      if (this.lock) return;

      camera.updateMatrix();
      invRotMat.extractRotation(camera.matrix).invert();

      for (let i = 0, length = axes.length; i < length; i++)
        setAxisPosition(axes[i], invRotMat);

      // Sort the layers where the +Z position is last so its drawn on top of anything below it
      axes.sort((a, b) => (a.position.z > b.position.z ? 1 : -1));

      // Draw the layers
      drawLayers(true);
    };

    this.dispose = () => {
      orbit.removeEventListener('change', this.update);
      orbit.removeEventListener('start', () =>
        this.domElement.classList.add('inactive'),
      );
      orbit.removeEventListener('end', () =>
        this.domElement.classList.remove('inactive'),
      );

      this.domElement.removeEventListener('pointerdown', onPointerDown, false);
      this.domElement.removeEventListener(
        'pointerenter',
        onPointerEnter,
        false,
      );
      this.domElement.removeEventListener('pointermove', onPointerMove, false);
      this.domElement.removeEventListener('click', onMouseClick, false);
      window.removeEventListener('pointermove', onDrag, false);
      window.removeEventListener('pointerup', onPointerUp, false);
      this.domElement.remove();
    };

    // Internals
    const scoped = this;
    const orbit = orbitControls;
    const camera = orbitControls.object;
    const invRotMat = new Matrix4();
    const mouse = new Vector3();
    const rotateStart = new Vector2();
    const rotateEnd = new Vector2();
    const rotateDelta = new Vector2();
    const center = new Vector3(options.size / 2, options.size / 2, 0);
    const axes = createAxes();
    let selectedAxis = null;
    let isDragging = false;
    let context;
    let rect;
    let orbitState;

    orbit.addEventListener('change', this.update);
    orbit.addEventListener('start', () =>
      this.domElement.classList.add('inactive'),
    );
    orbit.addEventListener('end', () =>
      this.domElement.classList.remove('inactive'),
    );

    function createAxes() {
      // Generate list of axes
      const colors = options.colors;
      const line = options.lineWidth;
      const size = {
        primary: options.bubbleSizePrimary,
        secondary: options.bubbleSizeSecondary,
      };
      return [
        {
          axis: 'x',
          direction: new Vector3(1, 0, 0),
          size: size.primary,
          color: colors.x,
          line,
          label: 'X',
          position: new Vector3(0, 0, 0),
        },
        {
          axis: 'y',
          direction: new Vector3(0, 1, 0),
          size: size.primary,
          color: colors.y,
          line,
          label: 'Y',
          position: new Vector3(0, 0, 0),
        },
        {
          axis: 'z',
          direction: new Vector3(0, 0, 1),
          size: size.primary,
          color: colors.z,
          line,
          label: 'Z',
          position: new Vector3(0, 0, 0),
        },
        {
          axis: '-x',
          direction: new Vector3(-1, 0, 0),
          size: size.secondary,
          color: colors.x,
          position: new Vector3(0, 0, 0),
        },
        {
          axis: '-y',
          direction: new Vector3(0, -1, 0),
          size: size.secondary,
          color: colors.y,
          position: new Vector3(0, 0, 0),
        },
        {
          axis: '-z',
          direction: new Vector3(0, 0, -1),
          size: size.secondary,
          color: colors.z,
          position: new Vector3(0, 0, 0),
        },
      ];
    }

    function createCanvas() {
      const canvas = document.createElement('canvas');
      canvas.width = options.size;
      canvas.height = options.size;
      canvas.classList.add(options.className);

      canvas.addEventListener('pointerdown', onPointerDown, false);
      canvas.addEventListener('pointerenter', onPointerEnter, false);
      canvas.addEventListener('pointermove', onPointerMove, false);
      canvas.addEventListener('click', onMouseClick, false);

      context = canvas.getContext('2d');

      return canvas;
    }

    function onPointerDown(e) {
      rotateStart.set(e.clientX, e.clientY);
      orbitState = orbit.enabled;
      orbit.enabled = false;
      window.addEventListener('pointermove', onDrag, false);
      window.addEventListener('pointerup', onPointerUp, false);
    }

    function onPointerUp() {
      setTimeout(() => (isDragging = false), 0);
      scoped.domElement.classList.remove('dragging');
      orbit.enabled = orbitState;
      window.removeEventListener('pointermove', onDrag, false);
      window.removeEventListener('pointerup', onPointerUp, false);
    }

    function onPointerEnter() {
      rect = scoped.domElement.getBoundingClientRect();
    }

    function onPointerMove(e) {
      if (isDragging || scoped.lock) return;

      const currentAxis = selectedAxis;

      selectedAxis = null;
      if (e) mouse.set(e.clientX - rect.left, e.clientY - rect.top, 0);

      // Loop through each layer
      for (let i = 0, length = axes.length; i < length; i++) {
        const distance = mouse.distanceTo(axes[i].position);

        if (distance < axes[i].size) selectedAxis = axes[i];
      }

      if (currentAxis !== selectedAxis) drawLayers();
    }

    function onDrag(e) {
      if (scoped.lock) return;

      if (!isDragging) scoped.domElement.classList.add('dragging');

      isDragging = true;

      selectedAxis = null;

      rotateEnd.set(e.clientX, e.clientY);

      rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(0.5);

      if (!scoped.lockX)
        orbit.rotateLeft(
          (2 * Math.PI * rotateDelta.x) / scoped.domElement.height,
        );

      if (!scoped.lockY)
        orbit.rotateUp(
          (2 * Math.PI * rotateDelta.y) / scoped.domElement.height,
        );

      rotateStart.copy(rotateEnd);

      orbit.update();
    }

    function onMouseClick() {
      //FIXME Don't like the current animation
      if (isDragging || !selectedAxis) return;

      const vec = selectedAxis.direction.clone();
      const distance = camera.position.distanceTo(orbit.target);
      vec.multiplyScalar(distance);

      const duration = 400;
      const start = performance.now();
      const maxAlpha = 1;
      function loop() {
        const now = performance.now();
        const delta = now - start;
        const alpha = Math.min(delta / duration, maxAlpha);
        camera.position.lerp(vec, alpha);
        orbit.update();

        if (alpha !== maxAlpha) return requestAnimationFrame(loop);

        onPointerMove();
      }

      loop();

      selectedAxis = null;
    }

    function drawCircle(p, radius = 10, color = '#FF0000') {
      context.beginPath();
      context.arc(p.x, p.y, radius, 0, 2 * Math.PI, false);
      context.fillStyle = color;
      context.fill();
      context.closePath();
    }

    function drawLine(p1, p2, width = 1, color = '#FF0000') {
      context.beginPath();
      context.moveTo(p1.x, p1.y);
      context.lineTo(p2.x, p2.y);
      context.lineWidth = width;
      context.strokeStyle = color;
      context.stroke();
      context.closePath();
    }

    function drawLayers(clear) {
      if (clear)
        context.clearRect(
          0,
          0,
          scoped.domElement.width,
          scoped.domElement.height,
        );

      // For each layer, draw the axis
      for (let i = 0, length = axes.length; i < length; i++) {
        const axis = axes[i];

        // Set the color
        const highlight = selectedAxis === axis;
        const color = axis.position.z >= -0.01 ? axis.color[0] : axis.color[1];

        // Draw the line that connects it to the center if enabled
        if (axis.line) drawLine(center, axis.position, axis.line, color);

        // Draw the circle for the axis
        drawCircle(axis.position, axis.size, highlight ? '#FFFFFF' : color);

        // Write the axis label (X,Y,Z) if provided
        if (axis.label) {
          context.font = [
            options.fontWeight,
            options.fontSize,
            options.fontFamily,
          ].join(' ');
          context.fillStyle = options.fontColor;
          context.textBaseline = 'middle';
          context.textAlign = 'center';
          context.fillText(axis.label, axis.position.x, axis.position.y);
        }
      }
    }

    function setAxisPosition(axis) {
      const position = axis.direction.clone().applyMatrix4(invRotMat);
      const size = axis.size;
      axis.position.set(
        position.x * (center.x - size / 2 - options.padding) + center.x,
        center.y - position.y * (center.y - size / 2 - options.padding),
        position.z,
      );
    }

    // Initialization
    this.domElement = createCanvas();
    this.update();
  }
}

export { OrbitControlsGizmo };
