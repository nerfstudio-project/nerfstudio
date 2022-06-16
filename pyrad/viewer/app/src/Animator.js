import * as THREE from 'three';
import CCapture from 'ccapture.js'; //TODO(ethan): check if this is correct
const dat = require('dat.gui');

function remove_folders(gui) {
    for (let name of Object.keys(gui.__folders)) {
        let folder = gui.__folders[name];
        remove_folders(folder);
        dat.dom.dom.unbind(window, 'resize', folder.__resizeHandler);
        gui.removeFolder(folder);
    }
}

export class Animator {
    constructor(viewer) {
        this.viewer = viewer;
        this.folder = this.viewer.gui.addFolder("Animations");
        this.mixer = new THREE.AnimationMixer();
        this.loader = new THREE.ObjectLoader();
        this.clock = new THREE.Clock();
        this.actions = [];
        this.playing = false;
        this.time = 0;
        this.time_scrubber = null;
        this.setup_capturer("png");
        this.duration = 0;
    }

    setup_capturer(format) {
        this.capturer = new CCapture({
            format: format,
            name: "pyrad_" + String(Date.now())
        });
        this.capturer.format = format;
    }

    play() {
        this.clock.start();
        // this.mixer.timeScale = 1;
        for (let action of this.actions) {
            action.play();
        }
        this.playing = true;
    }

    record() {
        this.reset();
        this.play();
        this.recording = true;
        this.capturer.start();
    }

    pause() {
        // this.mixer.timeScale = 0;
        this.clock.stop();
        this.playing = false;

        if (this.recording) {
            this.stop_capture();
            this.save_capture();
        }
    }

    stop_capture() {
        this.recording = false;
        this.capturer.stop();
        this.viewer.animate(); // restore the animation loop which gets disabled by capturer.stop()
    }

    save_capture() {
        this.capturer.save();
        if (this.capturer.format === "png") {
            alert("To convert the still frames into a video, extract the `.tar` file and run: \nffmpeg -r 60 -i %07d.png \\\n\t -vcodec libx264 \\\n\t -preset slow \\\n\t -crf 18 \\\n\t output.mp4");
        } else if (this.capturer.format === "jpg") {
            alert("To convert the still frames into a video, extract the `.tar` file and run: \nffmpeg -r 60 -i %07d.jpg \\\n\t -vcodec libx264 \\\n\t -preset slow \\\n\t -crf 18 \\\n\t output.mp4");
        }
    }

    display_progress(time) {
        this.time = time;
        if (this.time_scrubber !== null) {
            this.time_scrubber.updateDisplay();
        }
    }

    seek(time) {
        this.actions.forEach((action) => {
            action.time = Math.max(0, Math.min(action._clip.duration, time));
        });
        this.mixer.update(0);
        this.viewer.set_dirty();
    }

    reset() {
        for (let action of this.actions) {
            action.reset();
        }
        this.display_progress(0);
        this.mixer.update(0);
        this.setup_capturer(this.capturer.format);
        this.viewer.set_dirty();
    }

    clear() {
        remove_folders(this.folder);
        this.mixer.stopAllAction();
        this.actions = [];
        this.duration = 0;
        this.display_progress(0);
        this.mixer = new THREE.AnimationMixer();
    }

    load(animations, options) {
        this.clear();

        this.folder.open();
        let folder = this.folder.addFolder("default");
        folder.open();
        folder.add(this, "play");
        folder.add(this, "pause");
        folder.add(this, "reset");

        // Note, for some reason when you call `.max()` on a slider controller it does
        // correctly change how the slider behaves but does not change the range of values
        // that can be entered into the text box attached to the slider. Oh well. We work
        // around this by creating the slider with an unreasonably huge range and then calling
        // `.min()` and `.max()` on it later.
        this.time_scrubber = folder.add(this, "time", 0, 1e9, 0.001);
        this.time_scrubber.onChange((value) => this.seek(value));

        folder.add(this.mixer, "timeScale").step(0.01).min(0);
        let recording_folder = folder.addFolder("Recording");
        recording_folder.add(this, "record");
        recording_folder.add({format: "png"}, "format", ["png", "jpg"]).onChange(value => {
            this.setup_capturer(value);
        });


        if (options.play === undefined) {
            options.play = true
        }
        if (options.loopMode === undefined) {
            options.loopMode = THREE.LoopRepeat
        }
        if (options.repetitions === undefined) {
            options.repetitions = 1
        }
        if (options.clampWhenFinished === undefined) {
            options.clampWhenFinished = true
        }

        this.duration = 0;
        this.progress = 0;
        for (let animation of animations) {
            let target = this.viewer.scene_tree.find(animation.path).object;
            let clip = this.loader.parseAnimations([animation.clip])[0];
            let action = this.mixer.clipAction(clip, target);
            action.clampWhenFinished = options.clampWhenFinished;
            action.setLoop(options.loopMode, options.repetitions);
            this.actions.push(action);
            this.duration = Math.max(this.duration, clip.duration);
        }
        this.time_scrubber.min(0);
        this.time_scrubber.max(this.duration);
        this.reset();
        if (options.play) {
            this.play();
        }
    }

    update() {
        if (this.playing) {
            this.mixer.update(this.clock.getDelta());
            this.viewer.set_dirty();
            if (this.duration != 0) {
                let current_time = this.actions.reduce((acc, action) => {
                    return Math.max(acc, action.time);
                }, 0);
                this.display_progress(current_time);
            } else {
                this.display_progress(0);
            }

            if (this.actions.every((action) => action.paused)) {
                this.pause();
                for (let action of this.actions) {
                    action.reset();
                }
            }
        }
    }

    after_render() {
        if (this.recording) {
            this.capturer.capture(this.viewer.renderer.domElement);
        }
    }
}

// Generates a gradient texture without filling up
// an entire canvas. We simply create a 2x1 image
// containing only the two colored pixels and then
// set up the appropriate magnification and wrapping
// modes to generate the gradient automatically
function gradient_texture(top_color, bottom_color) {
    let colors = [bottom_color, top_color];

    let width = 1;
    let height = 2;
    let size = width * height;
    var data = new Uint8Array(3 * size);
    for (let row = 0; row < height; row++) {
        let color = colors[row];
        for (let col = 0; col < width; col++) {
            let i = 3 * (row * width + col);
            for (let j = 0; j < 3; j++) {
                data[i + j] = color[j];
            }
        }
    }
    var texture = new THREE.DataTexture(data, width, height, THREE.RGBFormat);
    texture.magFilter = THREE.LinearFilter;
    texture.encoding = THREE.LinearEncoding;
    // By default, the points in our texture map to the center of
    // the pixels, which means that the gradient only occupies
    // the middle half of the screen. To get around that, we just have
    // to tweak the UV transform matrix
    texture.matrixAutoUpdate = false;
    texture.matrix.set(0.5, 0, 0.25,
        0, 0.5, 0.25,
        0, 0, 1);
    texture.needsUpdate = true
    return texture;
}