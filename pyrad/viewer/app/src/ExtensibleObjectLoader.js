import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";
import { MTLLoader } from "three/examples/jsm/loaders/MTLLoader.js";
import { MtlObjBridge } from "wwobjloader2";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils.js";

// Merges a hierarchy of collada mesh geometries into a single
// `BufferGeometry` object:
//   * A new merged `BufferGeometry` if the input contains meshes
//   * empty `BufferGeometry` otherwise
function merge_geometries(object, preserve_materials = false) {
    let materials = [];
    let geometries = [];
    let root_transform = object.matrix.clone();

    function collectGeometries(node, parent_transform) {
        let transform = parent_transform.clone().multiply(node.matrix);
        if (node.type === "Mesh") {
            node.geometry.applyMatrix(transform);
            geometries.push(node.geometry);
            materials.push(node.material);
        }
        for (let child of node.children) {
            collectGeometries(child, transform);
        }
    }

    collectGeometries(object, root_transform);
    let result = null;
    if (geometries.length === 1) {
        result = geometries[0];
        if (preserve_materials) {
            result.material = materials[0];
        }
    } else if (geometries.length > 1) {
        result = BufferGeometryUtils.mergeBufferGeometries(geometries, true);
        if (preserve_materials) {
            result.material = materials;
        }
    } else {
        result = new THREE.BufferGeometry();
    }
    return result;
}

// Handler for special texture types that we want to support
// in addition to whatever three.js supports. This function
// takes a json object representing a single texture, and should
// return either:
//   * A new `THREE.Texture` if that json represents a special texture
//   * `null` otherwise
function handle_special_texture(json) {
    if (json.type === "_text") {
        let canvas = document.createElement("canvas");
        // canvas width and height should be in the power of 2; otherwise although
        // the page usually loads successfully, WebGL does complain/warn
        canvas.width = 256;
        canvas.height = 256;
        let ctx = canvas.getContext("2d");
        ctx.textAlign = "center";
        let font_size = json.font_size;
        // auto-resing the font_size to fit in the canvas
        ctx.font = font_size + "px " + json.font_face;
        while (ctx.measureText(json.text).width > canvas.width) {
            font_size--;
            ctx.font = font_size + "px " + json.font_face;
        }
        ctx.fillText(json.text, canvas.width / 2, canvas.height / 2);
        let canvas_texture = new THREE.CanvasTexture(canvas);
        canvas_texture.uuid = json.uuid;
        return canvas_texture;
    } else {
        return null;
    }
}

// Handler for special geometry types that we want to support
// in addition to whatever three.js supports. This function
// takes a json object representing a single geometry, and should
// return either:
//   * A new `THREE.Mesh` if that json represents a special geometry
//   * `null` otherwise
function handle_special_geometry(geom) {
    if (geom.type === "_meshfile") {
        console.warn(
            "_meshfile is deprecated. Please use _meshfile_geometry for geometries and _meshfile_object for objects with geometry and material"
        );
        geom.type = "_meshfile_geometry";
    }
    if (geom.type === "_meshfile_geometry") {
        if (geom.format === "obj") {
            let loader = new OBJLoader();
            let obj = loader.parse(geom.data + "\n");
            let loaded_geom = merge_geometries(obj);
            loaded_geom.uuid = geom.uuid;
            return loaded_geom;
        } else if (geom.format === "dae") {
            let loader = new ColladaLoader();
            let obj = loader.parse(geom.data);
            let result = merge_geometries(obj.scene);
            result.uuid = geom.uuid;
            return result;
        } else if (geom.format === "stl") {
            let loader = new STLLoader();
            let loaded_geom = loader.parse(geom.data.buffer);
            loaded_geom.uuid = geom.uuid;
            return loaded_geom;
        } else if (geom.format === "ply") {
            let loader = new PLYLoader();
            let loaded_geom = loader.parse(geom.data.buffer);
            loaded_geom.uuid = geom.uuid;
            return loaded_geom;
        } else {
            console.error("Unsupported mesh type:", geom);
            return null;
        }
    }
    return null;
}

// The ExtensibleObjectLoader extends the THREE.ObjectLoader
// interface, while providing some hooks for us to perform some
// custom loading for things other than three.js native JSON.
//
// We currently use this class to support some extensions to
// three.js JSON for objects which are easy to construct in
// javascript but hard to construct in Python and/or Julia.
// For example, we perform the following transformations:
//
//   * Converting "_meshfile" geometries into actual meshes
//     using the THREE.js native mesh loaders
//   * Converting "_text" textures into text by drawing the
//     requested text onto a canvas.
export class ExtensibleObjectLoader extends THREE.ObjectLoader {
    delegate(special_handler, base_handler, json, additional_objects) {
        let result = {};
        if (json === undefined) {
            return result;
        }
        let remaining_json = [];
        for (let data of json) {
            let x = special_handler(data);
            if (x !== null) {
                result[x.uuid] = x;
            } else {
                remaining_json.push(data);
            }
        }
        return Object.assign(
            result,
            base_handler(remaining_json, additional_objects)
        );
    }

    parseTextures(json, images) {
        return this.delegate(
            handle_special_texture,
            super.parseTextures,
            json,
            images
        );
    }

    parseGeometries(json, shapes) {
        return this.delegate(
            handle_special_geometry,
            super.parseGeometries,
            json,
            shapes
        );
    }

    parseObject(json, geometries, materials) {
        if (json.type === "_meshfile_object") {
            let geometry;
            let material;
            let manager = new THREE.LoadingManager();
            let path =
                json.url === undefined ?
                undefined :
                THREE.LoaderUtils.extractUrlBase(json.url);
            manager.setURLModifier((url) => {
                if (json.resources[url] !== undefined) {
                    return json.resources[url];
                }
                return url;
            });
            if (json.format === "obj") {
                let loader = new OBJLoader(manager);
                if (json.mtl_library) {
                    let mtl_loader = new MTLLoader(manager);
                    let mtl_parse_result = mtl_loader.parse(json.mtl_library + "\n", "");
                    let materials =
                        MtlObjBridge.addMaterialsFromMtlLoader(mtl_parse_result);
                    loader.setMaterials(materials);
                    this.onTextureLoad();
                }
                let obj = loader.parse(json.data + "\n", path);
                geometry = merge_geometries(obj, true);
                geometry.uuid = json.uuid;
                material = geometry.material;
            } else if (json.format === "dae") {
                let loader = new ColladaLoader(manager);
                loader.onTextureLoad = this.onTextureLoad;
                let obj = loader.parse(json.data, path);
                geometry = merge_geometries(obj.scene, true);
                geometry.uuid = json.uuid;
                material = geometry.material;
            } else if (json.format === "stl") {
                let loader = new STLLoader();
                geometry = loader.parse(json.data.buffer, path);
                geometry.uuid = json.uuid;
                material = geometry.material;
            } else {
                console.error("Unsupported mesh type:", json);
                return null;
            }
            let object = new THREE.Mesh(geometry, material);

            // Copied from ObjectLoader
            object.uuid = json.uuid;

            if (json.name !== undefined) object.name = json.name;

            if (json.matrix !== undefined) {
                object.matrix.fromArray(json.matrix);

                if (json.matrixAutoUpdate !== undefined)
                    object.matrixAutoUpdate = json.matrixAutoUpdate;
                if (object.matrixAutoUpdate)
                    object.matrix.decompose(
                        object.position,
                        object.quaternion,
                        object.scale
                    );
            } else {
                if (json.position !== undefined)
                    object.position.fromArray(json.position);
                if (json.rotation !== undefined)
                    object.rotation.fromArray(json.rotation);
                if (json.quaternion !== undefined)
                    object.quaternion.fromArray(json.quaternion);
                if (json.scale !== undefined) object.scale.fromArray(json.scale);
            }

            if (json.castShadow !== undefined) object.castShadow = json.castShadow;
            if (json.receiveShadow !== undefined)
                object.receiveShadow = json.receiveShadow;

            if (json.shadow) {
                if (json.shadow.bias !== undefined)
                    object.shadow.bias = json.shadow.bias;
                if (json.shadow.radius !== undefined)
                    object.shadow.radius = json.shadow.radius;
                if (json.shadow.mapSize !== undefined)
                    object.shadow.mapSize.fromArray(json.shadow.mapSize);
                if (json.shadow.camera !== undefined)
                    object.shadow.camera = this.parseObject(json.shadow.camera);
            }

            if (json.visible !== undefined) object.visible = json.visible;
            if (json.frustumCulled !== undefined)
                object.frustumCulled = json.frustumCulled;
            if (json.renderOrder !== undefined) object.renderOrder = json.renderOrder;
            if (json.userjson !== undefined) object.userjson = json.userData;
            if (json.layers !== undefined) object.layers.mask = json.layers;

            return object;
        } else if (json.type === "CameraHelper") {
            console.log("processing CameraHelper");
            console.log(json);
            console.log(geometries);
            console.log(materials);
            return super.parseObject(json, geometries, materials);
        } else {
            return super.parseObject(json, geometries, materials);
        }
    }
}