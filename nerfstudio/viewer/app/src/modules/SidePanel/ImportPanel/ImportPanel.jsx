import * as React from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader';
import { useControls, useStoreContext } from 'leva';
import { useDispatch, useSelector } from 'react-redux';
import { LevaPanel, LevaStoreProvider, useCreateStore } from 'leva';
import LevaTheme from '../../../themes/leva_theme.json';
import FolderRoundedIcon from '@mui/icons-material/FolderRounded';
import {
  Text,
  Button,
  Input,
  InputLabel,
  FormControl,
  FormHelperText,
  List,
  ListItemText,
  ListItemButton,
  ListItemIcon,
} from '@mui/material';
import SceneNode from '../../../SceneNode';

interface OutlinerProps {
  sceneTree: SceneNode;
}

function Outliner(props: OutlinerProps) {
  const scene = props.sceneTree.object;
  const children = scene.children;

  return (
    <div className="Outliner">
      <List>
        {scene.children
          .filter((obj) => obj.name.indexOf('[USER]') > -1)
          .map((child, index) => (
            <ListItemButton key={index}>
              <ListItemIcon>
                <FolderRoundedIcon />
              </ListItemIcon>
              <ListItemText primary={child.name.replace('[USER]', '')} />
            </ListItemButton>
          ))}
      </List>
    </div>
  );
}

export default function ImportPanel(props) {
  const sceneTree = props.sceneTree;
  const [reload, setReload] = React.useState(0);

  const store = useStoreContext();
  const dispatch = useDispatch();

  const importPath = useSelector(
    (state) => state.file_path_info.export_path_name,
  );

  const fbxLoader = React.useMemo(() => new FBXLoader());
  const objLoader = React.useMemo(() => new OBJLoader());
  const stlLoader = React.useMemo(() => new STLLoader());

  const handleFileImport = React.useCallback(
    (event) => {
      event.stopPropagation();
      event.preventDefault();

      for (let file of event.target.files) {
        const reader = new FileReader();
        reader.addEventListener('loadend', (evt) => {
          const blob = reader.result;
          if (file.name.match(/.*\.fbx$/gi)) {
            const model = fbxLoader.parse(blob);
            // sceneTree.set_object(model);
          } else if (file.name.match(/.*\.obj$/gi)) {
            const geom = objLoader.parse(blob);
            const mat = new THREE.MeshPhongMaterial({ color: '#049ef4' });
            const mesh = new THREE.Mesh(geom, mat);
            mesh.name = '[USER]' + file.name;
            sceneTree.object.add(mesh);
          } else if (file.name.match(/.*\.stl/gi)) {
            const geom = stlLoader.parse(blob);
            const mat = new THREE.MeshPhongMaterial({
              color: '#049ef4',
              specular: '#111',
              shininess: 80,
            });
            const mesh = new THREE.Mesh(geom, mat);
            const name = file.name.replace(/.stl/i, '');
            mesh.name = '[USER]' + name;
            sceneTree.set_object_from_path([mesh.name], mesh);
            setReload((p) => p + 1);
          }
        });
        reader.readAsBinaryString(file);
      }
    },
    [sceneTree],
  );

  const importFileRef = React.createRef();
  const onClickImportFile = React.useCallback(() => {
    importFileRef.current.click();
  });

  // const [, setControls] = useControls(() => ({
  //   path: {
  //     label: '3D Model File',
  //     value: importPath,
  //     onChange: (v) => {
  //       setImportFile(v);
  //     },
  //   }
  // }));

  // setControls({ path: importPath });

  return (
    <div className="ImportPanel">
      <div className="ImportPanel-props">
        {/* <InputLabel id="">3D Model File</InputLabel>
        <Input type="file" name="file" /> */}
        <FormControl>
          <Button
            sx={{}}
            variant="outlined"
            size="medium"
            onClick={onClickImportFile}
          >
            Import 3D Model
          </Button>
          <input
            ref={importFileRef}
            type="file"
            name="3D Model File"
            accept=".obj, .stl, .fbx"
            onChange={handleFileImport}
            hidden
          />
        </FormControl>
        {Boolean(reload) && <Outliner sceneTree={sceneTree} />}
      </div>
    </div>
  );
}
