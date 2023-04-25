import argparse
import pickle

import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
import vtk

def point_cloud_to_voxel_grid(point_cloud, densities, voxel_size=0.1, buffer_size=1):
    """
    Convert a 3D point cloud to a voxel grid with densities at each voxel.
    Args:
    point_cloud: (N,3) numpy array giving the xyz coordinates of each point.
        densities: (N,) numpy array giving the densities of each point.
        voxel_size: The size of each voxel (in the same units as the point cloud).
        buffer_size: The number of voxels to add as a buffer around the point cloud.
    Returns: 
        A 3D numpy array giving the density at each xyz position.
    """
    # Find the extents of the point cloud.
    min_coords = np.min(point_cloud, axis=0) - buffer_size * voxel_size
    max_coords = np.max(point_cloud, axis=0) + buffer_size * voxel_size

    # Compute the number of voxels in each dimension.
    num_voxels = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    # Compute the voxel grid indices for each point.
    voxel_indices = np.floor((point_cloud - min_coords) / voxel_size).astype(int)
    
    # Create the voxel grid and accumulate the densities.
    voxel_grid = np.zeros(num_voxels)
    voxel_grid[tuple(voxel_indices.T)] = densities

    return voxel_grid

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog='densities_to_vti',
        description='Script for converting from the densities (e.g. generated with gerrys_eval.py) and convert them to a vti file that can be used in Paraview.',
    )
    parser.add_argument('--positions', type=str, required=True,
        help='Path to the pickle file containing the xyz coordinates of the points observed. \
            The file should contain a (n,3) numpy array.'
    )
    parser.add_argument('--densities', type=str, required=True,
        help='Path to the pickle file containing the densities of the points observed along each wavelength. \
            The file should contain a (n,ch) numpy array, with n the number of points and ch the number of density channels (i.e. number of wavelengths)'
    )
    parser.add_argument('--voxel-size', type=float, help='Voxel size of the grid', default=0.0125)
    
    parser.add_argument('--split-channels', action='store_true',
        help='If true, then one VTI file will be created for each density channel. Otherwise, a single global VTI file \
           will be created by computing average densities across channels.'
    )

    parser.add_argument('--output-basename', type=str,
        help='Base name of output VTI files. If none provided, the density filepath will be used as basename'
    )
    args = parser.parse_args()

    if args.output_basename:
        basename = args.output_basename
    else:
        # Use density filename as basename
        # basename = args.densities.split("/")[-1].split(".")[0]
        basename = args.densities.split(".")[0]

    # Read positions and densities
    with open(args.densities, 'rb') as handle:
        densities = pickle.load(handle)

    with open(args.positions, 'rb') as handle:
        positions = pickle.load(handle)
    
    # Average or not all channels
    if args.split_channels:
        densities = densities.mean(axis=-1, keepdims=True)

    # Convert all densities to voxel grid
    for ch in range(len(densities[0])):
        # if ch > 0:
        #     print("EQ ", np.all(np.abs(densities_ch - densities[:, ch]) < 1e-6))
        densities_ch = densities[:, ch]

        # Compute voxel grid
        voxel_grid = point_cloud_to_voxel_grid(positions, densities_ch, voxel_size=args.voxel_size)

        # Convert to VTI
        imdata = vtk.vtkImageData()
        depthArray = numpy_to_vtk(voxel_grid.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        imdata.SetDimensions(voxel_grid.shape)
        imdata.SetSpacing([1,1,1])
        imdata.SetOrigin([0,0,0])
        imdata.GetPointData().SetScalars(depthArray)

        vtk_f = vtk.vtkXMLImageDataWriter()
        vtk_f.SetInputData(imdata)

        # Save file
        if args.split_channels:
            output_path = basename + "_mean_chs.vti"
            print(f"Saving VTI with mean densities to {output_path}")
        else:
            output_path = basename + f"_{ch+1}ch.vti"
            print(f"Saving VTI for channel {ch+1} to {output_path}")

        vtk_f.SetFileName(output_path)
        vtk_f.Write()

