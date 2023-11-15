# Script for generating an EPS file for Google Earth Studio image capturing
# Input: Lat, Long, Alt of the region of interest
# Output: EPS file with the required parameters for capturing images

import copy
import json
import os

import numpy as np


def geodetic2ecef(geodetic_coords):
    lat = geodetic_coords[0]
    lon = geodetic_coords[1]
    alt = geodetic_coords[2]
    a = 6378137.0
    b = 6356752.314245
    f = (a - b) / a
    e_sq = f * (2-f)
    phi = np.deg2rad(lat)
    lambd = np.deg2rad(lon)
    N = a / np.sqrt(1 - e_sq*np.sin(phi)**2)
    x = (N+alt) * np.cos(phi) * np.cos(lambd)
    y = (N+alt) * np.cos(phi) * np.sin(lambd)
    z = ((b**2 / a**2) * N + alt) * np.sin(phi)
    return np.array([x, y, z])

def ges_to_lla(x,y,z):
  alt_slope =  65130844.36517137
  alt_intercept =  0.5924511869714024
  return (x*180 - 90), (y*360 - 180), (z*alt_slope + alt_intercept)

def ges_to_ecef(x,y,z):
  lat, lon, alt = ges_to_lla(x,y,z)
  ecef = geodetic2ecef(np.array([lat, lon, alt]))
  return ecef

def lla_to_ges(x,y,z):
  alt_slope =  65130844.36517137
  alt_intercept =  0.5924511869714024
  return (x+90)/180, (y+180)/360, (z-alt_intercept)/alt_slope


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help='output file path (EPS)')
    parser.add_argument("--template", type=str, default='"Templates/Transamerica.json"', help='template name (JSON)')
    parser.add_argument("lat", type=float, help='lat of ROI')
    parser.add_argument("lon", type=float, help='lon of ROI')
    parser.add_argument("alt", type=float, help='alt of ROI')
    
    return parser
    

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    lat = args.lat
    lon = args.lon
    alt = args.alt
    
    lat_ges, lon_ges, alt_ges = lla_to_ges(lat,lon,alt)

    # Load template JSON file
    print("Loading template...", args.template)
    with open(args.template, 'r') as f:
        template_json = json.load(f)['scenes'][0]

    new_trajectory = copy.deepcopy(template_json)
    print("Done.")
    
    # SETTING CAMERA TARGET
    def set_camera_target(new_trajectory, target_coords, target_idx):
        new_trajectory['attributes'][0]['attributes'][1]['attributes'][1]['attributes'][target_idx]['value']['relative'] = target_coords
        new_trajectory['attributes'][0]['attributes'][1]['attributes'][1]['attributes'][target_idx]['keyframes'][0]['value'] = target_coords

    set_camera_target(new_trajectory, lon_ges, 0)
    set_camera_target(new_trajectory, lat_ges, 1)
    set_camera_target(new_trajectory, alt_ges, 2)

    #SETTING TRACKPOINTS - EMPTY LIST
    new_trajectory['cameraExport']['trackPoints'] = []

    #GET TIMES
    times = [keyframe['time'] for keyframe in template_json['attributes'][0]['attributes'][0]['attributes'][0]['attributes'][0]['keyframes']]

    num_steps=len(template_json['attributes'][0]['attributes'][0]['attributes'][0]['attributes'][0]['keyframes'])

    horizontal_scale = 0.2 * (alt / 100)
    vertical_scale = 0.3 * (alt / 100)
    delta_scale = 0.65

    def update_keyframes(template_json, new_trajectory, attribute_idx, ges_value, ges_centerpoint, delta_scale, ges_scale):
        # Get the keyframes from the template_json object
        keyframes = [keyframe['value'] for keyframe in template_json['attributes'][0]['attributes'][0]['attributes'][0]['attributes'][attribute_idx]['keyframes']]

        # Calculate the deltas based on the centerpoint and delta_scale
        deltas = [(x - ges_centerpoint) * delta_scale ** (i // 5) for i, x in enumerate(keyframes)]

        # Calculate the new keyframes by adding the ges_value and scaling the deltas
        new_keyframes = np.array([ges_value] * len(keyframes)) + np.array(deltas) * ges_scale

        # Update the keyframes in the new_trajectory object
        for i in range(len(keyframes)):
            new_trajectory['attributes'][0]['attributes'][0]['attributes'][0]['attributes'][attribute_idx]['keyframes'][i]['value'] = new_keyframes[i]

        # Update the relative value in the new_trajectory object
        new_trajectory['attributes'][0]['attributes'][0]['attributes'][0]['attributes'][attribute_idx]['value']['relative'] = new_trajectory['attributes'][0]['attributes'][0]['attributes'][0]['attributes'][attribute_idx]['keyframes'][-1]['value']

    alt_centerpoint=template_json['attributes'][0]['attributes'][1]['attributes'][1]['attributes'][2]['value']['relative']
    lat_centerpoint=template_json['attributes'][0]['attributes'][1]['attributes'][1]['attributes'][1]['value']['relative']
    long_centerpoint=template_json['attributes'][0]['attributes'][1]['attributes'][1]['attributes'][0]['value']['relative']

    update_keyframes(template_json, new_trajectory, 0, lon_ges, long_centerpoint, delta_scale, horizontal_scale)
    update_keyframes(template_json, new_trajectory, 1, lat_ges, lat_centerpoint, delta_scale, horizontal_scale)
    update_keyframes(template_json, new_trajectory, 2, alt_ges, alt_centerpoint, delta_scale, vertical_scale)


    template_json_full = json.load(open(args.template, 'r'))
    output_json = copy.deepcopy(template_json_full)
    output_json['scenes'][0]=new_trajectory

    print("Writing file...", args.filepath)
    with open(args.filepath, "w") as outfile:
        json.dump(output_json, outfile)
    print("Done.")
