# USE: matches gps coordinates in gps_and_speed.txt with image names in the same folder
# The intent is to generate a CSV of matches that can be converted into a KMZ file for import into GIS systems.
import sys
import pandas as pd
import math
import io
import os
import shutil
from PIL import Image
import shapefile
from tqdm import tqdm
import numpy as np
from yaspin import yaspin
import time

project_folder = "./"
source_extract_folder = "data/original"
target_matches_folder = "data/extracted"
customer_shapefile_folder = "data/original"

threshold_m = 0.1
fixed_asset_distance = 1
perpendicular_dist = 1

SourceImageFolder = project_folder + source_extract_folder
TargetImageFolder = project_folder + target_matches_folder
CustomerAssetFolder = customer_shapefile_folder

THRESHOLD = threshold_m
FIXED_ASSET_DIST = fixed_asset_distance

# How much distance to consider a pole is on the road (reachable by vehicle)
PERPENDICULAR_DIST = perpendicular_dist

if not os.path.exists(TargetImageFolder):
    os.mkdir(TargetImageFolder)

names_to_search = {}

# Create detections dataframe
detections = pd.DataFrame(columns=['image', 'lat', 'long'])

# Finds images in folder and stores partial filenames in names_to_search
for root, dirs, files in os.walk((os.path.normpath(SourceImageFolder)), topdown=False):
    #sp = yaspin(text='Collecting {} images'.format(len(files)), color="cyan")
    #sp.start()
    for name in files:        
        if name.endswith('jpg'):
            im = Image.open(os.path.join(SourceImageFolder, name))
            if im.size[0] != 640:
                split_name = name.split('_')
                images = names_to_search.get(split_name[0], [])
                images.append((split_name[1], name))
                names_to_search[split_name[0]] = images   
    #sp.stop() 

kEarthRadius = 6371000
kMathPI = 3.14159265359
kDeg2RadFactor = kMathPI / 180


def deg2rad(deg):
    return deg * kDeg2RadFactor


def rad2deg(rad):
    return rad / kDeg2RadFactor


def destination_from_heading(lat, long, distance, heading):
    ang_distance = distance / kEarthRadius
    new_lat = math.asin(math.sin(lat) * math.cos(ang_distance) + math.cos(lat) * math.sin(ang_distance) * math.cos(heading))
    new_long = long + math.atan2(math.cos(lat) * math.sin(ang_distance) * math.sin(heading), math.cos(ang_distance) - math.sin(lat) * math.sin(new_lat))
    return new_lat, new_long


def bearing(lat1, long1, lat2, long2):
    return math.atan2(math.sin(long2 - long1) * math.cos(lat2), math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(long2 - long1))


def distance(lat1, long1, lat2, long2):
    lat = (lat2-lat1)
    long = (long2-long1)

    a = math.sin(lat/2) * math.sin(lat/2) + math.cos(lat1) * math.cos(lat2) * math.sin(long/2) * math.sin(long/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return kEarthRadius * c

def normalize_angle(angle):
    if angle < 0:
        angle += 360
    if angle >= 360:
        angle -= 360
    return angle


def is_on_right(veh_to_asset_angle, vehicle_bearing):
    veh_to_asset_angle = normalize_angle(rad2deg(veh_to_asset_angle))
    vehicle_bearing = normalize_angle(rad2deg(vehicle_bearing))
    if veh_to_asset_angle >= vehicle_bearing and veh_to_asset_angle - vehicle_bearing < 180:
        return True

    if veh_to_asset_angle <= vehicle_bearing and vehicle_bearing - veh_to_asset_angle > 180:
        return True

    return False


poles_gps = pd.read_csv(os.path.join(SourceImageFolder, 'poles_gps.csv'))
alvium_gps = pd.read_csv(os.path.join(SourceImageFolder, 'vehicle_gps.csv'))

alvium_gps['course'] = 0.0
for i in range(alvium_gps.shape[0]):
    alvium_gps.loc[i, 'Latitude'] = deg2rad(alvium_gps.loc[i, 'Latitude'])
    alvium_gps.loc[i, 'Longitude'] = deg2rad(alvium_gps.loc[i, 'Longitude'])

course_lines = open(os.path.join(SourceImageFolder, 'speed_and_course.txt')).readlines()
data = []

for line in course_lines:        
    components = line.split()
    sec_nano = components[0].split('_')
    course = components[1].split(',')[0]
    alvium_gps.loc[alvium_gps['Timestamp'] == components[0][:-1], 'course'] = deg2rad(float(course))
    if sec_nano[0] in names_to_search.keys():
        images = names_to_search[sec_nano[0]]
        for nano, filename in images:
            timestamp = sec_nano[0] + '_' + nano
            poles_row = poles_gps[poles_gps['Timestamp'] == timestamp]
            alvium_row = alvium_gps[alvium_gps['Timestamp'] == components[0][:-1]]            
            data.append([timestamp,
                         float(poles_row['Latitude'][poles_row.index[0]]),
                         float(poles_row['Longitude'][poles_row.index[0]]),
                         filename,
                         rad2deg(float(alvium_row['Latitude'][alvium_row.index[0]])),
                         rad2deg(float(alvium_row['Longitude'][alvium_row.index[0]])),
                         deg2rad(float(course)), 0])            



gps_course = pd.DataFrame(data, columns=['timestamp', 'latitude', 'longitude', 'image filename',
                                         'vehicle latitude', 'vehicle longitude', 'vehicle bearing', 'filter flag'])
bbox = [gps_course["latitude"].min(), gps_course["latitude"].max(),
        gps_course["longitude"].min(), gps_course["longitude"].max()]

shp_file = open(CustomerAssetFolder+"/Poles.shp", "rb")
dbf_file = open(CustomerAssetFolder+"/Poles.dbf", "rb")
shx_file = open(CustomerAssetFolder+"/Poles.shx", "rb")
sf_reader = shapefile.Reader(shp=shp_file, dbf=dbf_file, shx=shx_file)
assets = sf_reader.shapes()
points = [(i, assets[i].points[0][1], assets[i].points[0][0]) for i in range(len(assets))
          if bbox[0] < assets[i].points[0][1] < bbox[1] and bbox[2] < assets[i].points[0][0] < bbox[3]]

asset_df = pd.DataFrame(points, columns=['asset_id', 'latitude', 'longitude'])
asset_df["nearby_images"] = np.nan
asset_df["nearby_images"] = asset_df["nearby_images"].astype('object')
asset_df["on road"] = False
possible_matches = len(points)

#print("Filtering unreachable poles")
for i in tqdm(range(possible_matches)):    
    row = asset_df.iloc[i]

    lat_asset = deg2rad(asset_df.iloc[i]["latitude"])
    lon_asset = deg2rad(asset_df.iloc[i]["longitude"])

    for lat_gps, lon_gps, course in zip(alvium_gps["Latitude"], alvium_gps["Longitude"], alvium_gps["course"]):
        dst = distance(lat_gps, lon_gps, lat_asset, lon_asset)
        if dst > PERPENDICULAR_DIST * 4:
            continue
        pole_angle = bearing(lat_gps, lon_gps, lat_asset, lon_asset)
        if is_on_right(pole_angle, course) and dst * math.fabs(math.sin(pole_angle - course)) < PERPENDICULAR_DIST:
            asset_df.loc[i, "on road"] = True
            break
    

asset_df_dropped = asset_df.drop(asset_df[asset_df["on road"] == False].index)
asset_df_dropped.to_csv(os.path.join(TargetImageFolder, 'asset_df_dropped.csv'), index=False, header=True)
possible_matches_drove_by = asset_df_dropped.shape[0]

total_matches = 0
total_matches_drove_by = 0
errors = []
errors_drove_by = []
#print("Matching poles.")
for i in tqdm(range(len(points))):
    asset = asset_df.iloc[i]
    nearby_images = []
    lat_asset = deg2rad(asset_df.iloc[i]["latitude"])
    lon_asset = deg2rad(asset_df.iloc[i]["longitude"])

    for image in zip(gps_course["image filename"], gps_course["latitude"],
                     gps_course["longitude"], gps_course["vehicle bearing"],
                     gps_course["vehicle latitude"], gps_course["vehicle longitude"]):
        lat_image = deg2rad(image[1])
        lon_image = deg2rad(image[2])

        if distance(lat_image, lon_image, lat_asset, lon_asset) < THRESHOLD:
            nearby_images.append(image)

    if nearby_images:        
        min_dist = sys.maxsize
        min_timestamp = None
        for image in nearby_images:
            var = image[0].split(".")[0].split("_")[2]
            if image[0].split(".")[0].split("_")[2] == "0":
                lat_image = deg2rad(image[1])
                lon_image = deg2rad(image[2])
                veh_lat_image = deg2rad(image[4])
                veh_lon_image = deg2rad(image[5])
                vehicle_bearing = image[3]
                veh_to_asset_angle = bearing(veh_lat_image, veh_lon_image, lat_asset, lon_asset)
                error_dist = distance(lat_image, lon_image, lat_asset, lon_asset)
                if is_on_right(veh_to_asset_angle, vehicle_bearing) and (error_dist < min_dist):
                    min_dist = error_dist
                    min_timestamp = "{}_{}".format(image[0].split(".")[0].split("_")[0],
                                                   image[0].split(".")[0].split("_")[1])

        if min_timestamp:
            total_matches += 1
            errors.append(min_dist)
            if asset_df.iloc[i]["on road"]:
                total_matches_drove_by += 1
                errors_drove_by.append(min_dist)
            for j in range(2):
                filename = min_timestamp + "_{}.jpg".format(j)
                gps_course.loc[gps_course['image filename'] == filename, "filter flag"] = 1

gps_course_dropped = gps_course.drop(gps_course[gps_course["filter flag"] == 0].index).drop("filter flag", axis=1)

save_gps_images = True
if save_gps_images:
  sp = yaspin(text='Copying {} images'.format(len(gps_course_dropped)), color="cyan")
  sp.start()
  for item, frame in gps_course_dropped.iterrows():
      shutil.copy(os.path.join(SourceImageFolder, frame["image filename"]), TargetImageFolder)
  sp.stop()
  
gps_course_dropped.to_csv(os.path.join(TargetImageFolder, 'gps_images_updated.csv'), index=False, header=True)

print("Results accounting for all customer poles (even unreachable ones)")
print("-----------------------------------------------------------------")
if possible_matches == 0:
  print("No customer poles. ")
else:
  print("Total Asset Matches: {} out of {} ({}%)".format(total_matches, possible_matches, 100 * total_matches/possible_matches))
  print("Matching Images Copied: {}".format(len(gps_course_dropped.index)))
  print("Error Stats:")
  print(" Min: {}".format(np.amin(errors)))
  print(" Max: {}".format(np.amax(errors)))
  print(" Mean: {}".format(np.mean(errors)))
  print(" Std Dev: {}".format(np.std(errors)))
print("-----------------------------------------------------------------\n")
print("Results after filtering unreachable customer poles")
print("-----------------------------------------------------------------")
if possible_matches_drove_by == 0:
  print("No reachable customer pole.")
else:
  print("Total Asset Matches: {} out of {} ({}%)".format(total_matches_drove_by, possible_matches_drove_by,100 * total_matches_drove_by/possible_matches_drove_by))
  print("Matching Images: {}".format(len(gps_course_dropped.index)))
  print("Error Stats:")
  print(" Min: {}".format(np.amin(errors_drove_by)))
  print(" Max: {}".format(np.amax(errors_drove_by)))
  print(" Mean: {}".format(np.mean(errors_drove_by)))
  print(" Std Dev: {}".format(np.std(errors_drove_by)))
print("-----------------------------------------------------------------")