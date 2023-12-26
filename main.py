#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cv2, configparser
from ultralytics import YOLO

import util, add_missing_data
from sort.sort import *

import csv
import numpy as np

mot_tracker = Sort()
config = configparser.ConfigParser()
results = {}

# read config file
config.read('config.ini')

# load models
coco_model = YOLO("./models/yolov8n.pt")
license_plate_detector = YOLO("./models/license_plate_detector_best.pt")

# load video
cap = cv2.VideoCapture(config['Video']['src'])

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_vehicles = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_vehicles.append([x1, y1, x2, y2, score])

        if len(detections_vehicles) == 0:
            continue

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_vehicles))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1): int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(
                    license_plate_crop, cv2.COLOR_BGR2GRAY
                )

                # # apply binary thresholding
                # _, license_plate_crop_bw = cv2.threshold(
                #     license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY
                # )

                # read license plate number
                license_plate_text, license_plate_text_score = util.read_license_plate(
                    license_plate_crop_gray
                )

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                        "license_plate": {
                            "bbox": [x1, y1, x2, y2],
                            "text": license_plate_text.replace(",",""),
                            "bbox_score": score,
                            "text_score": license_plate_text_score,
                        },
                    }
# write results
dinfo_path = config['Video']['dinfo_path']
dinfo_path_interpolated = config['Video']['dinfo_path_interpolated']
util.write_csv(results, dinfo_path)

# Load the CSV file
with open(dinfo_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = add_missing_data.interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open(dinfo_path_interpolated, 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
