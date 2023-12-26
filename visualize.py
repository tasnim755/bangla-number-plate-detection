import ast

import cv2, configparser
import numpy as np
import pandas as pd

from PIL import ImageFont, ImageDraw, Image

config = configparser.ConfigParser()
config.read('config.ini')

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def create_image_from_bangla_text(frame, text, text_pos):
    fontpath = "./resources/Siyamrupali.ttf"
    font = ImageFont.truetype(fontpath, 32)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(text_pos, text, font = font, fill = (0, 0, 0, 0))
    return np.array(img_pil)

results = pd.read_csv(config['Video']['dinfo_path_interpolated'], encoding='utf-8')

# load video
video_path = config['Video']['src']
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_path[:-4] + "_out.mp4", fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)][
                                 'license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()
    if frame is None:
        continue
    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[
                                          0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ',
                                                                                                               ','))
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop

frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # draw car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ',
                                                                                                                 ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ',
                                                                                                        ' ').replace(
                    ' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                # Check if license_crop is a valid image
                if license_crop is not None:
                    # Verify coordinates and dimensions
                    y1 = int(car_y1) - H
                    y2 = int(car_y1)
                    x1 = int((car_x2 + car_x1 - W) / 2)
                    x2 = int((car_x2 + car_x1 + W) / 2)

                    # Overlay license_crop on the frame
                    # frame[y1:y2, x1:x2, :] = license_crop

                    # Draw a white rectangle as background for the license plate text
                    # frame[y1 - 30: y1, x1:x2, :] = (255, 255, 255)

                    # Get text size and position
                    text = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                    print(f"y1: {car_y1}, y2: {car_y2}, x1: {car_x1}, x2: {car_x2}, H: {H}, W: {W}, text: {text}")
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                    text_position = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

                    # Draw the license plate text on the frame
                    img = create_image_from_bangla_text(frame, text, text_position)
                    cv2.putText(frame, "", text_position, cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17,
                                cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

out.release()
cap.release()
