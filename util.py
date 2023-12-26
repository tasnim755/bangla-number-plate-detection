import os
import xml.etree.ElementTree as ET

import easyocr

# Directory containing the XML files
input_dir = "/mnt/d/Coding/opencv-projects/real-time-bengali-number-plate-detection/dataset/v2/test/labels/test/"

# Ensure the output directory exists
output_dir = os.path.join(input_dir, "output")
os.makedirs(output_dir, exist_ok=True)


# Function to extract and save values from XML
def process_xml(input_file):
    with open(input_file, "r") as file:
        xml_content = file.read()

    root = ET.fromstring(xml_content)
    width = int(root.find(".//width").text)
    height = int(root.find(".//height").text)
    xmin = int(root.find(".//xmin").text)
    ymin = int(root.find(".//ymin").text)
    xmax = int(root.find(".//xmax").text)
    ymax = int(root.find(".//ymax").text)

    # Construct the output file name based on the input file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    # Convert them in yolov8 format
    x_c = (xmin + xmax) / 2.0
    y_c = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    # Normalize
    x_c_norm = x_c / width
    y_c_norm = y_c / height
    w_norm = w / width
    h_norm = h / height

    # Create the text file and write the values
    with open(output_file, "w") as file:
        file.write(f"0 {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}")


# Process all XML files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".xml"):
        input_file = os.path.join(input_dir, filename)
        process_xml(input_file)

# Initialize the OCR reader
reader = easyocr.Reader(['bn'])

# Mapping dictionaries for character conversion
dict_char_to_int = {'0': '০',
                    '1': '১',
                    '2': '২',
                    '3': '৫',
                    '4': '৭',
                    '5': '৩',
                    '6': '৮',
                    '7': '৯',
                    '8': '৪',
                    '9': '৭',
                    }


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def count_bengali_digits(text):
    # Assuming Bengali digits are in the Unicode range U+09E6 to U+09EF
    bengali_digits = [chr(i) for i in range(0x09E6, 0x09F0)]
    return sum(1 for char in text if char in bengali_digits)

def has_ascii_characters(input_string):
    return all(ord(char) < 128 for char in input_string)

def refactor_number_box(text):
    if len(text) > 6:
        text = text[:2] + "-" + text[3:]
    formatted = ''.join(dict_char_to_int.get(d, d) for d in text)
    return formatted


def parse_bengali_number_plate(detections):
    number_box = ""
    max_digits = 0
    number_box_conf_score = 0.0
    city_name_conf_score = 0.0
    city_name_comp = 0
    prefix = ""

    for _, parsed_text, cscore in detections:
        # We ignore anything less than 20% confidence and length more than 20.
        if cscore < 0.2 or len(parsed_text) > 20:
            continue
        dig_cnt = count_bengali_digits(parsed_text)
        if dig_cnt > max_digits and cscore > number_box_conf_score:
            max_digits = dig_cnt
            number_box = refactor_number_box(parsed_text)
            number_box_conf_score = cscore
        if dig_cnt < 2:
            prefix += ' '.join(parsed_text.split()) + ' '
            city_name_conf_score += cscore
            city_name_comp += 1

    prefix = prefix.strip()
    return prefix + ' ' + number_box, 0.5 * (number_box_conf_score + (city_name_conf_score / (city_name_comp + 1)))


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)
    return parse_bengali_number_plate(detections)


# Done
def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
