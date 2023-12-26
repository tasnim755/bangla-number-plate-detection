from PIL import Image
import cv2
import easyocr
import os
import pandas as pd
import numpy as np

# Specify the target dimensions
target_width = 300
target_height = 300

# Input and output directories
input_directory = "./resources/manual_train/"
output_directory_for_resized_image = "./resources/manual_train_resize/"
output_directory_for_grayscaled_image = "./resources/bn_filtered_bnw/"

# Ensure the output directory exists
if not os.path.exists(output_directory_for_resized_image):
    os.makedirs(output_directory_for_resized_image)
if not os.path.exists(output_directory_for_grayscaled_image):
    os.makedirs(output_directory_for_grayscaled_image)

def resize_all():
    # Loop through each image in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
            # Read the image
            image_path = os.path.join(input_directory, filename)
            img = cv2.imread(image_path)

            # Resize the image
            resized_img = cv2.resize(img, (target_width, target_height))

            # Save the resized image to the output directory
            output_path = os.path.join(output_directory_for_resized_image, filename)
            cv2.imwrite(output_path, resized_img)

    print("Resizing complete.")

def bnw_all():
    # Loop through each resized image in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
            # Read the image
            image_path = os.path.join(input_directory, filename)
            img = Image.open(image_path)

            # Convert the image to grayscale
            gray_img = img.convert('L')

            # Apply binary thresholding
            binary_img = gray_img.point(lambda x: 255 if x > 150 else 0)

            # Save the binary image to the output directory
            output_path = os.path.join(output_directory_for_grayscaled_image, filename)
            binary_img.save(output_path)

    print("Conversion to grayscale complete.")

def bnw_single(image_path):
    # Loop through each resized image in the input directory
    img = Image.open(image_path)

    # Convert the image to grayscale
    gray_img = img.convert('L')

    # Apply binary thresholding
    binary_img = gray_img.point(lambda x: 255 if x > 90 else 0)

    # Save the binary image to the output directory
    binary_img.save("./resources/result.jpg")

    print("Conversion to bnw complete.")

def curate_dataset():
    # Specify the path to the directory
    directory_path = './resources/bn_validation_bnw/curated/'

    # Get a list of filenames in the directory
    filenames = os.listdir(directory_path)

    # Filter out non-file entries (e.g., directories) if needed
    filenames = [filename for filename in filenames if os.path.isfile(os.path.join(directory_path, filename))]

    # Remove labels.csv
    try:
        filenames.remove("labels.csv")
    except:
        print("missing labels.csv")

    # Report number of files
    print(f"total number of files: {len(filenames)}")

    return filenames

def read_csv():
    # Specify the path to your CSV file
    csv_file_path = './resources/bn_filtered/labels.csv'

    # Read the CSV file into a DataFrame, specifying the encoding as 'utf-8'
    df = pd.read_csv(csv_file_path, encoding='utf-8')

    # Now, 'df' is your DataFrame with the CSV contents
    print(df)

def filter_csv_for_curated_dataset():
    # Specify the path to your input CSV file
    input_csv_path = './resources/bn_validation_bnw/labels.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path, encoding='utf-8')

    # Specify the list of contents you want to remove from the "filename" column
    contents_to_remove = curate_dataset()

    # Filter rows based on the condition and create a new DataFrame
    filtered_df = df[df['filename'].isin(contents_to_remove)]

    # Specify the path for the output CSV file
    output_csv_path = './resources/bn_validation_bnw/curated/labels.csv'

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False, encoding='utf-8')

    print(f"Filtered CSV file saved to: {output_csv_path}")

def CCA():
    # Read the image in grayscale
    image = cv2.imread('./resources/bn_filtered_bnw/curated/25100.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find connected components
    output = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    (num_labels, labels, stats, centroids) = output

    # Filter out small components
    min_size = 20  # Adjust this threshold based on your requirement
    filtered_image = np.zeros_like(binary_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_image[labels == i] = 255

    # Save the result
    cv2.imwrite('./resources/filtered_image.jpg', filtered_image)

def easyocr_read(image_path):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['bn'], recog_network='custom_model')
    result = reader.readtext(image)
    print(result)

# resize_all()
# grayscale_all()
# curate_dataset()
# read_csv()
# filter_csv_for_curated_dataset()
# CCA()
# bnw_single("./resources/test10.jpg")
easyocr_read("./resources/manual_train_resize/6.jpg")