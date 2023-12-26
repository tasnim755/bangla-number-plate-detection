import easyocr
from PIL import Image
import numpy as np

def convert_to_bnw(image_path, threshold=128):
    # Open the image using PIL (Python Imaging Library) and convert it to grayscale
    image = Image.open(image_path).convert('L')

    # Convert the grayscale image to a NumPy array
    image_np = np.array(image)

    # Apply binary thresholding
    image_bnw = (image_np > threshold) * 255

    # Convert the binary image to a PIL image
    image_bnw_pil = Image.fromarray(image_bnw.astype('uint8'))

    return image_bnw_pil

# Initialize the OCR reader
reader = easyocr.Reader(['bn'])

# Open the image using PIL (Python Imaging Library)
image = Image.open("./resources/number_plate.jpg").convert('L')

# Convert the grayscale image to a NumPy array
image = np.array(image)

# Read the content
detections = reader.readtext(image)

# Print the contents
for detection in detections:
    print(detection)