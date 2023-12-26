import os

# Directory containing the JPG and TXT files
jpg_dir = "/mnt/d/Coding/opencv-projects/real-time-bengali-number-plate-detection/dataset/v2/train/images/train/"
txt_dir = "/mnt/d/Coding/opencv-projects/real-time-bengali-number-plate-detection/dataset/v2/train/labels/train/"

# Create sets to store the names of XML and TXT files
jpg_files = set()
txt_files = set()

# Iterate through the files in the directory
for filename in os.listdir(jpg_dir):
    if filename.endswith(".jpg"):
        jpg_files.add(os.path.splitext(filename)[0])

for filename in os.listdir(txt_dir):
    if filename.endswith(".txt"):
        txt_files.add(os.path.splitext(filename)[0])

# Find the uncommon elements between JPG and TXT files
uncommon_files = jpg_files.difference(txt_files)

# Count the number of XML files with no corresponding TXT files
count = len(uncommon_files)

print(f"Number of JPG files with no corresponding TXT files: {count}")
print(uncommon_files)