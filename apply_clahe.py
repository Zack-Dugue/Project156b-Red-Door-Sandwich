import os
import cv2

# Function to apply CLAHE to an image and save it
def apply_clahe_to_image(input_path, output_path):
    print(input_path)
    img = cv2.imread(input_path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    cv2.imwrite(output_path, img_clahe)

# Function to process a directory and its subdirectories recursively
def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input directory
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # If it's a directory, recursively process it
        if os.path.isdir(input_path):
            process_directory(input_path, output_path)
        # If it's a file, apply CLAHE and save it to the output directory
        elif os.path.isfile(input_path) and input_path.endswith('.jpg'):
            apply_clahe_to_image(input_path, output_path)

# Specify the input and output directories
input_directory = os.path.join('data', 'train')
output_directory = os.path.join('data', 'clahe', 'train')

# Process the input directory
process_directory(input_directory, output_directory)