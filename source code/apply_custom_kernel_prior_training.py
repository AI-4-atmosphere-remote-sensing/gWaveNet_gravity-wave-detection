import cv2
import numpy as np
import os

def apply_custom_filter_to_images(src_directory, dst_directory):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    # Define the custom filter

    custom_filter = np.array([[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]], dtype=np.float32)

## uncomment as necessary
    # custom_filter = np.array([[1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1]], dtype=np.float32)

    # custom_filter = np.array([[1, 0, 1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1, 0, 1]], dtype=np.float32)

    # custom_filter = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1, 0, 1, 0, 1],
    #                           [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #                           [1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.float32)


    # Normalize the filter
    custom_filter = custom_filter / np.sum(custom_filter)

    # Iterate through all files and subdirectories in the source directory
    for root, dirs, files in os.walk(src_directory):
        for file in files:
            # Check if the file is an image (you can modify this condition based on your image file extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construct the full source file path
                src_file_path = os.path.join(root, file)

                # Create the corresponding destination directory structure
                dst_subdir = os.path.relpath(root, src_directory)
                dst_directory_path = os.path.join(dst_directory, dst_subdir)
                os.makedirs(dst_directory_path, exist_ok=True)

                # Construct the full destination file path
                dst_file_path = os.path.join(dst_directory_path, f'filtered_{file}')

                # Read the image
                image = cv2.imread(src_file_path)

                # Convert the image to float32 for the convolution
                image = image.astype(np.float32)

                # Apply the convolution operation using filter2D
                filtered_image = cv2.filter2D(image, -1, custom_filter)

                # Convert the filtered image back to uint8
                filtered_image = np.clip(filtered_image, 0, 255)
                filtered_image = filtered_image.astype(np.uint8)

                # Save the filtered image
                cv2.imwrite(dst_file_path, filtered_image)

                print(f"Filtered image saved: {dst_file_path}")

# Specify the source and destination directories
src_directory = '/path/to/your/source/directory'
dst_directory = '/path/to/your/destination/directory'

# Apply the custom filter to images in the source directory and save them to the destination directory
apply_custom_filter_to_images(src_directory, dst_directory)
