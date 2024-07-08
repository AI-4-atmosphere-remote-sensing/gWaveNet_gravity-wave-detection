import os
import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def apply_custom_kernel(image_path, custom_kernel):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None or len(image.shape) != 3:
        print(f"Error: {image_path} is not a valid image.")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(custom_kernel.shape) != 2:
        print(f"Error: The custom kernel is not a 2-D array.")
        return None

    convoluted_image = convolve2d(gray_image, custom_kernel, mode='same', boundary='symm')
    convoluted_image_normalized = (convoluted_image - np.min(convoluted_image)) / (np.max(convoluted_image) - np.min(convoluted_image)) * 255

    highlighted_image = np.zeros_like(image)
    highlighted_image[convoluted_image_normalized > np.percentile(convoluted_image_normalized, 90)] = (0, 255, 255)

    combined_image = cv2.addWeighted(image, 0.7, highlighted_image, 0.3, 0)
    return combined_image

custom_kernel = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1]])

directory_path = '/image/pat/to/dir'

image_list = []
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image_path = os.path.join(directory_path, filename)
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        combined_image = apply_custom_kernel(image_path, custom_kernel)

        if original_image is not None and combined_image is not None:
            image_list.append((original_image, combined_image))

plot_index = 1
for i in range(0, len(image_list), 1):
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    if i < len(image_list):
        original_image, combined_image = image_list[i]
        ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")
        ax[1].imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        ax[1].axis("off")
    else:
        ax[0].axis("off")
        ax[1].axis("off")
    plt.tight_layout()

    plt.tight_layout(pad=0)
    plt.savefig(f"/save/image/results/to/dir/plot_{plot_index}.png", dpi=400)
    plt.show()
    plot_index += 1
