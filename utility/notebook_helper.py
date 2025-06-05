import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ipcv import cvlib
import seaborn as sns


def display_for_comparisons(prev_img, curr_img, prev_label, curr_label):
    # Display the previous and current images side-by-side for visual comparison.
    plt.figure(figsize=(14, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(prev_img, cmap='gray')
    plt.title(prev_label)

    plt.subplot(1, 2, 2)
    plt.imshow(curr_img, cmap='gray')
    plt.title(curr_label)


def display_color_grids(binary_array):
    cmap = colors.ListedColormap(['white', 'gray'])

    fig, ax = plt.subplots()
    ax.imshow(binary_array, cmap=cmap, origin='upper', extent=[0, binary_array.shape[1], 0, binary_array.shape[0]])
    ax.grid(color='green', linewidth=1)
    ax.set_xticks(np.arange(0, binary_array.shape[1], 1))
    ax.set_yticks(np.arange(0, binary_array.shape[0], 1))
    plt.show()

def display_histograms(img, title):

    # Flatten the grayscale image data
    pixel_intensities = img.flatten()

    # Create the histogram using seaborn
    plt.figure(figsize=(8, 6))
    sns.histplot(pixel_intensities, bins=256, color='gray', kde=True)
    plt.title(title)
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
