import math

import cv2
import numpy as np


def load_image(file_path):
    """
    Loads an image from the given path, converts it to RGB from BGR and returns it as a numpy array.
    :param file_path: The path to the image.
    :return: The image as a numpy array.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)


def convert_rgb2gray(image):
    """
    Converts the given RGB image to grayscale.
    :param image: The RGB image to convert.
    :return: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width, height):
    """
    Resizes the given image to the given width and height.
    :param image: The image to resize.
    :param width: The width of the resized image.
    :param height: The height of the resized image.
    :return: The resized image.
    """
    return cv2.resize(image, (width, height))


def adjust_gamma(image, gamma=1.0):
    """
    Adjusts the intensity of the given image using the power law,
    :param image: The image to adjust.
    :param gamma: The pixel intensity is raised to the power of gamma. When gamma >1 intensity becomes darker, meanwhile
    when gamma < 1 (non-negative value) intensity becomes brighter. When gamma = 1, there is no effect.
    :return: The adjusted image.
    """
    # s = (c*r)^gamma
    return np.array(255 * (image / 255) ** gamma, dtype='uint8')


def gaussian_blur(image, ksize, sigma):
    """
    Applies a gaussian blur on the given image.
    :param image: The image to blur.
    :param ksize: The size of the kernel to use (requires larges kernal compared to mean/median)
    :param sigma: Gaussian kernel standard deviation in X direction.
    :return: The blurred image.
    """
    return np.array(cv2.GaussianBlur(image, ksize, sigma), dtype='uint8')


def average_blur(image, ksize):
    """
    Applies a mean blurring on the given image.
    :param image: The image to smooth.
    :param ksize: The size of the kernel to use.
    :return: The smoothed image.
    """
    mean = np.ones(ksize, np.float32) / math.prod(ksize)
    blurred = cv2.filter2D(image, -1, mean)
    return np.array(255 * (blurred / 255), dtype='uint8')


def morph_close(image, ksize=None):
    """
    Performs close operation on the given image using the given kernel.
    :param image: The image to dilate.
    :param ksize: The size of the kernel to use.
    :return: The dilated image.
    """
    se = None
    if ksize is not None:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)


def morph_dilate(image, iterations, ksize=None):
    """
    Performs dilate operation on the given image using the given kernel.
    :param image: The image to dilate.
    :param ksize: The size of the kernel to use.
    :param iterations: The number of iterations to perform.
    :return: The dilated image.
    """
    se = None
    if ksize is not None:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.dilate(image, se, iterations=iterations)


def resize_box(box, offset):
    """
    Resizes the box according to the given offset and offset size.
    :param box: The detected box to resize.
    :param offset: The offset of the box to resize to.
    :return: The resized box.
    """
    box[0] = box[0] - offset
    box[1] = box[1] - offset
    box[2] = box[2] + offset
    box[3] = box[3] + offset
    return box


def get_prominent_contour(source_image, processed_image, offset=0):
    """
    Gets the prominent contour from the given processed image and draws a box on the source image.
    :param source_image: The source image to draw on.
    :param processed_image: The processed image to trace the biggest contour on.
    :param offset: The offset of the box to resize to.
    :return: Returns the prominent contour.
    """
    contours, hierarchy = cv2.findContours(processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))

    # Increase the box offset for better detection.
    box = resize_box(box, offset)

    # draw a bounding box rounded the detected barcode and display the image
    cv2.drawContours(source_image, [box], -1, (0, 255, 0), 3);

    [X, Y, W, H] = cv2.boundingRect(box)
    cropped = source_image[Y - offset:Y + H + offset, X + offset:X + W + offset]

    return cropped


def detect_gradient(image):
    """
    Detects the magnitude of vertical and horizontal gradients of the given image for both .
    :param image: The source image.
    :return: The gradient of the given image.
    """

    # Find vertical line intensity change (using x-axis, when y = 0).
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

    # Find horizontal line intensity change(using y-axis, when x = 0).
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Subtract the y-gradient from the x-gradient to find the magnitude.
    grad = cv2.subtract(grad_x, grad_y)

    # Converts negative values to absolute values |x|.
    return cv2.convertScaleAbs(grad)


def binarize(image, min_val=127):
    """
    Binarize the given image with a given minimum threshold value.
    :param image: The image to binarize.
    :param min_val: Any value below this threshold will be set to 0.
    :return: A binarized image.
    """
    (_, thresh) = cv2.threshold(image, min_val, 255, cv2.THRESH_BINARY)
    return thresh
