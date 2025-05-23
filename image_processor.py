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


def adjust_gamma2(image, gamma=1.0):
    """
    Adjusts the intensity of the given image using the power law,
    :param image: The image to adjust.
    :param gamma: The power to which the image will be adjusted. Greater than 1 make it darker, lower than 1 makes it
    brighter.
    :return: The adjusted image.
    """
    return np.array(255 * (image / 255) ** gamma, dtype='uint8')


def gaussian_blur(image, ksize):
    """
    Applies a gaussian blur on the given image.
    :param image: The image to blur.
    :param ksize: The size of the kernel to use.
    :return: The blurred image.
    """
    return cv2.GaussianBlur(image, ksize, 0)


def average_smooth(image, ksize):
    """
    Applies a average smoothing on the given image.
    :param image: The image to smooth.
    :param ksize: The size of the kernel to use.
    :return: The smoothed image.
    """
    mean = np.ones(ksize, np.float32) / math.prod(ksize)
    blurred = cv2.filter2D(image, -1, mean)
    return np.array(255 * (blurred / 255), dtype='uint8')


def high_boost(image, blurred):
    """
    Applies a high boosting on the given image.
    :param image: The original image.
    :param blurred: The blurred image.
    :return: The blurred image.
    """
    return np.array(255 * ((image + cv2.convertScaleAbs(cv2.subtract(image, blurred))) / 255), dtype='uint8')


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
    # blurred = cv2.blur(image, ksize=ksize)
    (_, thresh) = cv2.threshold(image, min_val, 255, cv2.THRESH_BINARY)
    return thresh


def dilate(image, ksize, iteration):
    """
    Dilates the given image using the given kernel.
    :param image: The image to dilate.
    :param ksize: The size of the kernel to use.
    :param iteration: The number of iterations to dilate.
    :return: The dilated image.
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    closed = cv2.dilate(closed, None, iterations=iteration)
    return closed


def detect_barcode2(image, gamma, gaussian_ksize, avg_ksize, thresh_min):
    # Convert to grayscale for processing.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gamma_corrected = adjust_gamma(gray, gamma)

    blurred = gaussian_blur(gamma_corrected, gaussian_ksize)

    blurred2 = average_smooth(blurred, avg_ksize)

    boosted = high_boost(blurred2, blurred2)

    gradient = detect_gradient(boosted)

    avg_smoothed = average_smooth(gradient, [3,3])

    thresh = binarize(avg_smoothed)

    morphed = dilate(thresh, [21, 7], 4)

    # Shrink
    new_width = int(morphed.shape[1] / 6)
    new_height = int(morphed.shape[0] / 6)

    shrunk = resize_image(morphed, new_width, new_height)

    enlarged = resize_image(shrunk, morphed.shape[1], morphed.shape[0])

    # find the contours in the thresholded image, then sort the contours by their area, keeping only the largest one
    (contours, _) = cv2.findContours(enlarged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    [X, Y, W, H] = cv2.boundingRect(box)
    cropped = image[Y:Y + H, X:X + W]

    return image, cropped


### Deprecated

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def detect_barcode(image):
    # Convert to grayscale for processing.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find vertical lines (x-axis intensity change when y = 0).
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

    # Find horizontal lines (y-axis intensity change when x = 0).
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)

    # Converts negative values to absolute values |x|.
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (3, 3))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # find the contours in the thresholded image, then sort the contours by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    [X, Y, W, H] = cv2.boundingRect(box)

    cropped = image[Y:Y + H, X:X + W]

    return image, cropped
