from math import ceil

import cv2
import numpy as np
import math

from ipcv import cvlib

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def preprocess(image, gamma, gaussian_ksize, gaussian_sigma):
    # Adjust the contrast
    p = cvlib.adjust_gamma(image, gamma)

    # Convert image to gray scale for processing.
    p = cvlib.convert_rgb2gray(p)

    # Remove noise.
    p = cvlib.gaussian_blur(p, gaussian_ksize, gaussian_sigma)

    return p


def calculate_angle(point_a, point_b, point_c):
    """Calculates the angle formed by three points.

    Args:
        point_a: A tuple or list representing the coordinates of point A (x1, y1).
        point_b: A tuple or list representing the coordinates of point B (x2, y2).
        point_c: A tuple or list representing the coordinates of point C (x3, y3).

    Returns:
        The angle in degrees.
    """
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c

    # Create vectors
    vector_ba = (x1 - x2, y1 - y2)
    vector_bc = (x3 - x2, y3 - y2)

    # Calculate the dot product
    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]

    # Calculate the magnitudes
    magnitude_ba = math.sqrt(vector_ba[0] ** 2 + vector_ba[1] ** 2)
    magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)

    # Calculate the angle in radians
    angle_rad = math.acos(cos_theta)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def crop_roi(image, max_contour, color):
    perimeter = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.04 * perimeter, True)
    cv2.drawContours(image, [approx], -1, color, 3)
    (x, y, w, h) = cv2.boundingRect(approx)
    return image[y:y + h, x:x + w]


def perpendicular(angle):
    print(f'angle:{angle}')
    if abs(90 - angle) < 10:
        return True
    else:
        return False


def find_rectangle(source_img, processed_img):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = processed_img.shape[0] * processed_img.shape[1]
    required_aspect_ration = image_area / 30

    print(f'Image Area          : {image_area:,}')
    print(f'Min required area   : {required_aspect_ration:,}')
    print(f'Number of contours  : {len(contours)}\n')

    max_contour = None
    max_area = 0
    for contour in contours:
        # print(f'\nContour len: {len(contour)}')
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # print(f'Approx len: {len(approx)}')
        cv2.drawContours(source_img, [approx], -1, GREEN, 3)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            print(f'Calculated area : {area:,}')

            (x, y, w, h) = cv2.boundingRect(approx)

            coordinates = approx.reshape(4, 2)
            a = coordinates[0]
            b = coordinates[1]
            c = coordinates[2]
            d = coordinates[3]

            if perpendicular(calculate_angle(a, b, c)) and perpendicular(calculate_angle(b, c, d)) and perpendicular(
                    calculate_angle(c, d, a)) and perpendicular(calculate_angle(d, a, b)):

                aspect_ratio = w / float(h)
                print(f'Calculated ratio: {aspect_ratio:,}')

                # Ensure the QR-Code is a box shaped (max ratio of w:h = 1.2:1)
                # AND min area => 12 x 10
                # if area > 100 and 0.8 <= aspect_ratio <= 1.2:
                # if area > required_aspect_ration and 0.8 <= aspect_ratio <= 1.2:
                if area > required_aspect_ration and aspect_ratio > 1.2:
                    if area > max_area:
                        max_area = area
                        print(f'Max-Area: {max_area:,}')
                        max_contour = contour
            print('\r')
    return max_contour


def proportionate_close(image, dilate_iteration, erode_iteration):
    horizontal_se = np.array([
        [1, 1, 1]
    ])

    vertical_se = np.array([
        [1],
        [1]
    ])
    image = cv2.dilate(image, horizontal_se, iterations=dilate_iteration)
    image = cv2.dilate(image, vertical_se, iterations=erode_iteration)
    return image


def detect_barcode_v2(**kwargs):
    cropped = None
    # source_img = kwargs['image'].copy()
    # p = kwargs['image'].copy()
    p = preprocess(kwargs['image'], kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    p = cvlib.binarize_inv(p, kwargs['thresh_min'])

    # 4:1 (dilate:erosion)
    erode_iteration = ceil(kwargs['iteration'] * 1)
    for i in range(1, 10):
        p = proportionate_close(p, kwargs['iteration'], erode_iteration)
        contour = find_rectangle(kwargs['image'], p)
        if contour is not None:
            cropped = crop_roi(kwargs['image'], contour, BLUE)
            break
    return p


def detect_barcode(**kwargs):
    """
    Processed the given image to detect barcode using the following parameters:.
    :param kwargs:
    - image (matrix): Image in which the barcode needs to be detected.
    - gamma (float): A number to set the gamma value.
    - gaussian_ksize (tuple): The gaussian kernel size used for smoothing an image. E.g. (3,3) or (9,9).
    - gaussian_sigma (float): The gaussian_sigma used for smoothing an image.
    - avg_ksize1 (tuple): The kernel size used for the first average smoothing. E.g. (3,3) or (9,9).
    - avg_ksize2 (tuple): The kernel size used for the second average smoothing. E.g. (3,3) or (9,9).
    - thresh_min (uint): The threshold value for binarizing an image.
    - dilate_kernel (tuple): The kernel size of structuring element used for dilation. E.g. (21,7) or (51,9).
    - dilate_iteration (uint): The number of interation the dilation is performed.
    - shrink_factor (uint):  The original image size is divided with shrink_factor to resize an image (shrinking).
    - offset (uint):  The offest of the box to resize to.
    :return: The detected barcodes are annotated on the original image first, and cropped barcode is returned.
    """

    p = cvlib.convert_rgb2gray(kwargs['image'])

    p = cvlib.adjust_gamma(p, kwargs['gamma'])

    p = cvlib.gaussian_blur(p, kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])

    p = cvlib.average_blur(p, kwargs['avg_ksize1'])

    p = cvlib.detect_gradient(p)

    # Only [3,3] works
    p = cvlib.average_blur(p, kwargs['avg_ksize2'])

    p = cvlib.binarize(p, kwargs['thresh_min'])

    # p = cvlib.dilate(p, kwargs['dilate_kernel'], kwargs['dilate_iteration'])
    p = cvlib.morph_close(p, kwargs['dilate_kernel'])
    p = cvlib.morph_dilate(p, kwargs['dilate_iteration'])

    # Record the image's original size for enlarging.
    x = p
    new_width = int(p.shape[1] / kwargs['shrink_factor'])
    new_height = int(p.shape[0] / kwargs['shrink_factor'])

    p = cvlib.resize_image(p, new_width, new_height)

    p = cvlib.resize_image(p, x.shape[1], x.shape[0])

    cropped = cvlib.get_prominent_contour(kwargs['image'], p, kwargs['offset'])

    return cropped


def detect_qrcode(**kwargs):
    p = kwargs['image'].copy()
    p = cvlib.binarize(p, kwargs['thresh_min'])

    ksize = (13, 13)
    iteration = 5
    p = cvlib.morph_close(p, ksize)
    p = cvlib.morph_open(p, ksize)
    p = cvlib.morph_dilate(p, iteration, ksize)
    p = cvlib.morph_erode(p, iteration)

    contours, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def decode_barcode(img):
    """
    This function is used only to verify the detected barcode
    :param img: The cropped image with barcode detected.
    :return:
    """
    detector = cv2.barcode_BarcodeDetector()
    # decoded_text, points, barcode_type = detector.detectAndDecode(img)
    decoded_text, _, _ = detector.detectAndDecode(img)
    if decoded_text == '':
        return None

    return decoded_text
