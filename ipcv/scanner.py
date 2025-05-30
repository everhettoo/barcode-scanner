from math import ceil
from pickletools import uint8

import cv2
import numpy as np
import math

from ipcv import cvlib

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = 0
WHITE = 255


def preprocess_image(image, gamma, gaussian_ksize, gaussian_sigma):
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


def is_perpendicular_angle(angle):
    # print(f'angle:{angle}')
    if abs(90 - angle) < 10:
        return True
    else:
        return False


def rectangle_coordinates(approx):
    """
    Reshapes the approx coordinates of a rectangle inside a list-of-list to flat-list and returns the four coordinates.
    :param approx: Coordinates of a most-likely rectangle.
    :return: Four coordinates.
    """
    coordinates = approx.reshape(4, 2)
    return coordinates[0], coordinates[1], coordinates[2], coordinates[3]


def find_rectangle(source_img, processed_img, min_area_factor, box=False, draw=False):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = processed_img.shape[0] * processed_img.shape[1]
    min_required_area = image_area * min_area_factor

    print(f'Image Area          : {image_area:,}')
    print(f'Min required area   : {min_required_area:,.2f} (at {min_area_factor} rate)')
    print(f'Number of contours  : {len(contours)}')

    selected_max_contour = None
    selected_max_area = 0

    # Epsilon: is maximum distance from contour to approximated contour.
    # eps = 0.04
    eps = 0.0391
    # eps = 0.05
    # Contour has coordinates for drawing the detected shape (polygons - can be more than 3 or 4)
    for contour in contours:
        # Contour-perimeter: calculates the perimeter for the given points when it's a closed lines.
        perimeter = cv2.arcLength(contour, True)

        # Ramer–Douglas–Peucker algorithm: It approximates a contour shape to another shape with reduced vertices
        # depending upon the precision of epsilon. So, coordinates for polygons are returned.
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if draw: cv2.drawContours(source_img, [approx], -1, GREEN, 3)

        # Only polygons with 4 angles are considered since a box or rectangle is needed.
        if len(approx) == 4:
            calculated_area = cv2.contourArea(contour)
            # print(f'\nCalculated area : {calculated_area:,.2f} A=({coordinates[0][0]},{coordinates[0][1]})')

            # Retrieve the four coordinates (of a most-likely rectangle) to verify if it is a rectangle.
            [a, b, c, d] = rectangle_coordinates(approx)
            print(
                f'\nCalculated area    : {calculated_area:,.2f}, Coordinates (X,Y)=[A=({a}), B=({b}), C=({c}), D=({d})]')

            if (is_perpendicular_angle(calculate_angle(a, b, c))
                    and is_perpendicular_angle(calculate_angle(b, c, d))
                    and is_perpendicular_angle(calculate_angle(c, d, a))
                    and is_perpendicular_angle(calculate_angle(d, a, b))):

                (x, y, w, h) = cv2.boundingRect(approx)
                calculated_aspect_ratio = w / float(h)
                print(f'Calculated ratio   : {calculated_aspect_ratio:,.2f}')

                # Selecting the coordinates that fulfills min required area for box and rectangle.
                if box:
                    if calculated_area > min_required_area and 0.8 <= calculated_aspect_ratio <= 1.2:
                        if calculated_area > selected_max_area:
                            selected_max_area = calculated_area
                            print(
                                f'Selected max-area : {selected_max_area:,.2f} '
                                f'at rated={(selected_max_area / image_area) * 100:.2f}%\n')
                            selected_max_contour = contour
                else:
                    if calculated_area > min_required_area and calculated_aspect_ratio > 1.2:
                        if calculated_area > selected_max_area:
                            selected_max_area = calculated_area
                            print(
                                f'Selected max-area : {selected_max_area:,.2f} '
                                f'at rated={(selected_max_area / image_area) * 100:.2f}%\n')
                            selected_max_contour = contour

    return selected_max_contour


def proportionate_close(image, dilate_iteration, erode_iteration):
    horizontal_se = np.array([
        [1, 1, 1]
    ])

    vertical_se = np.array([
        [1],
        [1]
    ])
    image = cv2.dilate(image, horizontal_se, iterations=dilate_iteration)
    image = cv2.erode(image, vertical_se, iterations=erode_iteration)
    return image


def pixel_percentage(image, color):
    return np.sum(image == color) / (image.shape[0] * image.shape[1]) * 100


def adjust_threshold(i, threshold_cnt, limit_cnt, min_threshold, threshold_inc_rate, black_pixels, white_pixels,
                     max_pixel_limit):
    if threshold_cnt == 3 or limit_cnt == 3 or (threshold_cnt + limit_cnt == 3) > 3:
        return min_threshold, threshold_cnt, limit_cnt, True

    if min_threshold > 254 or black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
        if min_threshold > 254:
            threshold_cnt = threshold_cnt + 1
        else:
            limit_cnt = limit_cnt + 1
        return min_threshold, threshold_cnt, limit_cnt, False
    else:
        if i > 1:
            thresh = ceil(min_threshold + (min_threshold * ((i - 1) * threshold_inc_rate)))
            if thresh > 254 or black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
                if thresh > 254:
                    threshold_cnt = threshold_cnt + 1
                else:
                    limit_cnt = limit_cnt + 1
            return thresh, threshold_cnt, limit_cnt, False

        else:
            return min_threshold, threshold_cnt, limit_cnt, False


def detect_barcode_v2(**kwargs):
    attempt_limit = 19
    max_pixel_limit = 97
    threshold_inc_rate = 0.2
    iteration_rate = 0.5

    min_threshold = int(kwargs['thresh_min'])
    # threshold_exceeded = False
    cropped = None
    thresh_exceeded = False

    p = None

    pre = preprocess_image(kwargs['image'], kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    image_size = kwargs['image'].shape[0] * kwargs['image'].shape[1]
    print(f'Image               : image-size={image_size:,}, [RXC]=[{pre.shape[0]:,} X {pre.shape[1]:,}]')

    current_threshold = kwargs['thresh_min']
    black_pixels = 0
    white_pixels = 0
    thresh_cnt = 0
    limit_cnt = 0
    for i in range(1, attempt_limit):
        # If thresh_exceeded, then the interation will continue with 230 or 245 until fails 3 times.

        if not thresh_exceeded:
            # current_threshold, thresh_exceeded = adjust_threshold(i, min_threshold, threshold_inc_rate)
            p = cvlib.binarize_inv(pre, current_threshold)

            black_pixels = ceil(pixel_percentage(p, BLACK))
            white_pixels = ceil(pixel_percentage(p, WHITE))

            # if (black_pixels > max_pixel_limit or white_pixels > max_pixel_limit
            #         or current_threshold > 254):
            #     current_threshold, thresh_cnt, limit_cnt, thresh_exceeded = adjust_threshold(i, thresh_cnt, limit_cnt,
            #                                                                                  min_threshold,
            #                                                                                  threshold_inc_rate,
            #                                                                                  black_pixels,
            #                                                                                  white_pixels,
            #                                                                                  max_pixel_limit)
            #
            #     print(f'{[i]} --Binarize      : threshold exceeded {max_pixel_limit:}% '
            #           f'with min-threshold={current_threshold:,}, [black={black_pixels:,.2f}%, '
            #           f'white={white_pixels:,.2f}%], cnt={thresh_cnt:,}. Adjusted={current_threshold:,} for re-attempt!')
            #
            #     # Adjusting the blown binarization with adjusted parameters.
            #     p = cvlib.binarize_inv(p, current_threshold)
            #
            #     # # Update pixel counts for accurate trace.
            #     # black_pixels = ceil(pixel_percentage(p, BLACK))
            #     # white_pixels = ceil(pixel_percentage(p, WHITE))
            # else:
            print(f'{[i]} --Binarize      : at min-thresh={current_threshold:,}, '
                  f'black={black_pixels:,.2f}%, '
                  f'white={white_pixels:,.2f}%, thresh-cnt={thresh_cnt:,}, limit-cnt={limit_cnt:,}')

            current_threshold, thresh_cnt, limit_cnt, thresh_exceeded = adjust_threshold(i, thresh_cnt, limit_cnt,
                                                                                         min_threshold,
                                                                                         threshold_inc_rate,
                                                                                         black_pixels,
                                                                                         white_pixels,
                                                                                         max_pixel_limit)


        else:
            if thresh_cnt == 3:
                print(f'{[i]} --Binarize(254): threshold exceeded {max_pixel_limit:}% '
                      f'with min-threshold={current_threshold:,}, [black={black_pixels:,.2f}%, '
                      f'white={white_pixels:,.2f}%], cnt={thresh_cnt:}, limit-cnt={limit_cnt:,}.')
            elif limit_cnt == 3:
                print(f'{[i]} --Binarize(97%): threshold exceeded {max_pixel_limit:}% '
                      f'with min-threshold={current_threshold:,}, [black={black_pixels:,.2f}%, '
                      f'white={white_pixels:,.2f}%], cnt={thresh_cnt:}, limit-cnt={limit_cnt:,}.')

        # # 4:1 (dilate:erosion)
        # iteration = kwargs['iteration']
        # iteration = ceil(iteration + (iteration * i * 0.1))
        # erode_iteration = ceil(iteration * iteration_rate)
        # print(
        #     f'[{i}] --Iteration : at iteration-rate={iteration_rate:,}, '
        #     f'dilate-interation={iteration:,}, '
        #     f'erode-interation={erode_iteration:,}')
        # print('---------->')
        #
        # p = proportionate_close(p, iteration, erode_iteration)
        #
        # contour = find_rectangle(source_img=kwargs['image'],
        #                          processed_img=p,
        #                          min_area_factor=kwargs['min_area_factor'],
        #                          box=kwargs['box'],
        #                          draw=False)
        # if contour is not None:
        #     cropped = crop_roi(kwargs['image'], contour, BLUE)
        #     break
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
