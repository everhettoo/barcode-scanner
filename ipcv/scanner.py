import math
from math import ceil

import cv2
import numpy as np

from ipcv import cvlib

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = 0
WHITE = 255

MIN_THRESHOLD_LIMIT = 220

calculate_threshold = lambda t, i, d: ceil(t + (t * i * d))


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


def draw_bounding_box(image, contour, color):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    cv2.drawContours(image, [approx], -1, color, 3)


def draw_bounding_box2(image, box, color):
    cv2.drawContours(image, [box], -1, color, 3)


def crop_roi2(image, box):
    (x, y, w, h) = cv2.boundingRect(box)
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


def find_rectangle(source_img, processed_img, min_area_factor, cnt, box=False, draw=False, verbose=False):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = processed_img.shape[0] * processed_img.shape[1]
    min_required_area = image_area * min_area_factor
    print(
        f'[{cnt}] --Contour       : found={len(contours)}; min-req-area={min_required_area:,.2f} '
        f'({min_area_factor} at rate); area should be > {(min_required_area / image_area) * 100:.2f}%')

    selected_max_contour = None
    selected_max_area = 0

    # Epsilon: is maximum distance from contour to approximated contour.
    # eps = 0.04
    eps = 0.0391
    # eps = 0.05

    curr_cnt = 1
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

            # Retrieve the four coordinates (of a most-likely rectangle) to verify if it is a rectangle.
            [a, b, c, d] = rectangle_coordinates(approx)
            if verbose:
                print(f'[c:{curr_cnt}] --Contour     : calculated area={calculated_area:,.2f}, '
                      f'coordinates (X,Y)=[A=({a}), B=({b}), C=({c}), D=({d})]')

            if (is_perpendicular_angle(calculate_angle(a, b, c))
                    and is_perpendicular_angle(calculate_angle(b, c, d))
                    and is_perpendicular_angle(calculate_angle(c, d, a))
                    and is_perpendicular_angle(calculate_angle(d, a, b))):

                (x, y, w, h) = cv2.boundingRect(approx)
                calculated_aspect_ratio = w / float(h)
                if verbose:
                    print(f'[c:{curr_cnt}] --Contour     : calculated ratio={calculated_aspect_ratio:,.2f}')

                # Selecting the coordinates that fulfills min required area for box and rectangle.
                if box:
                    if calculated_area > min_required_area and 0.8 <= calculated_aspect_ratio <= 1.2:
                        if calculated_area > selected_max_area:
                            selected_max_area = calculated_area
                            print(
                                f'[c:{curr_cnt}] --Contour(box) : selected max-area={selected_max_area:,.2f} '
                                f'at rate={(selected_max_area / image_area) * 100:.2f}%')
                            selected_max_contour = contour
                else:
                    if calculated_area > min_required_area and calculated_aspect_ratio > 1.2:
                        if calculated_area > selected_max_area:
                            selected_max_area = calculated_area
                            print(
                                f'[c:{curr_cnt}] --Contour     : selected max-area={selected_max_area:,.2f} '
                                f'at rate={(selected_max_area / image_area) * 100:.2f}%')
                            selected_max_contour = contour
            curr_cnt += 1
            if verbose:
                print('\r')
    return selected_max_contour


def proportionate_close(image, dilate_iteration, erode_iteration, dilate_size, erode_size):
    # horizontal_se = np.array([
    #     [1, 1, 1]
    # ])
    #
    # vertical_se = np.array([
    #     [1],
    #     [1]
    # ])

    horizontal_se = np.ones((1, dilate_size)).astype('uint8')
    vertical_se = np.ones((erode_size, 1)).astype('uint8')

    image = cv2.dilate(image, horizontal_se, iterations=dilate_iteration)
    image = cv2.erode(image, vertical_se, iterations=erode_iteration)
    return image


def pixel_percentage(image, color):
    return np.sum(image == color) / (image.shape[0] * image.shape[1]) * 100


def adjust_threshold(i, threshold_min, rate, black_pixels, white_pixels, max_pixel_limit):
    if threshold_min > MIN_THRESHOLD_LIMIT:
        return MIN_THRESHOLD_LIMIT, True
    elif black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
        if white_pixels > max_pixel_limit:
            # Reduce 20%
            thresh = calculate_threshold(threshold_min, i, rate)
            thresh = thresh - (thresh * 0.2)
            return thresh, True
        else:
            # Increase 20%
            thresh = calculate_threshold(threshold_min, i, rate)
            thresh = thresh + (thresh * 0.2)
            return thresh, True
    else:
        # thresh = ceil(threshold_min + (threshold_min * i * rate))
        thresh = calculate_threshold(threshold_min, i, rate)
        if thresh > MIN_THRESHOLD_LIMIT:
            return MIN_THRESHOLD_LIMIT, True
        else:
            return thresh, False


def detect_barcode_v3(image, **kwargs):
    max_pixel_limit = int(kwargs['max_pixel_limit'])
    min_threshold = int(kwargs['min_threshold'])
    cropped = None
    thresh_exceeded = False
    cnt = 0
    # To prevent repetition of warning.
    displayed_warning = False
    # Assign current-threshold for processing with config-threshold.
    curr_threshold = int(kwargs['min_threshold'])
    # Current processed image.
    p = None

    pre = preprocess_image(image, kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    image_size = image.shape[0] * image.shape[1]
    print(f'Info                : size={image_size:,}, RxC=[{pre.shape[0]:,}x{pre.shape[1]:,}], '
          f'box-ratio-on: {kwargs["box"]}, attempt-limit: {kwargs["attempt_limit"]}')

    p = cvlib.binarize_inv(pre, curr_threshold)

    for i in range(1, int(kwargs['attempt_limit']) + 1):
        cnt = i
        print(f'----------> [BEGIN, attempt={cnt}]')

        # Pixel-ratio checking:
        black_pixels = pixel_percentage(p, BLACK)
        white_pixels = pixel_percentage(p, WHITE)
        if black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] Exceeded {max_pixel_limit:}% limit!')
            break
        else:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] within {max_pixel_limit:}% limit.')

        print(f'[{i}] --Morphing      : dilation:erosion=[{kwargs["dilate_iteration"]}:{kwargs["erode_iteration"]}]')

        p = proportionate_close(p, kwargs["dilate_iteration"], kwargs["erode_iteration"],
                                kwargs['dilate_size'], kwargs['erode_size'])

        p = cvlib.morph_open(p, (45, 45))

        contour = find_rectangle(source_img=image,
                                 processed_img=p,
                                 min_area_factor=kwargs['min_area_factor'],
                                 cnt=cnt,
                                 box=kwargs['box'],
                                 draw=False,
                                 verbose=False)
        if contour is not None:
            cropped = crop_roi(image, contour, GREEN)
            break

    # for i in range(1, int(kwargs['attempt_limit']) + 1):
    #     print(f'[{i}] --Closing       : dilation:erosion=[{kwargs["dilate_iteration"]}:{kwargs["erode_iteration"]}]')
    #     # p = cvlib.morph_dilate(p, kwargs["dilate_iteration"], (3, 3))
    #     p = cvlib.morph_open(p, (25, 25))
    #
    #     contour = find_rectangle(source_img=image,
    #                              processed_img=p,
    #                              min_area_factor=kwargs['min_area_factor'],
    #                              cnt=cnt,
    #                              box=kwargs['box'],
    #                              draw=False,
    #                              verbose=False)
    #     if contour is not None:
    #         cropped = crop_roi(image, contour, GREEN)
    #         break

    print(f'----------> [END, attempt={cnt}/{kwargs["attempt_limit"]}]\n')
    return cropped, p


def detect_barcode_v2(image, **kwargs):
    max_pixel_limit = int(kwargs['max_pixel_limit'])
    min_threshold = int(kwargs['min_threshold'])
    cropped = None
    thresh_exceeded = False
    cnt = 0
    # To prevent repetition of warning.
    displayed_warning = False
    # Assign current-threshold for processing with config-threshold.
    curr_threshold = int(kwargs['min_threshold'])
    # Current processed image.
    p = None

    pre = preprocess_image(image, kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    image_size = image.shape[0] * image.shape[1]
    print(f'Info                : size={image_size:,}, RxC=[{pre.shape[0]:,}x{pre.shape[1]:,}], '
          f'box-ratio-on: {kwargs["box"]}, attempt-limit: {kwargs["attempt_limit"]}')

    for i in range(1, int(kwargs['attempt_limit']) + 1):
        cnt = i
        print(f'----------> [BEGIN, attempt={cnt}]')
        if not thresh_exceeded:
            # Binarize image using the threshold (initially from config, subsequently using calculated).
            p = cvlib.binarize_inv(pre, curr_threshold)

            # Get pixel values in % after binarization.
            black_pixels = pixel_percentage(p, BLACK)
            white_pixels = pixel_percentage(p, WHITE)

            # No logic processing, only for displaying.
            if black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
                print(
                    f'{[i]} --Binarize      : pixel ratio exceeded limit {max_pixel_limit:}% with min-threshold={curr_threshold}!'
                    f' [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%]')
            elif curr_threshold > MIN_THRESHOLD_LIMIT:
                print(
                    f'{[i]} --Binarize      : min-threshold={curr_threshold} exceeded limit ({MIN_THRESHOLD_LIMIT})!\n'
                    f'[black={black_pixels:,.2f}%, white={white_pixels:,.2f}%].')
            else:
                print(f'{[i]} --Binarize      : at min-thresh={curr_threshold:,}, '
                      f'[black={black_pixels:,.2f}%, white={white_pixels:,.2f}%]')

            # Pass the config-threshold to obtain the calculated current-threshold for processing.
            # On breaching threshold limit, revised threshold is made for correction before flagging threshold off.
            curr_threshold, thresh_exceeded = adjust_threshold(i,
                                                               int(kwargs['min_threshold']),
                                                               kwargs["threshold_rate"],
                                                               black_pixels,
                                                               white_pixels,
                                                               max_pixel_limit)

            if thresh_exceeded:
                calculated_limit = calculate_threshold(min_threshold, i - 1, kwargs["threshold_rate"])
                # Display the descriptive warning message once.
                if not displayed_warning:
                    if curr_threshold == MIN_THRESHOLD_LIMIT:
                        print(
                            f'{[i]} --Binarize      : min-threshold={calculated_limit} exceeded limit ({MIN_THRESHOLD_LIMIT})!\n'
                            f'                      indefinite-adjustment made to min-threshold={curr_threshold:} ...')
                    else:
                        print(
                            f'{[i]} --Binarize      : pixel ratio exceeded limit {max_pixel_limit:}% with min-threshold={calculated_limit}!\n'
                            f'                      indefinite-adjustment made to min-threshold={curr_threshold:} ...')

                    # Turn off to avoid repeating warning.
                    displayed_warning = True

                # Binarize image using corrected threshold before turning off binarization.
                print(f'{[i]} --Binarize      : reverting image with corrected min-thresh={curr_threshold:,}')
                p = cvlib.binarize_inv(pre, curr_threshold)

                # Skip morphing so in the next cycle the binarization will be corrected before turning it off.
                print(f'{[i]} --Binarize      : skipping attempt {cnt}!')
                continue

        # Pixel-ratio checking:
        black_pixels = pixel_percentage(p, BLACK)
        white_pixels = pixel_percentage(p, WHITE)
        if black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] Exceeded {max_pixel_limit:}% limit!')
            break
        else:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] within {max_pixel_limit:}% limit.')

        print(f'[{i}] --Morphing      : dilation:erosion=[{kwargs["dilate_iteration"]}:{kwargs["erode_iteration"]}]')

        p = proportionate_close(p, kwargs["dilate_iteration"], kwargs["erode_iteration"],
                                kwargs['dilate_size'], kwargs['erode_size'])

        contour = find_rectangle(source_img=image,
                                 processed_img=p,
                                 min_area_factor=kwargs['min_area_factor'],
                                 cnt=cnt,
                                 box=kwargs['box'],
                                 draw=False,
                                 verbose=False)
        if contour is not None:
            cropped = crop_roi(image, contour, GREEN)
            break

    print(f'----------> [END, attempt={cnt}/{kwargs["attempt_limit"]}]\n')
    return cropped, p


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
    cropped = None
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

    contour = cvlib.get_prominent_contour(p, kwargs['offset'])
    #
    # contour = find_rectangle(source_img=kwargs['image'],
    #                          processed_img=p,
    #                          min_area_factor=0.03,
    #                          cnt=1,
    #                          box=False,
    #                          draw=True,
    #                          verbose=True)
    # if contour is not None:
    #     cropped = crop_roi(kwargs['image'], contour, GREEN)

    return contour


def detect_qrcode(image, **kwargs):
    pre = preprocess_image(image, kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    binary = cv2.threshold(pre, kwargs['thresh_min'], 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    vertical = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
    horizontal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))
    rect = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    p = binary
    box = None
    for i in range(1, 3):
        p = cv2.dilate(p, vertical, iterations=3)
        p = cv2.dilate(p, horizontal, iterations=3)
        # p = cv2.morphologyEx(p, cv2.MORPH_OPEN, rect)
        p = cv2.erode(p, horizontal, iterations=2)
        p = cv2.erode(p, vertical, iterations=2)
        # p = cv2.dilate(p, rect, iterations=1)
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, rect)
        p = cv2.dilate(p, vertical, iterations=1)
        p = cv2.dilate(p, horizontal, iterations=1)
        # p = cv2.morphologyEx(p, cv2.MORPH_OPEN, rect)
        p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, rect)

        contour = find_rectangle(source_img=image,
                                 processed_img=p,
                                 min_area_factor=kwargs['min_area_factor'],
                                 cnt=i,
                                 box=kwargs['box'],
                                 draw=False,
                                 verbose=False)
        if contour is not None:
            rect = cv2.minAreaRect(contour)
            box = np.intp(cv2.boxPoints(rect))
            break
    return box


def decode_barcode(img):
    """
    This function is used only to verify the detected barcode.
    :param img: The cropped image with barcode detected.
    :return:
    """
    detector = cv2.barcode_BarcodeDetector()
    # decoded_text, points, barcode_type = detector.detectAndDecode(img)
    decoded_text, _, _ = detector.detectAndDecode(img)
    if decoded_text == '':
        return None

    return decoded_text


def decode_qrcode(img):
    """
    This function is used only to verify the detected qrcode.
    :param box: The coordinates of the bounding box.
    :param img: The cropped image with barcode detected.
    :return:
    """
    detector = cv2.QRCodeDetector()

    # Check if a QR code was detected
    decoded_text, _, _ = detector.detectAndDecode(img)
    if decoded_text == '':
        return None

    return decoded_text
