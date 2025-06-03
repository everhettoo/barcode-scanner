"""
This module holds set of tested parameters used for barcode and qrcode.
"""

"""
Set of parameters used in barcode detection version 2. This version uses a lot of parameters to control the 
threshold during morphing processes by using series of identified structuring element to not exceed single 
pixel intensity threshold.

- wide_and_high     : Uses a wider horizontal structuring element with higher iterations.
- wide_and_low      : Uses a wider horizontal structuring element with lower iterations.
- narrow_and_high   : Uses a narrower horizontal structuring element with higher iterations.
- general           : Uses the structuring element identified in tested theory: horizontal=3, vertical=2 with 
                      2:1 iteration rate for dilation and erosion.
"""

v2_wide_and_high1 = {'gamma': 1,
                     'gaussian_ksize': (3, 3),
                     'gaussian_sigma': 1,
                     'min_threshold': 80,
                     'threshold_rate': 0.2,
                     'max_pixel_limit': 97,
                     'attempt_limit': 3,
                     'dilate_iteration': 10,
                     'erode_iteration': 5,
                     'dilate_size': 10,
                     'erode_size': 1,
                     'min_area_factor': 0.04,
                     'box': False}

v2_wide_and_high2 = {'gamma': 1,
                     'gaussian_ksize': (3, 3),
                     'gaussian_sigma': 1,
                     'min_threshold': 80,
                     'threshold_rate': 0.2,
                     'max_pixel_limit': 97,
                     'attempt_limit': 3,
                     'dilate_iteration': 10,
                     'erode_iteration': 5,
                     'dilate_size': 20,
                     'erode_size': 1,
                     'min_area_factor': 0.04,
                     'box': False}

v2_wide_and_low1 = {'gamma': 1,
                    'gaussian_ksize': (3, 3),
                    'gaussian_sigma': 1,
                    'min_threshold': 100,
                    'threshold_rate': 0.00001,
                    'max_pixel_limit': 100,
                    'attempt_limit': 10,
                    'dilate_iteration': 2,
                    'erode_iteration': 1,
                    'dilate_size': 10,
                    'erode_size': 1,
                    'min_area_factor': 0.04,
                    'box': False}

v2_narrow_and_high1 = {'gamma': 0.8,
                       'gaussian_ksize': (3, 3),
                       'gaussian_sigma': 1,
                       'min_threshold': 80,
                       'threshold_rate': 0.2,
                       'max_pixel_limit': 97,
                       'attempt_limit': 5,
                       'dilate_iteration': 8,
                       'erode_iteration': 4,
                       'dilate_size': 3,
                       'erode_size': 1,
                       'min_area_factor': 0.04,
                       'box': False}

v2_general1 = {'gamma': 1,
               'gaussian_ksize': (3, 3),
               'gaussian_sigma': 1,
               'min_threshold': 110,
               'threshold_rate': 0.00001,
               'max_pixel_limit': 100,
               'attempt_limit': 100,
               'dilate_iteration': 2,
               'erode_iteration': 1,
               'dilate_size': 3,
               'erode_size': 2,
               'min_area_factor': 0.001,
               'box': False}

"""
After experimentation with version 2 of barcode detection that uses very specialized set of parameters, two types of 
parameters were identified in version 3 to serve wider purposes.

- general       : Uses the structuring element identified in tested theory: horizontal=3, vertical=2 with 
                  2:1 iteration rate for dilation and erosion.
- special       : Uses a little wider structuring element as backed by theory (a close operation with dilation 
                  to erosion with 2:1 proportion).
"""

v3_general = {'gamma': 1,
              'gaussian_ksize': (3, 3),
              'gaussian_sigma': 1,
              'min_threshold': 120,
              'max_pixel_limit': 100,
              'attempt_limit': 30,
              'dilate_iteration': 4,
              'erode_iteration': 2,
              'dilate_size': 3,
              'erode_size': 1,
              'min_area_factor': 0.04,
              'box': False}

v3_special = {'gamma': 1,
              'gaussian_ksize': (3, 3),
              'gaussian_sigma': 1,
              'min_threshold': 120,
              'max_pixel_limit': 100,
              'attempt_limit': 20,
              'dilate_iteration': 2,
              'erode_iteration': 1,
              'dilate_size': 6,
              'erode_size': 2,
              'min_area_factor': 0.03,
              'box': False}

"""
The version 3 was experimented to improve barcode and qr-code detection based on segmentation using a threshold.
"""
v4_general = {'gamma': 0.5,
              'gaussian_ksize': (15, 15),
              'gaussian_sigma': 2,
              'min_threshold': 200,
              'max_pixel_limit': 100,
              'attempt_limit': 50,
              'dilate_iteration': 4,
              'erode_iteration': 2,
              'dilate_size': 4,
              'erode_size': 1,
              'min_area_factor': 0.03,
              'box': False}
