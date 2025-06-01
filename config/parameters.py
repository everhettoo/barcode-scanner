# Working for 101, 102, 140
barcode_wide_and_high1 = {'gamma': 1,
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

# Working for 111, 112, 113 (finalized)
barcode_wide_and_high2 = {'gamma': 1,
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

# Working for 120
barcode_narrow_and_high1 = {'gamma': 0.8,
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

# Working for 150 (finalized)
barcode_ps_charles = {'gamma': 1,
                      'gaussian_ksize': (3, 3),
                      'gaussian_sigma': 1,
                      'min_threshold': 80,
                      'threshold_rate': 0.1,
                      'max_pixel_limit': 97,
                      'attempt_limit': 10,
                      'dilate_iteration': 10,
                      'erode_iteration': 5,
                      'dilate_size': 10,
                      'erode_size': 5,
                      'min_area_factor': 0.04,
                      'box': False}

# Working for 160 (finalized) close(25,25)
barcode_general1 = {'gamma': 1,
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

# Works for 150
barcode_general2 = {'gamma': 1,
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

# 120
# image = cvlib.load_image('resources/barcode/barcode101.png')
# image = cvlib.load_image('resources/barcode/barcode102.png')
# image = cvlib.load_image('resources/barcode/barcode111.png')
# image = cvlib.load_image('resources/barcode/barcode112.png')
# image = cvlib.load_image('resources/barcode/barcode113.png')
# image = cvlib.load_image('resources/barcode/barcode114.png')
# image = cvlib.load_image('resources/barcode/barcode120.jpg')
# Img-140: Needs threshold=100
# image = cvlib.load_image('resources/barcode/barcode140.jpg')
# image = cvlib.load_image('resources/barcode/barcode500.jpg')
barcode_general = {'gamma': 1,
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

# Not working with general-3
# image = cvlib.load_image('resources/barcode/barcode150.jpg')

# Img-160: threshold=100
# image = cvlib.load_image('resources/barcode/barcode160.jpg')
# Img-170: threshold=80
# image = cvlib.load_image('resources/barcode/barcode170.jpg')
barcode_special = {'gamma': 1,
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
