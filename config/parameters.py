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
barcode_ps_charles = {'gamma': 0.6,
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
