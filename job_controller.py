import threading

from camera import Camera
from config import parameters
from ipcv import cvlib, scanner
from trace_handler import TraceHandler


class JobController:
    """
    Acknowledgement: No-image was taken from "https://www.flaticon.com/free-icons/no-image", and credit to sonnycandra.
    """

    NO_DISPLAY_IMG = 'resources/no-image.png'

    def __init__(self, device, frame_callback, trace_callback, process_callback, interval):
        self.camera = Camera(device, frame_callback, interval)
        # The default mode when application starts.
        self.auto_mode = True
        self.upload_mode = False
        self.trace_callback = trace_callback
        self.trace = TraceHandler(self.trace_callback)
        self.process_callback = process_callback
        self.trace_callback('Trace')

    def start(self):
        self.camera.start()

    def close(self):
        self.camera.close()

    def reset(self):
        self.camera.close()
        self.camera.start()

    def on_auto_mode(self):
        if not self.auto_mode:
            self.auto_mode = True
            self.upload_mode = False
            self.reset()
            self.trace.write(f'[{threading.currentThread().native_id}] Auto-request (ON): Accepted')
            return True
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Auto-request (ON): Denied')
            return False

    def off_auto_mode(self):
        if self.auto_mode:
            self.auto_mode = False
            self.reset()
            self.trace.write(f'[{threading.currentThread().native_id}] Auto-request (OFF): Accepted')
            return True
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Auto-request (OFF): Denied')
            return False

    def on_manual_upload(self):
        if not self.auto_mode or not self.upload_mode:
            self.upload_mode = True
            self.close()
            self.trace.write(f'[{threading.currentThread().native_id}] Upload-request: Accepted')
            return True
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Upload-request: Denied')
            return False

    def load_image(self, file_path):
        try:
            # IMREAD_UNCHANGED - loads alpha channel.
            # img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            # Convert the image from BGR to RGB
            img = cvlib.load_image(file_path)
            if img is None:
                self.trace.write(
                    f"[{threading.currentThread().native_id}] Error: Couldn't read image from {file_path}!")
                return None
            self.trace.write(f"[{threading.currentThread().native_id}] Selected File: {file_path}")
            return img
        except Exception as e:
            self.trace.write(f"[{threading.currentThread().native_id}] Error: {e}")
            return None

    def process_image(self, img):
        try:
            cropped = None
            self.trace.write(f'\n[{threading.currentThread().native_id}] <<<< Processing-Start >>>>')
            b_box = self.process_barcode(img)
            q_box = self.process_qrcode(img)

            # TODO: To process the image annotation based on contours.
            if b_box is not None and q_box is not None:
                # Both is present.
                self.trace.write(
                    f'[{threading.currentThread().native_id}] Processing detected both barcode & qrcode ...')
                scanner.draw_bounding_box2(img, b_box, scanner.GREEN)
                scanner.draw_bounding_box2(img, q_box, scanner.BLUE)
                cropped = img
                # TODO: Bigger box or intersection need to be calculated. For now, the whole image can be shown.
            elif b_box is None and q_box is None:
                # Both is absent - sending no-image to UI for displaying.
                cropped = cvlib.load_image(self.NO_DISPLAY_IMG)
                self.trace.write(
                    f'[{threading.currentThread().native_id}] Both barcode & qrcode is not detected!')
            else:
                # Either is present.
                if b_box is not None:
                    self.trace.write(f'[{threading.currentThread().native_id}] Processing detected barcode ...')
                    scanner.draw_bounding_box2(img, b_box, scanner.GREEN)
                    cropped = scanner.crop_roi2(img, b_box)
                    if cropped is not None:
                        code = scanner.decode_barcode(cropped)
                        self.trace.write(f'[{threading.currentThread().native_id}] Decoded barcode : {code}')
                if q_box is not None:
                    self.trace.write(f'[{threading.currentThread().native_id}] Processing detected qrcode ...')
                    scanner.draw_bounding_box2(img, b_box, scanner.BLUE)
                    cropped = scanner.crop_roi2(img, b_box)
                    if cropped is not None:
                        code = scanner.decode_qrcode(cropped)
                        self.trace.write(f'[{threading.currentThread().native_id}] Decoded barcode : {code}')

            if cropped is None:
                # This should not happen.
                raise ValueError('Something went wrong during cropping!\n')

            # Send the processed image and cropped images for UI display.
            self.process_callback(img, cropped)

            self.trace.write(f'[{threading.currentThread().native_id}] <<<< Processing-End >>>>\n')

        except Exception as e:
            self.trace.write(f"[{threading.currentThread().native_id}] Error: {e}")

    def process_barcode(self, img):
        self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode ...')
        # The copy is used in subsequent finding when barcode was not detected.
        copy = img.copy()

        # TODO: Integrated with barcode-detection v3.
        # box = scanner.detect_barcode(image=copy,
        #                              gamma=0.5,
        #                              gaussian_ksize=(15, 15),
        #                              gaussian_sigma=2,
        #                              avg_ksize1=(9, 9),
        #                              avg_ksize2=(3, 3),
        #                              thresh_min=200,
        #                              dilate_kernel=(21, 7),
        #                              dilate_iteration=4,
        #                              shrink_factor=6,
        #                              offset=0)

        box, p = scanner.detect_barcode_v3(copy, **parameters.barcode_general)
        cnt = 0
        detected = False
        ksize = (21, 7)
        if box is not None:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: OK')
        else:
            # Few attempts before failing.
            for i in range(1, 50, 5):
                ksize = (21 + i, 7)
                copy = img.copy()
                self.trace.write(
                    f'[{threading.currentThread().native_id}] Detecting barcode: ksize={ksize},attempt=[{cnt} ...]')
                box = scanner.detect_barcode(image=copy,
                                             gamma=0.5,
                                             gaussian_ksize=(15, 15),
                                             gaussian_sigma=2,
                                             avg_ksize1=(9, 9),
                                             avg_ksize2=(3, 3),
                                             thresh_min=200,
                                             dilate_kernel=ksize,
                                             dilate_iteration=4,
                                             shrink_factor=6,
                                             offset=0)
                if box is not None:
                    detected = True
                    break

                cnt = cnt + 1

            if detected:
                self.trace.write(
                    f'[{threading.currentThread().native_id}] Detecting barcode: OK, ksize={ksize},attempt=[{cnt}]')
            else:
                self.trace.write(
                    f'[{threading.currentThread().native_id}] Detecting barcode: KO, ksize={ksize},attempt=[{cnt}]')

        return box

    def process_qrcode(self, img):
        self.trace.write(f'[{threading.currentThread().native_id}] Detecting qr-code ...')
        box = scanner.detect_qrcode(img.copy(),
                                    gamma=0.1,
                                    gaussian_ksize=(3, 3),
                                    gaussian_sigma=2,
                                    thresh_min=128,
                                    box=True,
                                    min_area_factor=0.02)

        if box is not None:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting qr-code: OK')
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting qr-code: KO')

        return box

    def process_barcode_v2(self, img):
        """
        This function has adjustment using barcode detection using cv library. Therefore, this won't be used.
        :param img:
        :return:
        """
        self.trace.write(f'[{threading.currentThread().native_id}] Processing barcode ...')
        # The copy is used in subsequent finding when barcode was not detected.
        copy = img.copy()
        cropped, contour = scanner.detect_barcode(image=img,
                                                  gamma=0.5,
                                                  gaussian_ksize=(15, 15),
                                                  gaussian_sigma=2,
                                                  avg_ksize1=(9, 9),
                                                  avg_ksize2=(3, 3),
                                                  thresh_min=200,
                                                  dilate_kernel=(21, 7),
                                                  dilate_iteration=4,
                                                  shrink_factor=6,
                                                  offset=0)
        if cropped is not None:
            cnt = 0
            decoded = False
            ksize = (21, 7)
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: OK')

            barcode = scanner.decode_barcode(cropped)
            if barcode is not None:
                decoded = True
            else:
                for i in range(1, 50, 5):
                    ksize = (21 + i, 7)
                    img = copy.copy()
                    self.trace.write(
                        f'[{threading.currentThread().native_id}] Detecting attempt [{cnt}] Increasing SE kernel {ksize} ...')
                    cropped, contour = scanner.detect_barcode(image=img,
                                                              gamma=0.5,
                                                              gaussian_ksize=(15, 15),
                                                              gaussian_sigma=2,
                                                              avg_ksize1=(9, 9),
                                                              avg_ksize2=(3, 3),
                                                              thresh_min=200,
                                                              dilate_kernel=ksize,
                                                              dilate_iteration=4,
                                                              shrink_factor=6,
                                                              offset=0)
                    barcode = scanner.decode_barcode(cropped)
                    if barcode is not None:
                        decoded = True
                        break

                    cnt = cnt + 1

            if decoded:
                self.trace.write(f'[{threading.currentThread().native_id}] Decoded: [{barcode}] using ksize={ksize}')
            else:
                self.trace.write(
                    f'[{threading.currentThread().native_id}] No barcode detected at attempt [{cnt}] using max ksize={ksize}!')

            # self.process_callback(img, cropped)
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Processing barcode: KO')

        return img, cropped
