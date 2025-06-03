import threading

from ipcv.camera import Camera
from config import parameters
from ipcv import cvlib, scanner, imutil
from utility.trace_handler import TraceHandler


class JobController:
    """
    This is a controller to coordinate UI and image processing tasks.
    Acknowledgement: 'No-image' was taken from "https://www.flaticon.com/free-icons/no-image", and credit to sonnycandra.
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

            # Caters the three scenarios of processing barcode and qrcode together.
            if b_box is not None and q_box is not None:
                # Both is present.
                self.trace.write(
                    f'[{threading.currentThread().native_id}] Processing detected both barcode & qrcode ...')
                imutil.draw_bounding_box2(img, b_box, imutil.GREEN_COLOR)
                imutil.draw_bounding_box2(img, q_box, imutil.BLUE_COLOR)
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
                    imutil.draw_bounding_box2(img, b_box, imutil.GREEN_COLOR)
                    cropped = imutil.crop_roi2(img, b_box)
                    if cropped is not None:
                        code = scanner.decode_barcode(cropped)
                        self.trace.write(f'[{threading.currentThread().native_id}] Decoded barcode : {code}')
                if q_box is not None:
                    self.trace.write(f'[{threading.currentThread().native_id}] Processing detected qrcode ...')
                    imutil.draw_bounding_box2(img, b_box, imutil.BLUE_COLOR)
                    cropped = imutil.crop_roi2(img, b_box)
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
        """
        Processes the barcode-v3 using the scanner module.
        :param img: The image to process.
        :return: Returns the coordinates of the barcode.
        """
        self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode ...')
        box, p = scanner.detect_barcode_v4(img, **parameters.v4_general)
        if box is not None:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: OK')
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: KO')

        return box

    def process_qrcode(self, img):
        """
        Processes the qrcode using the scanner module.
        :param img: The image to process.
        :return: Returns the coordinates of the qrcode.
        """
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
