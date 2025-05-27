import threading

from camera import Camera
from ipcv import cvlib, scanner
from trace_handler import TraceHandler


class JobController:
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
            self.trace.write(f'\n[{threading.currentThread().native_id}] <<<< Processing-Start >>>>')
            self.process_barcode(img)
            self.trace.write(f'[{threading.currentThread().native_id}] <<<< Processing-End >>>>\n')

        except Exception as e:
            self.trace.write(f"[{threading.currentThread().native_id}] Error: {e}")

    def process_barcode(self, img):
        self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode ...')

        cropped = scanner.detect_barcode(image=img,
                                         gamma=0.5,
                                         gaussian_ksize=(15, 15),
                                         gaussian_sigma=2,
                                         avg_ksize1=(9, 9),
                                         avg_ksize2=(3, 3),
                                         thresh_min=200,
                                         dilate_kernel=(21, 7),
                                         dilate_iteration=4,
                                         shrink_factor=6)

        if cropped is not None:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: OK')
            barcode = scanner.decode_barcode(cropped)
            if barcode is not None:
                self.trace.write(f'[{threading.currentThread().native_id}] Decoded: {barcode}')
            self.process_callback(img, cropped)
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: KO')
