import threading

import cv2

import image_processor
from camera import Camera
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
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
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
        annotated_img, cropped = image_processor.detect_barcode(img)
        if cropped is not None:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: OK')
            self.process_callback(annotated_img, cropped)
        else:
            self.trace.write(f'[{threading.currentThread().native_id}] Detecting barcode: KO')
