import cv2

from camera import Camera
from trace_handler import TraceHandler


class JobController:
    def __init__(self, device, frame_callback, trace_callback, interval):
        self.camera = Camera(device, frame_callback, interval)
        # The default mode when application starts.
        self.auto_mode = True
        self.upload_mode = False
        self.trace_callback = trace_callback
        self.trace = TraceHandler(self.trace_callback)

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
            self.trace.write(f'Auto-request (ON): Accepted')
            return True
        else:
            self.trace.write(f'Auto-request (ON): Denied')
            return False

    def off_auto_mode(self):
        if self.auto_mode:
            self.auto_mode = False
            self.reset()
            self.trace.write(f'Auto-request (OFF): Accepted')
            return True
        else:
            self.trace.write(f'Auto-request (OFF): Denied')
            return False

    def on_manual_upload(self):
        if not self.auto_mode or not self.upload_mode:
            self.upload_mode = True
            self.close()
            return True
        else:
            return False

    def load_image(self, file_path):
        try:
            # IMREAD_UNCHANGED - loads alpha channel.
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error: Couldn't read image from {file_path}!")
                return None
            return img
        except Exception as e:
            print(f"Error: {e}")
            return None
