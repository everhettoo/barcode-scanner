from camera import Camera


class JobController:
    def __init__(self, device, frame_callback, interval):
        self.camera = Camera(device, frame_callback, interval)
        # The default mode when application starts.
        self.auto_mode = True

    def start(self):
        self.camera.start()

    def stop(self):
        self.camera.close()

    def on_auto_mode(self):
        if not self.auto_mode:
            self.auto_mode = True
            return True
        else:
            return False

    def off_auto_mode(self):
        if self.auto_mode:
            self.auto_mode = False
            return True
        else:
            return False
