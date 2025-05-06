from camera import Camera


class JobController:
    def __init__(self, device, frame_callback, interval):
        self.camera = Camera(device, frame_callback, interval)
        # The default mode when application starts.
        self.auto_mode = True

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
            self.reset()
            return True
        else:
            return False

    def off_auto_mode(self):
        if self.auto_mode:
            self.auto_mode = False
            self.reset()
            return True
        else:
            return False

    def on_manual_upload(self):
        if not self.auto_mode:
            self.close()
            return True
        else:
            return False
