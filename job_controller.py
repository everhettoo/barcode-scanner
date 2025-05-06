from camera import Camera


class JobController:
    def __init__(self, device, frame_callback, interval):
        self.camera = Camera(device, frame_callback, interval)

    def start(self):
        self.camera.start()

    def stop(self):
        self.camera.close()
