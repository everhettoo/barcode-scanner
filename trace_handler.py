class TraceHandler:
    def __init__(self, callback):
        self.callback = callback

    def write(self, msg: str):
        self.callback(msg)
        print(msg)
