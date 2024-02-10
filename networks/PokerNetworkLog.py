class PokerNetworkLog(dict):
    def __init__(self, dict):
        for key, value in dict.items():
            self[key] = value

    def __add__(self, other):
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other

        for type in self.keys():
            other_result = other[type]
            self[type] += other_result

        return self

    def __truediv__(self, value):
        for type in self.keys():
            self[type] /= value

        return self

    def log(self, writer, epoch, images=True):
        for type, log in self.items():
            log_items = log.get_logs()
            for name, value in log_items:
                writer.add_scalar(f"{type}/{name}", value, epoch)

            if images and epoch % 50 == 0:
                log_images = log.get_images()
                for name, img in log_images:
                    writer.add_figure(f"{type}/{name}", img, epoch)
