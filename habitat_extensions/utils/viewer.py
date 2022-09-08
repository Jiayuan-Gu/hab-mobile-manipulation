import cv2
import numpy as np


class OpenCVViewer:
    def __init__(self, name="OpenCVViewer", exit_on_escape=True):
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        self.exit_on_escape = exit_on_escape

    def imshow(self, image: np.ndarray, rgb=True, non_blocking=False, delay=0):
        if image.ndim == 2:
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.ndim == 3, image.shape

        if rgb:
            image = image[..., ::-1]
        cv2.imshow(self.name, image)

        if non_blocking:
            return
        else:
            key = cv2.waitKey(delay)
            if key == 27:  # escape
                if self.exit_on_escape:
                    exit(0)
                else:
                    return None
            elif key == -1:  # timeout
                pass
            else:
                return chr(key)

    def close(self):
        cv2.destroyWindow(self.name)

    def __del__(self):
        self.close()
