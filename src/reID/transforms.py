import torchvision.transforms as tt
import cv2

class BGR2RGB:
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.shape[:2])
        p_left, p_top = [(max_wh - s) // 2 for s in image.shape[:2]]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.shape[:2], [p_left, p_top])]
        return cv2.copyMakeBorder(image, p_top, p_bottom, p_left, p_right, cv2.BORDER_CONSTANT, None, value = 0)


class Resize():
    def __init__(self, output_size=(128, 128)):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        return cv2.resize(image, self.output_size, interpolation = cv2.INTER_LINEAR)

class Transforms:
    def __init__(self):
        self.transform = tt.Compose([BGR2RGB(), 
                        SquarePad(),
                        Resize((128, 128)),
                        tt.ToTensor(),
                        tt.Normalize(0, 0.5)])

    def __call__(self, image):
        return self.transform(image)
    