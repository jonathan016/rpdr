from .modules import PredictionConvolutions
from .originals import SSD300


class RPSSD300(SSD300):
    def __init__(self, load_pretrained_base=True):
        prior_boxes = PredictionConvolutions.PriorBoxesConfig(config={'conv4_3': 6,
                                                                      'conv7': 6,
                                                                      'conv8_2': 6,
                                                                      'conv9_2': 6,
                                                                      'conv10_2': 6,
                                                                      'conv11_2': 6})

        super().__init__(num_classes=120, prior_boxes=prior_boxes, load_pretrained_base=load_pretrained_base)
