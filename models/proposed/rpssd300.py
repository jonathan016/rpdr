from models.originals.ssd.internal_modules import PriorBoxesConfig
from models.originals import SSD300


class RPSSD300(SSD300):
    def __init__(self, load_pretrained_base=True):
        prior_boxes = PriorBoxesConfig().all(6)
        aspect_ratios = PriorBoxesConfig().all([1., 2., 3., 0.5, .333])

        super().__init__(num_classes=120, prior_boxes=prior_boxes, aspect_ratios=aspect_ratios,
                         load_pretrained_base=load_pretrained_base)
