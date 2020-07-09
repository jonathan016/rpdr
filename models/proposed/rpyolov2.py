from models.originals import YOLOv2
from models import YOLOLossSpecification


class RPYOLOv2(YOLOv2):
    def __init__(self, anchor_boxes=None, bounding_boxes_per_cell=5,
                 spec=YOLOLossSpecification(version=2, num_classes=120, max_object=15,
                                            anchor_box_learning_seen_images_limit=12500)):
        if anchor_boxes is None:
            anchor_boxes = [1.07, 1.07, 1.55, 2.33, 3.12, 4.05, 3.25, 1.53, 6.32, 6.41]

        super().__init__(class_count=120, anchor_boxes=anchor_boxes, bounding_boxes_per_cell=bounding_boxes_per_cell,
                         spec=spec)
