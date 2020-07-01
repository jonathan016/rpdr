from models.originals import YOLOv2
from models import YOLOLossSpecification


class RPYOLOv2(YOLOv2):
    def __init__(self):
        # TODO Set anchor_boxes real values
        anchor_boxes = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

        super().__init__(class_count=120, anchor_boxes=anchor_boxes, bounding_boxes_per_cell=5,
                         spec=YOLOLossSpecification(version=2, num_classes=120, max_object=15,
                                                    anchor_box_learning_seen_images_limit=12500))
