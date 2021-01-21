from models import YOLOLossSpecification
from models.originals import YOLOv3, YOLOv3PredictorConfiguration


class RPYOLOv3(YOLOv3):
    def __init__(self, bounding_boxes_per_cell=3,
                 large_predictor_config=YOLOv3PredictorConfiguration(
                     anchor_boxes=[112, 158, 193, 86, 205, 247],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 medium_predictor_config=YOLOv3PredictorConfiguration(
                     anchor_boxes=[59, 21, 84, 97, 95, 46],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 small_predictor_config=YOLOv3PredictorConfiguration(
                     anchor_boxes=[19, 37, 35, 57, 52, 78],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False))):
        super().__init__(class_count=120, large_predictor_config=large_predictor_config,
                         medium_predictor_config=medium_predictor_config, small_predictor_config=small_predictor_config,
                         bounding_boxes_per_cell=bounding_boxes_per_cell)
