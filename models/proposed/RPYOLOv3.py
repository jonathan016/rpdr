from models.originals import YOLOv3, YOLOv3PredictorConfiguration
from models import YOLOLossSpecification


class RPYOLOv3(YOLOv3):
    def __init__(self, bounding_boxes_per_cell=3,
                 large_predictor_config=YOLOv3PredictorConfiguration(
                     anchor_boxes=[19.56, 37.19, 35.13, 57.81, 52.90, 78.26],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 medium_predictor_config=YOLOv3PredictorConfiguration(
                     anchor_boxes=[59.87, 21.10, 84.39, 97.68, 95.15, 46.35],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False)),
                 small_predictor_config=YOLOv3PredictorConfiguration(
                     anchor_boxes=[112.70, 158.04, 193.72, 86.14, 205.25, 247.73],
                     spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False))):
        super().__init__(class_count=120, large_predictor_config=large_predictor_config,
                         medium_predictor_config=medium_predictor_config, small_predictor_config=small_predictor_config,
                         bounding_boxes_per_cell=bounding_boxes_per_cell)
