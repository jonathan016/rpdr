from .originals import YOLOv3, YOLOv3PredictorConfiguration
from .modules.loss_modules import YOLOLossSpecification


class RPYOLOv3(YOLOv3):
    def __init__(self):
        # TODO Set anchor_boxes real values for all config
        large_predictor_config = YOLOv3PredictorConfiguration(
            anchor_boxes=[10, 13, 16, 30, 33, 23],
            spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False))
        medium_predictor_config = YOLOv3PredictorConfiguration(
            anchor_boxes=[30, 61, 62, 45, 59, 119],
            spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False))
        small_predictor_config = YOLOv3PredictorConfiguration(
            anchor_boxes=[116, 90, 156, 198, 373, 326],
            spec=YOLOLossSpecification(version=3, num_classes=120, max_object=15, is_multilabel=False))

        super().__init__(class_count=120, large_predictor_config=large_predictor_config,
                         medium_predictor_config=medium_predictor_config, small_predictor_config=small_predictor_config,
                         bounding_boxes_per_cell=3)
