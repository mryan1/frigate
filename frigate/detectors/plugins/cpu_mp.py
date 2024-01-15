import logging

import mediapipe as mp
import numpy as np
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)

DETECTOR_KEY = "mediapipe"


class MediaPipeDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]


class CpuMp(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: MediaPipeDetectorConfig):
        self.model_path = detector_config.model.path
        self.BaseOptions = mp.tasks.BaseOptions
        self.ObjectDetector = mp.tasks.vision.ObjectDetector
        self.ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # TODO: define max_results in config
        self.options = self.ObjectDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            max_results=5,
            running_mode=self.VisionRunningMode.IMAGE,
            score_threshold=0.2,
        )

    def detect_raw(self, tensor_input):
        tensor_input_reshaped = tensor_input[0]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=tensor_input_reshaped
        )

        with self.ObjectDetector.create_from_options(self.options) as detector:
            detection_result = detector.detect(mp_image).detections

        detections = np.zeros((20, 6), np.float32)

        i = 0
        for detection in detection_result:
            # logger.info(detection.categories)
            # TODO: get proper label id

            x = detection.bounding_box.origin_x / tensor_input_reshaped.shape[1]
            y = detection.bounding_box.origin_y / tensor_input_reshaped.shape[0]
            width = detection.bounding_box.width / tensor_input_reshaped.shape[1]
            height = detection.bounding_box.height / tensor_input_reshaped.shape[0]

            top = 1 - y - height
            left = x
            bottom = 1 - y
            right = x + width
            score = float(detection.categories[0].score)
            # TODO: add proper score below
            detections[i] = [
                float(15),
                0.8,
                top,
                left,
                bottom,
                right,
            ]
            i += 1

        # logger.info(detections)
        return detections
