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
        tensor_input_reshaped = tensor_input.reshape((720, 1280, 3))
        # logger.info(f"tensor_input.shape: {tensor_input.shape}")
        # logger.info(f"tensor_input reshaped {tensor_input_reshaped.shape}")

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=tensor_input_reshaped
        )
        # logger.info(f"mp_image.shape: {mp_image.width}x{mp_image.height}")

        with self.ObjectDetector.create_from_options(self.options) as detector:
            detection_result = detector.detect(mp_image).detections

        detections = np.zeros((20, 6), np.float32)

        i = 0
        for detection in detection_result:
            logger.info(detection.bounding_box)
            logger.info(detection.categories)
            # TODO: get proper label id
            x = float(detection.bounding_box.origin_x / tensor_input_reshaped.shape[0])
            y = float(detection.bounding_box.origin_y / tensor_input_reshaped.shape[1])
            width = float(detection.bounding_box.width / tensor_input_reshaped.shape[0])
            height = float(
                detection.bounding_box.height / tensor_input_reshaped.shape[1]
            )
            score = float(detection.categories[0].score)
            # TODO: add proper score below
            detections[i] = [
                float(1),
                0.8,
                x,
                y,
                x + width,
                y + height,
            ]
            i += 1
            logger.info(detections)

        # for i in range(count):
        #     if scores[i] < 0.4 or i == 20:
        #         break
        #     detections[i] = [
        #         class_ids[i],
        #         float(scores[i]),
        #         boxes[i][0],
        #         boxes[i][1],
        #         boxes[i][2],
        #         boxes[i][3],
        #     ]

        return detections
