import logging
from typing import List

from nobi import dirs
from nobi.ai.inference.objects.interface import ObjectDetector
from nobi.ai.inference.objects.model import DetectedObject, BoundingBox
from nobi.ai.inference.shared.model import CameraImage
from nobi.util.benchmark import Benchmark
import darknet

logger = logging.getLogger(__name__)

YOLOV4_MODEL_NAME = 'retrained-yolov4'
YOLOV4_MODEL_DIRECTORY = dirs.AI_MODEL_DIRECTORY / 'yolov4'


class YoloV4(ObjectDetector):

    def __init__(self, use_cuda=True):
        super().__init__(use_cuda)
        self.config_file = YOLOV4_MODEL_DIRECTORY / f'{YOLOV4_MODEL_NAME}.cfg'
        self.data_file = YOLOV4_MODEL_DIRECTORY / f'{YOLOV4_MODEL_NAME}.data'
        self.weights_file = YOLOV4_MODEL_DIRECTORY / f'{YOLOV4_MODEL_NAME}.weights'

        self.detection_benchmark = Benchmark()
        self.class_names = None 
        self.classcolors = None
        self.model = None
        self.width = None
        self.height = None

        self.darknet_images = []

    def start(self):
        assert self.config_file.exists()
        assert self.weights_file.exists()

        # INIT NET
        self.model, self.class_names, self.classcolors = darknet.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            batch_size=1
        )
        logger.debug(f'Loaded YoloV4 network with configuration:\n{self.model.string_representation}')
        self.width = darknet.network_width(network)
        self.height = darknet.network_height(network)

    def process(self, camera_images: List[CameraImage]) -> List[DetectedObject]:

        images = self.__preprocess(camera_images)

        with self.detection_benchmark:
            all_detections = self.__infer(images)

        return self.__postprocess(camera_images, all_detections)

    def __preprocess(self, camera_images: List[CameraImage]):
        import cv2
        # Reshape to the models input dimensions
        images = [cv2.resize(img.image, (self.width, self.height), interpolation=cv2.INTER_LINEAR) for img in camera_images]
        for image in images:
            img_for_detect = darknet.make_image(self.width, self.height, 3)
            darknet.copy_image_from_bytes(img_for_detect, image.tobytes())
            self.darknet_images.append(img_for_detect)
            darknet.free_image(img_for_detect)

        return self.darknet_images

    def __infer(self, images):
        all_detections = []
        for image in images:
            detections = darknet.detect_image(self.model, self.class_names, image, 0.1)
            all_detections.append(detections)
            return all_detections

    def __postprocess(self, camera_images: List[CameraImage], all_detections: List[List]) -> List[DetectedObject]:
        detected_objects = []
        for image_idx in range(len(all_detections)):
            image_detections = all_detections[image_idx]
            for detection in image_detections:
                camera_image = camera_images[image_idx]
                label, confidence, (x0, y0, x1, y1) = detection

                x0, x1 = x0 / self.width * camera_image.width, x1 / self.width * camera_image.width
                y0, y1 = y0 / self.height * camera_image.height, y1 / self.height * camera_image.height

                detected_objects.append(DetectedObject(
                    original_image=camera_image,  # CameraImage
                    coco_class_idx=1,  # int
                    bounding_box=BoundingBox.from_floats(x0, y0, x1, y1),  # BoundingBox
                    confidence=int(round(100 * confidence))  # int
                ))

        return detected_objects
