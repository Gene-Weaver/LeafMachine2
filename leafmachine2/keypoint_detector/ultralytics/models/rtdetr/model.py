# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
RT-DETR model interface
"""
import os
import sys
import inspect

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir1 = os.path.dirname(os.path.dirname(currentdir))
parentdir2 = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
parentdir3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))))
sys.path.append(currentdir)
sys.path.append(parentdir1)
sys.path.append(parentdir2)
sys.path.append(parentdir3)

from leafmachine2.keypoint_detector.ultralytics.engine.model import Model
from leafmachine2.keypoint_detector.ultralytics.nn.tasks import RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    RTDETR model interface.
    """

    def __init__(self, model='rtdetr-l.pt') -> None:
        if model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt file or *.yaml file.')
        super().__init__(model=model, task='detect')

    @property
    def task_map(self):
        return {
            'detect': {
                'predictor': RTDETRPredictor,
                'validator': RTDETRValidator,
                'trainer': RTDETRTrainer,
                'model': RTDETRDetectionModel}}
    
if __name__ == '__main__':
    print("hi")
